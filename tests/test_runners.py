"""Runners end to end, with the OpenRouter transport replaced by a canned client.

Each runner is exercised through its ``main(argv)`` on the real core datasets
(``--max_items`` keeps it to a handful of prompts), so what is tested is the
same path a user hits: argument parsing, dataset selection, output layout,
the evaluator loop and the summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from fakes import FakeBackend

from anchorbench.inference import async_api
from anchorbench.runners import api, icl_dist_api, mitigation_baseline, sampling, tool


@pytest.fixture
def canned(monkeypatch):
    """Answer '42' to every request; record every conversation sent."""
    sent: list[list[dict]] = []

    async def fake_query(self, model_id, messages_list, max_tokens=8, temperature=0.0):
        sent.extend(messages_list)
        return [{"raw_text": "42", "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                 "status": 200} for _ in messages_list]

    monkeypatch.setattr(async_api.AsyncOpenRouterClient, "query_messages_batch", fake_query)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    return sent


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# --- runners.api -----------------------------------------------------------------

def test_api_runner_writes_every_suite_it_is_given(canned, tmp_path):
    rc = api.main(["--model_id", "openai/test-model", "--suites", "external", "history",
                   "--max_items", "2", "--out_dir", str(tmp_path)])
    assert rc == 0
    for suite in ("external", "history"):
        d = tmp_path / suite / "openai_test-model"
        assert (d / "summary.json").exists()
        recs = _records(d / "results.jsonl")
        assert len(recs) == 10
        assert all(r["answer_int"] == 42 and r["api_usage"]["completion_tokens"] >= 1 for r in recs)
    hist = _records(tmp_path / "history" / "openai_test-model" / "results.jsonl")
    assert {r["condition"] for r in hist} == {
        "control_twostage", "irrelevant_low", "irrelevant_high", "plausible_low", "plausible_high"}
    anchored = [r for r in hist if r["condition"].startswith(("plausible", "irrelevant"))]
    assert all(r["stage1_answer"] == 42 and r["anchor_value"] == 42 for r in anchored)
    # Stage 2 went out as a real three-turn chat
    assert any(len(m) == 3 and m[1]["role"] == "assistant" for m in canned)


def test_api_runner_flat_history_quotes_the_exchange(canned, tmp_path):
    api.main(["--model_id", "openai/test-model", "--suites", "history", "--max_items", "1",
              "--out_dir", str(tmp_path), "--history_chat_format", "flat"])
    assert all(len(m) == 1 for m in canned), "flat format never sends multi-turn chats"
    # control_twostage is two-stage too, so five conversations are quoted per item
    assert sum(m[0]["content"].startswith("Previous conversation:") for m in canned) == 5


def test_api_runner_skips_finished_suites_unless_forced(canned, tmp_path):
    argv = ["--model_id", "openai/test-model", "--suites", "external", "--max_items", "1",
            "--out_dir", str(tmp_path)]
    api.main(argv)
    n = len(canned)
    api.main(argv)
    assert len(canned) == n, "second run must not call the API"
    api.main([*argv, "--force"])
    assert len(canned) == 2 * n


def test_api_runner_needs_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert api.main(["--model_id", "openai/x", "--suites", "external", "--out_dir", str(tmp_path)]) == 1


# --- runners.icl_dist_api --------------------------------------------------------

def test_icl_dist_api_uses_the_per_model_layout(canned, tmp_path):
    rc = icl_dist_api.main(["--model_ids", "openai/test-model", "--max_items", "1",
                            "--out_dir", str(tmp_path)])
    assert rc == 0
    d = tmp_path / "openai_test-model"
    assert (d / "summary.json").exists() and len(_records(d / "results.jsonl")) == 5


# --- runners.sampling ------------------------------------------------------------

def test_sampling_runner_labels_each_decoding_setting(canned, tmp_path):
    rc = sampling.main(["--model_id", "openai/test-model", "--backend", "openrouter",
                        "--suites", "external", "--max_items", "1", "--n_seeds", "2",
                        "--out_dir", str(tmp_path)])
    assert rc == 0
    base = tmp_path / "external" / "openai_test-model"
    assert sorted(p.name for p in base.iterdir()) == ["greedy", "t0.7_s0", "t0.7_s1"]
    greedy = _records(base / "greedy" / "results.jsonl")
    sampled = _records(base / "t0.7_s1" / "results.jsonl")
    assert all(r["temperature"] == 0.0 and r["seed_idx"] == 0 for r in greedy)
    assert all(r["temperature"] == 0.7 and r["seed_idx"] == 1 for r in sampled)


# --- runners.mitigation_baseline -------------------------------------------------

def test_mitigation_baseline_appends_the_reminder_and_tags_records(canned, tmp_path):
    rc = mitigation_baseline.main(["--model_id", "openai/test-model", "--backend", "openrouter",
                                   "--suites", "external", "history", "--max_items", "1",
                                   "--out_dir", str(tmp_path)])
    assert rc == 0
    for suite in ("external", "history"):
        recs = _records(tmp_path / suite / "openai_test-model" / "results.jsonl")
        assert recs and all(r["mitigation"] == "ignore_anchor" for r in recs)
    user_turns = [m[-1]["content"] for m in canned]
    assert all(t.endswith(mitigation_baseline.MITIGATION_SUFFIX) for t in user_turns)


# --- runners.tool ------------------------------------------------------------------

@pytest.mark.parametrize("model_id, flag, supports, expected", [
    ("Qwen/Qwen2.5-7B-Instruct", False, True, False),   # native tool messages
    ("google/gemma-3-4b-it", False, True, True),        # template without a tool role
    ("allenai/OLMo-2-1124-13B-Instruct", False, True, True),
    ("Qwen/Qwen2.5-7B-Instruct", True, True, True),     # asked for explicitly
    ("openai/gpt-5.4-mini", False, False, True),        # backend cannot send tool messages
])
def test_tool_plaintext_decision(monkeypatch, model_id, flag, supports, expected):
    monkeypatch.delenv("ANCHORBENCH_TOOL_PLAINTEXT", raising=False)
    args = argparse.Namespace(model_id=model_id, tool_plaintext=flag)
    assert tool.use_plaintext(args, FakeBackend(supports_tool_messages=supports)) is expected
