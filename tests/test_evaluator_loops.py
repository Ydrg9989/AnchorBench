"""The evaluation loops, run end to end on real smoke datasets with a fake backend.

These are the tests the runners never had: before the loops were unified,
every runner re-implemented parse -> build_record -> write JSONL and none of
them could be exercised without a model. Now the interface is the test
surface: items and a ``Backend`` in, records and a ``results.jsonl`` out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import FakeBackend, constant

from anchorbench.data.generate import generate_suite_dataset
from anchorbench.data.schema import ItemSpec
from anchorbench.data.suites.tool import TOOL_SCHEMAS, get_tool_messages
from anchorbench.eval.backends import Backend
from anchorbench.eval.evaluator import (
    CONDITIONS,
    flatten_conversation,
    prepare_items,
    run_history_two_stage,
    run_single_stage,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews


@pytest.fixture(scope="module")
def smoke_items(tmp_path_factory) -> dict[str, list[dict]]:
    """External, History and Tool at smoke size (6 items each), as item lists."""
    root = tmp_path_factory.mktemp("smoke")
    out: dict[str, list[dict]] = {}
    for suite in ("external", "history", "tool"):
        d = root / f"anchorbench_{suite}_smoke"
        generate_suite_dataset(suite=suite, size="smoke", seed=42, out_dir=str(d))
        views = load_promptviews(d / "promptviews.jsonl")
        specs = load_itemspecs(d / "itemspecs.jsonl")
        out[suite] = prepare_items(views, specs, None, 42)
    return out


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_fake_backend_satisfies_the_protocol():
    assert isinstance(FakeBackend(), Backend)


# --- run_single_stage ------------------------------------------------------------

def test_single_stage_writes_one_record_per_item_and_condition(smoke_items, tmp_path):
    items = smoke_items["external"]
    backend = FakeBackend(constant("42"))
    out = tmp_path / "results.jsonl"

    records = run_single_stage(backend, items, out, batch_size=8)

    assert len(records) == len(items) * len(CONDITIONS) == 30
    assert [(r["item_id"], r["condition"]) for r in records] == [
        (item["item_id"], c) for item in items for c in CONDITIONS
    ]
    assert _lines(out) == records
    rec = records[0]
    assert rec["model_id"] == "fake/model" and rec["answer_int"] == 42 and rec["parsed_ok"]
    assert {"suite", "domain", "difficulty", "anchor_relevance", "anchor_value",
            "y_star_evidence", "parse_strategy", "raw_text"} <= rec.keys()
    # one round trip to the backend, batching left to it
    assert backend.calls == [("generate_batch", 30)]


def test_single_stage_then_summary(smoke_items, tmp_path):
    items = smoke_items["external"]
    records = run_single_stage(FakeBackend(constant("42")), items, tmp_path / "results.jsonl")
    metrics = write_and_summarize(records, tmp_path, label="test")
    assert (tmp_path / "summary.json").exists()
    assert metrics["parse_rate"] == 1.0 and metrics["n_items"] == len(items)


def test_single_stage_records_backend_failure_instead_of_crashing(smoke_items, tmp_path):
    items = smoke_items["external"][:3]
    records = run_single_stage(FakeBackend(fail=True), items, tmp_path / "r.jsonl")
    assert len(records) == 15
    assert all(not r["parsed_ok"] and r["raw_text"].startswith("ERROR:") for r in records)


def test_single_stage_copies_backend_usage_and_extras(smoke_items, tmp_path):
    items = smoke_items["external"][:2]
    backend = FakeBackend(constant("7"), report_usage=True)
    records = run_single_stage(
        backend, items, tmp_path / "r.jsonl",
        record_extras={"temperature": 0.7, "seed_idx": 2},
    )
    assert all(r["temperature"] == 0.7 and r["seed_idx"] == 2 for r in records)
    assert all(r["api_usage"]["completion_tokens"] == 1 for r in records)


def test_single_stage_tool_messages_go_through_the_tool_interface(smoke_items, tmp_path):
    items = smoke_items["tool"][:4]
    backend = FakeBackend(constant("55"))

    def messages_fn(item, cond, pv):
        return get_tool_messages(ItemSpec.from_dict(item["spec"]), cond)[0]

    records = run_single_stage(
        backend, items, tmp_path / "r.jsonl", messages_fn=messages_fn, tools=TOOL_SCHEMAS,
    )
    assert backend.calls == [("generate_batch_tool", 20)]
    assert all(r["answer_int"] == 55 for r in records)


def test_single_stage_prompt_suffix_reaches_the_backend(smoke_items, tmp_path):
    items = smoke_items["external"][:1]
    seen: list[str] = []
    backend = FakeBackend(lambda text: (seen.append(text), "1")[1])
    run_single_stage(backend, items, tmp_path / "r.jsonl", prompt_suffix="\n\nIGNORE ANCHORS")
    assert all(t.endswith("IGNORE ANCHORS") for t in seen)


# --- run_history_two_stage -----------------------------------------------------

def _history_policy(items: list[dict]):
    """Answer 30 to every Stage-1 message of ``items`` and 45 to anything else."""
    stage1 = {
        item[c]["prompt_components"]["stage1_user_message"]
        for item in items for c in CONDITIONS if c != "control"
    }
    return lambda text: "30" if text in stage1 else "45"


def test_history_two_stage_uses_stage1_as_anchor(smoke_items, tmp_path):
    items = smoke_items["history"]
    backend = FakeBackend(_history_policy(items))
    out = tmp_path / "results.jsonl"

    records = run_history_two_stage(backend, items, out)

    assert len(records) == len(items) * 5 and _lines(out) == records
    by_cond = {c: [r for r in records if r["condition"] == c] for c in CONDITIONS}
    for r in by_cond["control"]:
        assert r["anchor_value"] is None and r["stage1_answer"] is None
    for c in ("plausible_low", "plausible_high", "irrelevant_low", "irrelevant_high"):
        for r in by_cond[c]:
            assert r["stage1_answer"] == 30 and r["stage1_raw_text"] == "30"
            assert r["answer_int"] == 45
            assert r["anchor_value"] == r["stage1_answer"]
            assert r["parsed_ok"] and r["answer_int"] is not None
    # three batched round trips: control, stage 1, stage 2 as a real chat
    assert [m for m, _ in backend.calls] == ["generate_batch", "generate_batch", "generate_chat_batch"]


def test_history_flat_format_quotes_the_exchange_in_one_user_message(smoke_items, tmp_path):
    items = smoke_items["history"][:2]
    seen: list[str] = []
    backend = FakeBackend(lambda text: (seen.append(text), "33")[1])
    run_history_two_stage(backend, items, tmp_path / "r.jsonl", chat_format="flat")
    assert [m for m, _ in backend.calls] == ["generate_batch", "generate_batch", "generate_batch"]
    stage2 = [t for t in seen if t.startswith("Previous conversation:")]
    assert len(stage2) == 2 * 4
    assert stage2[0] == flatten_conversation(*_parts(stage2[0]))


def _parts(flat: str) -> tuple[str, str, str]:
    body = flat[len("Previous conversation:\nUser: "):]
    s1, rest = body.split("\nAssistant: ", 1)
    raw, s2 = rest.split("\n\nUser: ", 1)
    return s1, raw, s2


def test_history_resume_skips_completed_pairs(smoke_items, tmp_path):
    items = smoke_items["history"][:3]
    out = tmp_path / "results.jsonl"
    first = run_history_two_stage(FakeBackend(_history_policy(items)), items, out)
    backend = FakeBackend(_history_policy(items))
    second = run_history_two_stage(backend, items, out, resume=True)
    assert len(first) == len(second) == 15
    assert len(_lines(out)) == 15
    assert backend.calls == [], "nothing left to run, so the backend is never called"


def test_history_rejects_unknown_chat_format(smoke_items, tmp_path):
    with pytest.raises(ValueError):
        run_history_two_stage(FakeBackend(), smoke_items["history"][:1], tmp_path / "r.jsonl",
                              chat_format="markdown")
