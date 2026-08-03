"""Smoke test for the AnchorBench pipeline (no GPU required).

Validates that the core data flow works end-to-end:
  1. Parsing raw model outputs into numeric answers
  2. Building result records with correct schema
  3. Computing unified metrics from synthetic records
  4. Writing and reading summary.json
  5. Discovering result files from the canonical directory layout
  6. runner_utils helpers produce correct outputs

Runs in < 5 seconds with no model loading.
"""

from __future__ import annotations

import json
from pathlib import Path

from anchorbench.eval.evaluator import build_record, parse_response, write_and_summarize
from anchorbench.eval.metrics import compute_unified_metrics
from anchorbench.eval.runner_utils import (
    build_suffix,
    discover_results,
    fmt,
    fmt_pct,
    model_output_dir,
)

GOLD = 42.0
ITEM_TEMPLATE = {
    "item_id": "smoke_001",
    "suite": "external",
    "domain": "test_domain",
    "difficulty": "easy",
    "y_star_evidence": GOLD,
    "y_star_theta": GOLD,
}

PV_TEMPLATE = {
    "prompt_text": "Estimate the value (0-100).",
    "anchor_relevance": "none",
    "anchor_value": None,
}


def _make_item(item_id: str = "smoke_001") -> dict:
    item = dict(ITEM_TEMPLATE, item_id=item_id)
    conditions = {}
    for cond in ["control", "irrelevant_low", "irrelevant_high", "plausible_low", "plausible_high"]:
        pv = dict(PV_TEMPLATE)
        if cond.startswith("irrelevant"):
            pv["anchor_relevance"] = "irrelevant"
            pv["anchor_value"] = 10.0 if "low" in cond else 90.0
        elif cond.startswith("plausible"):
            pv["anchor_relevance"] = "plausible"
            pv["anchor_value"] = 20.0 if "low" in cond else 80.0
        conditions[cond] = pv
    item.update(conditions)
    item["spec"] = {"item_id": item_id, "y_star": GOLD}
    return item


def _synthetic_records(n_items: int = 10) -> list[dict]:
    """Build synthetic result records mimicking a real run."""
    records = []
    for i in range(n_items):
        item = _make_item(f"item_{i:03d}")
        for cond in ["control", "irrelevant_low", "irrelevant_high", "plausible_low", "plausible_high"]:
            pv = item[cond]
            if cond == "control":
                answer = int(GOLD) + (i % 5)
            elif "irrelevant" in cond:
                answer = int(GOLD) + 3
            else:
                answer = int(pv["anchor_value"]) if pv["anchor_value"] else int(GOLD)
            rec = build_record(
                "test-model", item, cond, pv,
                answer=answer, parsed_ok=True,
                parse_strategy="regex_last_int", raw_text=f"Answer: {answer}",
            )
            records.append(rec)
    return records


class TestParsing:
    def test_plain_integer(self):
        answer, ok, strategy = parse_response("42", "prompt")
        assert ok
        assert answer == 42

    def test_answer_tag(self):
        answer, ok, strategy = parse_response(
            "I think 42.\n<answer>42</answer>", "prompt",
        )
        assert ok
        assert answer == 42

    def test_answer_line(self):
        answer, ok, strategy = parse_response(
            "Some reasoning.\nAnswer: 55", "prompt",
        )
        assert ok
        assert answer == 55

    def test_no_number(self):
        answer, ok, strategy = parse_response("No numeric answer here.", "prompt")
        assert not ok
        assert answer is None


class TestBuildRecord:
    def test_schema_fields(self):
        item = _make_item()
        pv = item["control"]
        rec = build_record(
            "test-model", item, "control", pv,
            answer=42, parsed_ok=True,
            parse_strategy="regex", raw_text="42",
        )
        required = {
            "model_id", "item_id", "suite", "domain", "difficulty",
            "condition", "anchor_relevance", "anchor_value",
            "y_star_evidence", "y_star_theta",
            "answer_int", "parsed_ok", "parse_strategy", "raw_text",
        }
        assert required.issubset(rec.keys()), f"Missing: {required - rec.keys()}"
        assert rec["model_id"] == "test-model"
        assert rec["answer_int"] == 42

    def test_extras_propagated(self):
        item = _make_item()
        pv = item["control"]
        rec = build_record(
            "m", item, "control", pv,
            answer=1, parsed_ok=True, parse_strategy="r", raw_text="1",
            custom_field="hello",
        )
        assert rec["custom_field"] == "hello"


class TestMetrics:
    def test_unified_metrics_schema(self):
        records = _synthetic_records(10)
        m = compute_unified_metrics(records)
        for key in ("uai_irr", "uai_plaus", "tar_irr", "tar_plaus",
                     "disc_delta", "mae_control", "acc10_control", "parse_rate"):
            assert key in m, f"Missing metric: {key}"

    def test_parse_rate_all_ok(self):
        records = _synthetic_records(5)
        m = compute_unified_metrics(records)
        assert m["parse_rate"] == 1.0

    def test_disc_delta_sign(self):
        records = _synthetic_records(10)
        m = compute_unified_metrics(records)
        if m["disc_delta"] is not None:
            assert isinstance(m["disc_delta"], float)


class TestWriteAndSummarize:
    def test_writes_summary_json(self, tmp_path: Path):
        records = _synthetic_records(5)
        results_path = tmp_path / "results.jsonl"
        with open(results_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        write_and_summarize(records, tmp_path, label="smoke")
        summary = tmp_path / "summary.json"
        assert summary.exists()
        data = json.loads(summary.read_text())
        assert "uai_irr" in data
        assert "parse_rate" in data


class TestDiscoverResults:
    def test_discovers_standard_layout(self, tmp_path: Path):
        suite_dir = tmp_path / "external" / "test_model"
        suite_dir.mkdir(parents=True)
        (suite_dir / "results.jsonl").write_text('{"a":1}\n')

        found = discover_results(tmp_path)
        assert len(found) == 1
        suite, slug, short, path = found[0]
        assert suite == "external"
        assert slug == "test_model"
        assert path == suite_dir / "results.jsonl"

    def test_empty_dir_returns_nothing(self, tmp_path: Path):
        assert discover_results(tmp_path) == []


class TestRunnerUtils:
    def test_build_suffix_empty(self):
        class Args:
            request_reasoning = False
            request_final_line = False
            request_xml_answer = False
        assert build_suffix(Args()) == ""

    def test_build_suffix_xml(self):
        class Args:
            request_reasoning = False
            request_final_line = False
            request_xml_answer = True
        s = build_suffix(Args())
        assert "<answer>" in s

    def test_fmt(self):
        assert fmt(None) == "---"
        assert fmt(0.1234, 2) == "0.12"

    def test_fmt_pct(self):
        assert fmt_pct(None) == "---"
        assert fmt_pct(0.956) == "95.6%"

    def test_model_output_dir(self, tmp_path: Path):
        class Args:
            model_id = "org/Model-Name"
            out_dir = tmp_path
        d = model_output_dir(Args())
        assert d == tmp_path / "org_Model-Name"
        assert d.is_dir()
