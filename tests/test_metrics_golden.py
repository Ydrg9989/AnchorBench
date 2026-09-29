"""A golden test for the metrics layer that runs on a clean clone.

The pinned artifacts in tests/golden/ need the bulk results tree, which a
clean clone does not have. This one does not: a seeded synthetic record set
(five suites' worth of conditions, parse failures, within-epsilon anchors,
both control conditions, History Stage-1 answers, offsets and difficulties
in the item ids) goes through compute_extended_metrics, compute_by_offset
and compute_by_difficulty, and the result is compared with the committed
tests/golden/metrics_synthetic.json.

Refresh the expected file only when a change to the metrics is intended:

    python -m tests.test_metrics_golden > tests/golden/metrics_synthetic.json
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from anchorbench.eval.metrics import (
    compute_by_difficulty,
    compute_by_offset,
    compute_extended_metrics,
)

EXPECTED = Path(__file__).parent / "golden" / "metrics_synthetic.json"
CONDITIONS = ("irrelevant_low", "irrelevant_high", "plausible_low", "plausible_high")


def synthetic_records(seed: int = 0, n_items: int = 24, *, two_stage: bool) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for i in range(n_items):
        offset = (15, 25, 40)[i % 3]
        difficulty = ("easy", "hard")[i % 2]
        theta = rng.randint(30, 70)
        item_id = f"EXT-d-{difficulty[0]}-off{offset}-{i:03d}"
        y_ctrl = theta + rng.randint(-8, 8)
        base = {"item_id": item_id, "suite": "external", "domain": "d",
                "difficulty": difficulty, "y_star_evidence": theta}
        records.append({**base, "condition": "control", "anchor_value": None,
                        "answer_int": y_ctrl, "parsed_ok": True})
        if two_stage:
            records.append({**base, "condition": "control_twostage", "anchor_value": None,
                            "answer_int": y_ctrl + rng.randint(-3, 3), "parsed_ok": True})
        for cond in CONDITIONS:
            direction = cond.rsplit("_", 1)[1]
            anchor = max(0, theta - offset) if direction == "low" else min(100, theta + offset)
            if i % 7 == 0 and cond == "irrelevant_low":
                anchor = y_ctrl + 1            # within epsilon: excluded from UAI, kept in TAR
            pull = 0.6 if cond.startswith("plausible") else 0.1
            answer = round(y_ctrl + pull * (anchor - y_ctrl) + rng.randint(-4, 4))
            parsed = not (i % 11 == 3 and cond == "plausible_high")
            rec = {**base, "condition": cond, "anchor_value": anchor,
                   "answer_int": answer if parsed else None, "parsed_ok": parsed}
            if two_stage:
                rec["stage1_answer"] = anchor
            records.append(rec)
    return records


def _sanitize(obj):
    """NaN is not JSON; keys must be strings."""
    if isinstance(obj, float) and math.isnan(obj):
        return "nan"
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def current_output() -> dict:
    single = synthetic_records(0, two_stage=False)
    history = synthetic_records(1, two_stage=True)
    return _sanitize({
        "single_stage": compute_extended_metrics(single),
        "single_stage_by_offset": compute_by_offset(single),
        "single_stage_by_difficulty": compute_by_difficulty(single),
        "history_vs_control": compute_extended_metrics(history),
        "history_vs_control_twostage": compute_extended_metrics(
            history, baseline_condition="control_twostage"),
    })


def test_metrics_match_the_committed_expected_output():
    expected = json.loads(EXPECTED.read_text())
    got = current_output()
    diffs = [k for k in expected if got.get(k) != expected[k]]
    assert got == expected, (
        f"metrics output changed for {diffs}; if intended, regenerate with\n"
        "  python -m tests.test_metrics_golden > tests/golden/metrics_synthetic.json"
    )


if __name__ == "__main__":
    print(json.dumps(current_output(), indent=1, sort_keys=True))
