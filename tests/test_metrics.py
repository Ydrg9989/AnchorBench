"""Tests for anchorbench.eval.metrics — unified metrics + statistical helpers."""

import numpy as np

from anchorbench.eval.metrics import (
    bh_correction,
    bootstrap_ci,
    compute_unified_metrics,
    paired_wilcoxon,
)


def _make_record(
    item_id: str,
    condition: str,
    answer_int: int | None,
    y_star: int = 50,
    anchor_value: int | None = None,
    parsed_ok: bool = True,
    **extras,
) -> dict:
    rec = {
        "model_id": "test-model",
        "item_id": item_id,
        "suite": "external",
        "domain": "test",
        "difficulty": "easy",
        "condition": condition,
        "anchor_relevance": "none" if condition == "control" else "plausible",
        "anchor_value": anchor_value,
        "y_star_evidence": y_star,
        "y_star_theta": y_star,
        "answer_int": answer_int,
        "parsed_ok": parsed_ok,
        "parse_strategy": "regex" if parsed_ok else "failed",
        "raw_text": str(answer_int) if answer_int is not None else "",
    }
    rec.update(extras)
    return rec


class TestComputeUnifiedMetrics:
    def test_basic_two_items(self):
        """Two items, known answers → hand-computable UAI/TAR."""
        records = [
            _make_record("A", "control", 50, y_star=50),
            _make_record("A", "irrelevant_low", 55, y_star=50, anchor_value=30),
            _make_record("A", "irrelevant_high", 45, y_star=50, anchor_value=70),
            _make_record("A", "plausible_low", 60, y_star=50, anchor_value=30),
            _make_record("A", "plausible_high", 40, y_star=50, anchor_value=70),
            _make_record("B", "control", 60, y_star=55),
            _make_record("B", "irrelevant_low", 62, y_star=55, anchor_value=40),
            _make_record("B", "irrelevant_high", 58, y_star=55, anchor_value=80),
            _make_record("B", "plausible_low", 55, y_star=55, anchor_value=40),
            _make_record("B", "plausible_high", 65, y_star=55, anchor_value=80),
        ]
        m = compute_unified_metrics(records, epsilon=3.0)

        assert m["n_items"] == 2
        assert m["n_records"] == 10
        assert m["parse_rate"] == 1.0

        assert m["mae_control"] is not None
        assert m["mae_control"] == round((abs(50 - 50) + abs(60 - 55)) / 2, 2)

        assert m["uai_irr"] is not None
        assert m["uai_plaus"] is not None
        assert m["disc_delta"] is not None

    def test_parse_failures_excluded(self):
        records = [
            _make_record("A", "control", 50, y_star=50),
            _make_record("A", "irrelevant_low", None, y_star=50,
                         anchor_value=30, parsed_ok=False),
            _make_record("A", "irrelevant_high", 45, y_star=50, anchor_value=70),
            _make_record("A", "plausible_low", 60, y_star=50, anchor_value=30),
            _make_record("A", "plausible_high", 40, y_star=50, anchor_value=70),
        ]
        m = compute_unified_metrics(records)
        assert m["parse_rate"] < 1.0
        assert m["n_items"] == 1

    def test_epsilon_exclusion(self):
        """When |anchor - y_control| < epsilon, UAI should be excluded."""
        records = [
            _make_record("A", "control", 50, y_star=50),
            _make_record("A", "irrelevant_low", 51, y_star=50, anchor_value=52),
            _make_record("A", "irrelevant_high", 49, y_star=50, anchor_value=48),
            _make_record("A", "plausible_low", 51, y_star=50, anchor_value=52),
            _make_record("A", "plausible_high", 49, y_star=50, anchor_value=48),
        ]
        m = compute_unified_metrics(records, epsilon=3.0)
        assert m["n_uai_irr"] == 0
        assert m["n_uai_plaus"] == 0
        assert m["uai_irr"] is None
        assert m["uai_plaus"] is None

    def test_tar_computation(self):
        """TAR = 1 when shift direction matches anchor direction."""
        records = [
            _make_record("A", "control", 50, y_star=50),
            _make_record("A", "irrelevant_low", 45, y_star=50, anchor_value=30),
            _make_record("A", "irrelevant_high", 55, y_star=50, anchor_value=70),
            _make_record("A", "plausible_low", 45, y_star=50, anchor_value=30),
            _make_record("A", "plausible_high", 55, y_star=50, anchor_value=70),
        ]
        m = compute_unified_metrics(records)
        assert m["tar_irr"] == 1.0
        assert m["tar_plaus"] == 1.0

    def test_history_acr_rr(self):
        records = [
            _make_record("A", "control", 50, y_star=50),
            _make_record("A", "irrelevant_low", 52, y_star=50,
                         anchor_value=40, stage1_answer=40),
            _make_record("A", "irrelevant_high", 48, y_star=50,
                         anchor_value=60, stage1_answer=60),
            _make_record("A", "plausible_low", 45, y_star=50,
                         anchor_value=30, stage1_answer=30),
            _make_record("A", "plausible_high", 55, y_star=50,
                         anchor_value=70, stage1_answer=70),
        ]
        m = compute_unified_metrics(records)
        assert "acr_mean" in m
        assert "rr_mean" in m

    def test_empty_records(self):
        m = compute_unified_metrics([])
        assert m["n_items"] == 0
        assert m["parse_rate"] == 0.0


class TestBootstrapCI:
    def test_basic_shape(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        mean, lo, hi = bootstrap_ci(values)
        assert lo <= mean <= hi

    def test_deterministic(self):
        values = [1.0, 2.0, 3.0]
        r1 = bootstrap_ci(values, seed=42)
        r2 = bootstrap_ci(values, seed=42)
        assert r1 == r2

    def test_empty(self):
        mean, lo, hi = bootstrap_ci([])
        assert np.isnan(mean)


class TestPairedWilcoxon:
    def test_identical(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        p = paired_wilcoxon(x, x)
        assert np.isnan(p)  # all differences are zero → not enough nonzero

    def test_different(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        p = paired_wilcoxon(x, y)
        assert 0.0 <= p <= 1.0


class TestBHCorrection:
    def test_basic(self):
        p_values = [0.01, 0.04, 0.03, 0.20]
        adjusted = bh_correction(p_values)
        assert len(adjusted) == 4
        for a, p in zip(adjusted, p_values):
            assert a >= p

    def test_empty(self):
        assert bh_correction([]) == []

    def test_single(self):
        adjusted = bh_correction([0.05])
        assert adjusted == [0.05]
