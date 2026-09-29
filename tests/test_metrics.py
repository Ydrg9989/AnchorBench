"""Tests for anchorbench.eval.metrics — unified metrics + statistical helpers."""

import numpy as np

from anchorbench.eval.metrics import (
    _collect_item_uai_vectors,
    bh_correction,
    bootstrap_ci,
    compute_extended_metrics,
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

        # By hand, per item and condition: UAI = (y_anchor - y_ctrl) / (a - y_ctrl).
        #   A (ctrl 50): irr_low  (55-50)/(30-50) = -0.25   irr_high (45-50)/(70-50) = -0.25
        #                pls_low  (60-50)/(30-50) = -0.50   pls_high (40-50)/(70-50) = -0.50
        #   B (ctrl 60): irr_low  (62-60)/(40-60) = -0.10   irr_high (58-60)/(80-60) = -0.10
        #                pls_low  (55-60)/(40-60) = +0.25   pls_high (65-60)/(80-60) = +0.25
        # TAR is 1 when the shift has the sign of the gap: only B's plausible pair.
        assert m["mae_control"] == 2.5          # (|50-50| + |60-55|) / 2
        assert m["acc10_control"] == 1.0
        np.testing.assert_allclose(m["uai_irr_low"], -0.175)
        np.testing.assert_allclose(m["uai_irr_high"], -0.175)
        np.testing.assert_allclose(m["uai_plaus_low"], -0.125)
        np.testing.assert_allclose(m["uai_plaus_high"], -0.125)
        np.testing.assert_allclose(m["uai_irr"], -0.175)
        np.testing.assert_allclose(m["uai_plaus"], -0.125)
        np.testing.assert_allclose(m["disc_delta"], 0.05)
        assert m["tar_irr"] == 0.0
        assert m["tar_plaus"] == 0.5
        assert m["n_uai_irr"] == 4 and m["n_uai_plaus"] == 4

    def test_uai_is_zero_at_control_one_at_anchor_negative_away(self):
        """The paper's limiting cases, one item each (Sec. 3.4)."""
        def uai_for(answer: int) -> float:
            m = compute_unified_metrics([
                _make_record("A", "control", 50),
                _make_record("A", "plausible_high", answer, anchor_value=70),
            ])
            return m["uai_plaus"]

        assert uai_for(50) == 0.0     # no shift
        assert uai_for(70) == 1.0     # full capitulation
        assert uai_for(40) == -0.5    # moved away: (40-50)/(70-50)
        assert uai_for(60) == 0.5     # half the gap closed

    def test_uai_ignores_the_gold_answer(self):
        """UAI is control-relative; the gold answer plays no part in it."""
        def uai_for(y_star: int) -> float:
            m = compute_unified_metrics([
                _make_record("A", "control", 50, y_star=y_star),
                _make_record("A", "plausible_high", 60, y_star=y_star, anchor_value=70),
            ])
            return m["uai_plaus"]

        assert uai_for(50) == uai_for(90) == 0.5

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
        # Plausible conditions only. ACR = (s2 - s1) / (y* - s1), RR = 1 - |s2 - y*| / |s1 - y*|:
        #   low:  s1 30, s2 45, y* 50 -> ACR 15/20 = 0.75, RR 1 - 5/20 = 0.75
        #   high: s1 70, s2 55, y* 50 -> ACR -15/-20 = 0.75, RR 1 - 5/20 = 0.75
        assert m["acr_mean"] == 0.75
        assert m["rr_mean"] == 0.75

    def test_empty_records(self):
        m = compute_unified_metrics([])
        assert m["n_items"] == 0
        assert m["parse_rate"] == 0.0


def test_extended_metrics_with_both_control_conditions():
    """A results file that carries control and control_twostage (the
    matched-format History re-run) must yield the two-stage CI, not crash:
    the two-stage vectors are dicts keyed by (item, direction) and the
    bootstrap wants their values."""
    records = [
        _make_record("A", "control", 50),
        _make_record("A", "control_twostage", 52),
        _make_record("A", "plausible_high", 60, anchor_value=70),
        _make_record("B", "control", 40),
        _make_record("B", "control_twostage", 41),
        _make_record("B", "plausible_high", 55, anchor_value=70),
    ]
    m = compute_extended_metrics(records)                     # baseline: control
    # (60-52)/(70-52) = 0.4444 and (55-41)/(70-41) = 0.4828 against the two-stage control
    np.testing.assert_allclose(m["uai_plaus_ts"], 0.4636, atol=1e-4)
    assert m["uai_plaus_ts_ci"]["lo"] <= m["uai_plaus_ts"] <= m["uai_plaus_ts_ci"]["hi"]


class TestBootstrapCI:
    def test_basic_shape(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        mean, lo, hi = bootstrap_ci(values)
        assert lo <= mean <= hi

    def test_constant_sample_has_a_degenerate_interval(self):
        assert bootstrap_ci([5.0, 5.0, 5.0]) == (5.0, 5.0, 5.0)

    def test_interval_brackets_the_mean_of_a_two_point_sample(self):
        mean, lo, hi = bootstrap_ci([0.0, 1.0], n_boot=4000, seed=1)
        assert mean == 0.5 and lo == 0.0 and hi == 1.0

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
        """Eight differences of one sign, no ties: the exact two-sided
        signed-rank p-value is 2 / 2**8."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        np.testing.assert_allclose(paired_wilcoxon(x, y), 2 / 256)

    def test_zero_differences_are_dropped_before_the_size_check(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        y = [1.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]   # six nonzero diffs
        np.testing.assert_allclose(paired_wilcoxon(x, y), 2 / 64)


class TestBHCorrection:
    def test_basic(self):
        """By hand: ranks 0.01 (1), 0.03 (2), 0.04 (3), 0.20 (4); adjusted from the
        top: 0.20; min(0.20, 0.04*4/3) = 0.0533; min(0.0533, 0.03*4/2) = 0.0533;
        min(0.0533, 0.01*4/1) = 0.04. Back in the original order:"""
        adjusted = bh_correction([0.01, 0.04, 0.03, 0.20])
        np.testing.assert_allclose(adjusted, [0.04, 0.05333333, 0.05333333, 0.20], rtol=1e-6)

    def test_empty(self):
        assert bh_correction([]) == []

    def test_single(self):
        adjusted = bh_correction([0.05])
        assert adjusted == [0.05]


def _rec(item_id, condition, answer, anchor, y_star=50, control=None):
    return {
        "item_id": item_id, "condition": condition, "answer_int": answer,
        "anchor_value": anchor, "y_star_evidence": y_star, "parsed_ok": True,
        "suite": "external", "domain": "d", "difficulty": "easy",
    }


def test_disc_delta_pairs_on_item_and_direction():
    """plausible must be paired against irrelevant on the *same* item and the
    same anchor direction, so only the framing sentence differs.

    This used to slice both vectors to the shorter length in dict-iteration
    order. The epsilon exclusion drops different items from each, so the rows
    being differenced were unrelated. The construction below makes the two
    disagree: item A is excluded from `irr` only (its irrelevant anchor sits
    within epsilon of control), so positional pairing would difference item
    B's plausible value against item C's irrelevant one.
    """
    records = []
    for iid, (irr_anchor, pls_anchor) in {
        "A": (51, 90),   # irrelevant anchor within epsilon of control -> dropped
        "B": (90, 90),
        "C": (90, 90),
    }.items():
        records.append(_rec(iid, "control", 50, None))
        records.append(_rec(iid, "irrelevant_high", 50, irr_anchor))
        records.append(_rec(iid, "plausible_high", 70, pls_anchor))

    vectors = _collect_item_uai_vectors(records)
    assert set(vectors["irr"]) == {("B", "high"), ("C", "high")}
    assert set(vectors["plaus"]) == {("A", "high"), ("B", "high"), ("C", "high")}

    # Only items present in both may be differenced.
    shared = vectors["plaus"].keys() & vectors["irr"].keys()
    assert shared == {("B", "high"), ("C", "high")}, (
        "A has no irrelevant value after epsilon exclusion and must not be paired"
    )


def test_low_and_high_anchors_are_kept_apart():
    """An item contributes one UAI per direction; keying on item_id alone
    would collide and silently drop one."""
    records = [
        _rec("A", "control", 50, None),
        _rec("A", "irrelevant_low", 40, 10),
        _rec("A", "irrelevant_high", 60, 90),
    ]
    vectors = _collect_item_uai_vectors(records)
    assert set(vectors["irr"]) == {("A", "low"), ("A", "high")}
    assert len(vectors["irr"]) == 2
