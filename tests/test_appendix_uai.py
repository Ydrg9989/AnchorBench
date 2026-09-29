"""The appendix curves report the paper's UAI (Sec. 3.4): an item whose
anchor sits within epsilon = 3 of the control answer is excluded, not
counted with an unstable ratio."""

from anchorbench.runners.realism import realism_curve
from anchorbench.runners.rebuttal_intensity import _compute_intensity_curve


def _rec(item, cond, ans, anchor=None):
    return {"item_id": item, "condition": cond, "answer_int": ans,
            "anchor_value": anchor, "parsed_ok": True}


# Item a: gap 2 (excluded; the raw ratio would be 3.0). Item b: gap 10, UAI 0.5.
def _pair(cond):
    return [_rec("a", "control", 50), _rec("a", cond, 56, anchor=52),
            _rec("b", "control", 50), _rec("b", cond, 55, anchor=60)]


def test_realism_curve_uses_the_paper_epsilon():
    out = realism_curve(_pair("plausible_low"), ("rank1",))
    assert (out["plausible__baseline"], out["plausible__baseline_n"]) == (0.5, 1)


def test_intensity_curve_uses_the_paper_epsilon():
    out = _compute_intensity_curve(_pair("plausible_mild_low"), "control")
    assert (out["plausible_mild"], out["plausible_mild_n"]) == (0.5, 1)
