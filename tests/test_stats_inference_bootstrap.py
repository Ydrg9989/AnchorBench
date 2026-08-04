"""The range CI in tab:stats_inference must come from a real bootstrap.

`build_stats_inference` used to select bootstrap draws with
`r["model"] in sample_models`, a membership test, so a model drawn three
times contributed once. That is a random subset retaining each model with
probability 1-(1-1/n)^n ~ 0.65, not a resample with replacement, and it
understated the interval because it never produced the heavily-reweighted
draws a bootstrap depends on.

The bug is easy to reintroduce -- the correct form and the broken form differ
by one line -- and a golden hash would flag the change without explaining it.
See docs/RECONCILIATION.md D4.
"""

import json
import re

from pathlib import Path

import numpy as np
import pytest

from anchorbench.paper._common import OW_MODELS_ORDER
from anchorbench.paper.tables_appendix import build_stats_inference

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("External", "History", "Icl", "Rag", "Tool")


@pytest.fixture(scope="module")
def unified():
    rows = []
    for name in ("full_benchmark", "api_benchmark"):
        path = ROOT / "results" / name / "unified_all_suites.json"
        if not path.exists():
            pytest.skip(f"{path} not present")
        rows += json.loads(path.read_text())
    return rows


def test_range_ci_matches_a_correct_bootstrap(unified):
    """The emitted CI must equal an independently written correct bootstrap.

    Recomputed here from scratch rather than asserting a literal, so the test
    states the statistical property instead of a magic number.
    """
    ow = set(OW_MODELS_ORDER)
    models = sorted({r["model"] for r in unified if r["model"] in ow})
    disc = {
        (r["model"], r.get("suite")): r["disc_delta"]
        for r in unified
        if r["model"] in ow and r.get("disc_delta") is not None
    }

    rs = np.random.RandomState(42)
    draws = []
    for _ in range(2000):
        idx = rs.randint(0, len(models), size=len(models))
        boot = [models[i] for i in idx]  # keeps repeats -- the whole point
        means = []
        for s in SUITES:
            vals = [disc[(m, s)] for m in boot if (m, s) in disc]
            if vals:
                means.append(float(np.mean(vals)))
        if len(means) > 1:
            draws.append(max(means) - min(means))
    expected_lo = np.percentile(draws, 2.5)
    expected_hi = np.percentile(draws, 97.5)

    latex = build_stats_inference(unified)
    row = next(ln for ln in latex.splitlines() if "range (open-weight panel)" in ln)
    lo, hi = (float(x) for x in re.search(r"\[(-?[\d.]+), (-?[\d.]+)\]", row).groups())

    assert (lo, hi) == (round(expected_lo, 2), round(expected_hi, 2)), (
        f"emitted CI [{lo}, {hi}] does not match a correct bootstrap "
        f"[{expected_lo:.4f}, {expected_hi:.4f}]. Has the resampling regressed "
        f"to a membership test? See docs/RECONCILIATION.md D4."
    )


def test_subset_sampler_would_be_detectably_narrower(unified):
    """Guards the guard: the two samplers must give different answers.

    If they ever agreed, the test above would pass even with the bug back in.
    """
    ow = set(OW_MODELS_ORDER)
    models = sorted({r["model"] for r in unified if r["model"] in ow})
    disc = {
        (r["model"], r.get("suite")): r["disc_delta"]
        for r in unified
        if r["model"] in ow and r.get("disc_delta") is not None
    }

    def ci(with_replacement):
        rs = np.random.RandomState(42)
        draws = []
        for _ in range(2000):
            idx = rs.randint(0, len(models), size=len(models))
            boot = [models[i] for i in idx]
            if not with_replacement:
                boot = [m for m in models if m in set(boot)]  # the old bug
            means = []
            for s in SUITES:
                vals = [disc[(m, s)] for m in boot if (m, s) in disc]
                if vals:
                    means.append(float(np.mean(vals)))
            if len(means) > 1:
                draws.append(max(means) - min(means))
        return np.percentile(draws, 2.5), np.percentile(draws, 97.5)

    good_lo, good_hi = ci(True)
    bad_lo, bad_hi = ci(False)
    assert (good_hi - good_lo) > (bad_hi - bad_lo), (
        "the subset sampler is meant to understate the interval; if it no "
        "longer does, the test above has lost its teeth"
    )
