"""``anchorbench verify`` must pass on the checked-in unified summaries.

This is the enforcement point for the claim in ``docs/ARCHITECTURE.md`` that
"`anchorbench verify` is the source of truth; CI fails if any claim drifts".

Previously this test asserted only ``returncode in (0, 1)`` and skipped when
``results/`` was absent -- and ``results/`` was gitignored in its entirety, so
on a fresh clone it always skipped and on a dev box it passed whether or not
claims had drifted. It protected nothing. The two unified summaries and the
small extension artifacts are now committed, so the real assertion is
affordable: exit code 0, no mismatches.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OW_UNIFIED = REPO_ROOT / "results" / "full_benchmark" / "unified_all_suites.json"
API_UNIFIED = REPO_ROOT / "results" / "api_benchmark" / "unified_all_suites.json"


def _run_verify(*flags: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "anchorbench.paper.verify", *flags],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


def test_unified_summaries_are_committed() -> None:
    """Guard the .gitignore negation lane that makes the rest of this file run."""
    assert OW_UNIFIED.exists(), f"missing (should be committed): {OW_UNIFIED}"
    assert API_UNIFIED.exists(), f"missing (should be committed): {API_UNIFIED}"


@pytest.mark.parametrize("flags", [("--quick",), (), ("--strict",)],
                         ids=["quick", "full", "strict"])
def test_verify_reports_no_mismatches(flags: tuple[str, ...]) -> None:
    proc = _run_verify(*flags)
    assert "SUMMARY" in proc.stdout, (
        f"verify {' '.join(flags)} did not reach the SUMMARY block "
        f"(rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
    assert proc.returncode == 0, (
        f"verify {' '.join(flags)} reported drifted claims (rc={proc.returncode}).\n"
        "Either the paper number or the data is wrong -- do not widen the\n"
        f"tolerance to silence this.\n{proc.stdout}"
    )


def test_known_divergences_are_actually_exercised() -> None:
    """rc==0 must mean "checked and reconciled", not "checked nothing".

    Every entry in KNOWN_DIVERGENCES is a claim the paper and the data
    genuinely disagree on, recorded with the size of the disagreement. If the
    tolerance were widened past one, or the check that produces it stopped
    running, the entry would go silent and rc would still be 0. verify treats
    a silent entry as a failure; this asserts that machinery is live rather
    than trivially satisfied.
    """
    from anchorbench.paper.verify import KNOWN_DIVERGENCES

    out = _run_verify().stdout
    reported = out.count("[known]")
    assert reported == len(KNOWN_DIVERGENCES), (
        f"{len(KNOWN_DIVERGENCES)} divergences are recorded but {reported} were "
        f"reported. A recorded divergence that stops firing means either a "
        f"tolerance now swallows it or its check stopped running -- both hide "
        f"a real disagreement.\n{out}"
    )
    # The table is empty at present, because the four tables that used to
    # diverge are now \input-ed from generated bodies. An empty ledger is the
    # goal state, not a broken test -- but it must stay empty honestly, which
    # is what the equality above enforces in both directions.


def test_anchored_mae_is_actually_computed() -> None:
    """Regression: PAPER_AMAE verified nothing at all.

    verify_anchored_mae read ``mae_irr`` / ``mae_plaus`` from the unified
    summaries. Those keys have never existed there -- only ``mae_control``
    does -- so every suite took the "skipped" branch and the check silently
    passed at any tolerance, including the old 0.20. The deltas come from the
    per-record generations, which is where the table itself gets them.
    """
    out = _run_verify().stdout

    # The deltas need results/**/results.jsonl, which is published as tarballs
    # rather than committed, so a clean clone legitimately cannot run this.
    # That is reported as "NOT CHECKED", which is the honest outcome and quite
    # different from the silent "skipped" this test exists to catch.
    if "NOT CHECKED: needs results/" in out:
        pytest.skip("results/ generations absent (clean clone)")

    assert "dMAE_irr=" in out, f"anchored-MAE check produced no values:\n{out}"
    assert "skipped (no mae_irr" not in out, (
        "verify_anchored_mae is reading fields that do not exist in the "
        f"unified summaries again:\n{out}"
    )


def test_dose_response_is_checked_per_tier() -> None:
    """Regression: verify_dose used to pool all 14 models against
    open-weight-only paper values.

    ``PAPER_DOSE`` holds the open-weight tier means quoted in Finding 3 and
    drawn as the "Open-weight" curve in Figure 4. Filtering only on suite mixed
    the API tier in, which biased every one of the six cells downward; the
    0.04 tolerance hid five of them and only External@15 ever failed.
    """
    out = _run_verify("--quick").stdout
    assert "open-weight tier" in out, "dose check is not tier-aware"
    assert "API tier" in out, "API-tier dose claims are not being checked"
    assert "[FAIL]" not in out, f"dose-response cell failed:\n{out}"


def test_a_ledger_row_that_no_longer_diverges_is_reported_stale(monkeypatch, capsys) -> None:
    """The stale-ledger check compares section names; it used to compare
    against spellings ("posn", "gs", "samp", "mit") that no check ever wrote,
    so a stale row in those sections was never reported."""
    from anchorbench.paper import verify

    monkeypatch.setitem(verify.KNOWN_DIVERGENCES, ("pos_n", "a row that no longer diverges"), 0.01)
    monkeypatch.setitem(verify.KNOWN_DIVERGENCES, ("sampling", "another one"), 0.01)
    rc = verify.main([])
    out = capsys.readouterr().out
    assert "Stale ledger rows:  2" in out, out[-1500:]
    assert rc != 0


def test_a_nan_p_value_is_a_failed_check() -> None:
    """Three cells give the signed-rank test fewer than six nonzero
    differences, so p is NaN; NaN used to pass every comparison."""
    from anchorbench.paper import verify

    cells = [
        {"model": m, "model_slug": m, "suite": "External", "uai_plaus": 0.3 + i / 10,
         "uai_irr": 0.1, "disc_delta": 0.2 + i / 10, "acc10_control": 0.5 + i / 10}
        for i, m in enumerate(("Qwen-1.5B", "Qwen-3B", "Qwen-7B"))
    ]
    mismatches: list = []
    verify.verify_stats_inference(cells, mismatches)
    nan_rows = [m for m in mismatches if m.label == "External p_BH" and m.data != m.data]
    assert nan_rows, [str(m) for m in mismatches]
