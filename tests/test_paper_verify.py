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
