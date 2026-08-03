"""Smoke test: anchorbench.paper.verify --quick runs end-to-end on the
checked-in unified summary files.

This catches refactor regressions (broken imports, missing files,
crashes) but does NOT enforce numeric correctness — that is the job of
``anchorbench verify`` in CI, where tolerance failures are expected to
be reviewed by a human. Skipped automatically on minimal checkouts that
do not bundle ``results/``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIFIED = REPO_ROOT / "results" / "full_benchmark" / "unified_all_suites.json"


@pytest.mark.skipif(
    not UNIFIED.exists(),
    reason="unified_all_suites.json not present (skipped on minimal checkouts)",
)
def test_verify_quick_runs() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "anchorbench.paper.verify", "--quick"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    # Exit code 0 = all claims within tolerance; 1 = one or more drifted.
    # Any other exit code (or no SUMMARY block) means the verifier itself
    # crashed -- that is a refactor regression and must fail the smoke test.
    assert proc.returncode in (0, 1), (
        f"verify --quick crashed (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "SUMMARY" in proc.stdout, (
        f"verify --quick did not reach SUMMARY block:\n{proc.stdout}"
    )
