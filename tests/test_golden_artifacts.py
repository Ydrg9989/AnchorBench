"""Behaviour-preservation check for the paper artifact generators.

This answers "does the code still agree with itself", which is a different
question from "does the code agree with the paper" -- that one is verify.py's,
and the two must not be conflated. The manifests in tests/golden/ pin what the
generators emit today, drift and all, so a refactor can be shown not to have
changed any number. A genuine disagreement with the paper belongs in
docs/RECONCILIATION.md, never in a loosened check here.

Regenerate the manifests with scripts/update_golden.sh, and only when the
change to generator output is intended.

Three tiers, because not everything can be regenerated from a clean clone:

* outputs/tables/ and outputs/figures/ are regenerated here and compared. Most
  need the full results/ tree, which is gitignored and published as tarballs,
  so those tests skip when it is absent.
* results/rebuttal/*.tex are pinned but not regenerated: their inputs are the
  bulk rebuttal generations, which are not committed. The files themselves
  are, so this still catches an accidental edit.
* Figures are compared as PNG. Matplotlib stamps /CreationDate and /ID into
  PDFs, so PDF bytes differ on every run even when the plot is identical
  (verified); the PNGs are byte-stable.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"

# results.jsonl for the open-weight panel: present in a full working tree,
# absent in a clean clone. Most appendix tables and fig4 need it.
BULK_MARKER = ROOT / "results" / "full_benchmark" / "external"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(name: str) -> dict[str, str]:
    path = GOLDEN / name
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not present; run scripts/update_golden.sh")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            out[rel.strip()] = digest
    return out


def _compare(manifest: dict[str, str]) -> list[str]:
    problems = []
    for rel, digest in manifest.items():
        path = ROOT / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif _sha256(path) != digest:
            problems.append(f"changed: {rel}")
    return problems


def _requires_bulk():
    if not BULK_MARKER.exists():
        pytest.skip(
            "results/ generations absent (clean clone). Fetch the results "
            "bundles to run this; see scripts/make_result_bundles.sh"
        )


def _generate(module: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", f"anchorbench.paper.{module}", *args],
        cwd=ROOT, check=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )


def _check_against(manifest: dict[str, str], produced_dir: Path) -> list[str]:
    problems = []
    for rel, digest in manifest.items():
        produced = produced_dir / Path(rel).name
        if not produced.exists():
            problems.append(f"not produced: {Path(rel).name}")
        elif _sha256(produced) != digest:
            problems.append(f"differs: {Path(rel).name}")
    return problems


@pytest.mark.parametrize("name", ["rebuttal_tables.sha256", "unified.sha256"])
def test_committed_artifacts_unchanged(name):
    """Pinned files that are committed, so they need no regeneration."""
    problems = _compare(_manifest(name))
    assert not problems, (
        "committed artifacts no longer match tests/golden/:\n  "
        + "\n  ".join(problems)
        + "\n\nThese back the camera-ready. If the change is intended, say why "
        "in the commit message and refresh with scripts/update_golden.sh."
    )


def test_tables_regenerate_identically(tmp_path):
    """Regenerate the table set and compare against the pinned hashes."""
    _requires_bulk()
    manifest = _manifest("tables.sha256")

    for module in ("tables_main", "tables_appendix"):
        _generate(module, "--out_dir", str(tmp_path))

    problems = _check_against(manifest, tmp_path)
    assert not problems, (
        "table generators no longer reproduce tests/golden/tables.sha256:\n  "
        + "\n  ".join(problems)
    )


def test_figures_regenerate_identically(tmp_path):
    """Figures are compared as PNG; see the module docstring for why."""
    _requires_bulk()
    manifest = _manifest("figures.sha256")

    for module in ("fig4_dose_response", "fig5_acc_vs_disc"):
        _generate(module, "--out", str(tmp_path / f"{module}.pdf"))

    problems = _check_against(manifest, tmp_path)
    assert not problems, (
        "figure generators no longer reproduce tests/golden/figures.sha256:\n  "
        + "\n  ".join(problems)
    )
