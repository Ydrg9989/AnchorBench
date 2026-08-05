"""The committed datasets must stay regenerable from the current code.

These are the prompts the models actually saw, so if the generator drifts away
from them the paper stops being reproducible. Stage 1.1 of the camera-ready
audit established this baseline; see docs/RECONCILIATION.md D1-D3.

Two comparisons, for different reasons:

* `promptviews*.jsonl` are compared by hash. They are the model inputs and
  must match exactly.
* `itemspecs.jsonl` is compared field-by-field ignoring `generator_version`,
  which stamps the current git HEAD into every row (D2) and so changes the
  file hash on any commit without any data changing.
"""

import hashlib
import json
from pathlib import Path

import pytest

from anchorbench.data.generate import generate_suite_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

# Every suite now reaches 360 items from `--size core` alone: generate.py
# scales n_per_cell for suites without an offset dimension. History used to
# need an explicit n_per_cell=30 here (D1); that it no longer does, and still
# reproduces the committed data byte-for-byte, is what proves the fix changed
# the recipe without disturbing the data.
RECIPES: dict[str, int | None] = {
    "external": None,
    "history": None,
    "icl": None,
    "icl_dist": None,
    "rag": None,
    "tool": None,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture(scope="module", params=sorted(RECIPES))
def regenerated(request, tmp_path_factory):
    """Regenerate one suite into a temp dir. Never writes into datasets/."""
    suite = request.param
    committed = DATASETS / f"anchorbench_{suite}_core"
    if not committed.exists():
        pytest.skip(f"{committed.name} not present")
    out = tmp_path_factory.mktemp(f"regen_{suite}") / f"anchorbench_{suite}_core"
    generate_suite_dataset(
        suite=suite,
        size="core",
        seed=42,
        out_dir=str(out),
        n_per_cell=RECIPES[suite],
    )
    return suite, committed, out


def test_promptviews_reproduce_byte_for_byte(regenerated):
    suite, committed, out = regenerated
    checked = 0
    for name in ("promptviews.jsonl", "promptviews_core.jsonl", "promptviews_ablation.jsonl"):
        if not (committed / name).exists():
            continue
        assert (out / name).exists(), f"{suite}: regeneration did not produce {name}"
        assert _sha256(out / name) == _sha256(committed / name), (
            f"{suite}/{name} no longer regenerates byte-for-byte. The committed file is "
            f"the paper's ground truth -- do not overwrite it. See docs/RECONCILIATION.md."
        )
        checked += 1
    assert checked, f"{suite}: no promptview files were compared"


def test_uncertain_promptviews_reproduce():
    """The uncertain suite backs Table 2 in the MAIN paper, so it needs the
    same guarantee as the six core suites.

    It is produced differently from them: there is no
    `generate --suite external_uncertain`, and the directory holds only
    promptviews with no itemspecs of its own. The runner derives them from
    anchorbench_external_core/itemspecs.jsonl by re-rendering each external
    item at k = 1, 2, 3 visible ratings. Nothing carries
    suite="external_uncertain" at the itemspec level, which is why the
    renderer is deliberately absent from data.suites.SUITE_RENDERERS --
    registering it there would advertise a suite `generate` cannot produce.
    """
    from anchorbench.runners.rebuttal_uncertain import build_uncertain_promptviews

    committed = DATASETS / "anchorbench_external_uncertain" / "promptviews_uncertain.jsonl"
    core = DATASETS / "anchorbench_external_core"
    if not (committed.exists() and core.exists()):
        pytest.skip("uncertain or external core dataset not present")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        produced = build_uncertain_promptviews(core, Path(tmp))
        assert _sha256(produced) == _sha256(committed), (
            "promptviews_uncertain.jsonl no longer regenerates from "
            "anchorbench_external_core/itemspecs.jsonl. It backs Table 2 in "
            "the main paper; the committed file is ground truth."
        )


def test_itemspecs_reproduce_apart_from_provenance(regenerated):
    suite, committed, out = regenerated
    old, new = _rows(committed / "itemspecs.jsonl"), _rows(out / "itemspecs.jsonl")
    assert len(old) == len(new), (
        f"{suite}: itemspec count changed {len(old)} -> {len(new)}. "
        f"Check the n_per_cell recipe in this file against docs/RECONCILIATION.md D1."
    )
    for a, b in zip(old, new):
        a = {k: v for k, v in a.items() if k != "generator_version"}
        b = {k: v for k, v in b.items() if k != "generator_version"}
        assert a == b, f"{suite}: itemspec {a.get('item_id')} changed beyond generator_version"


def test_public_release_matches_the_evaluated_prompts():
    """Every released prompt must be the prompt the models actually saw.

    The previous exports had drifted badly -- only 90 to 165 of 1800 rows per
    suite still matched their source promptviews, because the evidence labels
    were regenerated after the exports were written. Publishing those would
    have shipped prompts no model was ever run on.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/export_public_promptviews.py", "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "datasets/hf_release/ is stale; run scripts/export_public_promptviews.py\n"
        f"{proc.stdout}\n{proc.stderr}"
    )

    release = ROOT / "datasets" / "hf_release"
    if not release.exists():
        pytest.skip("hf_release not generated")

    for suite in ("external", "history", "icl", "rag", "tool"):
        src_path = DATASETS / f"anchorbench_{suite}_core" / "promptviews_core.jsonl"
        out_path = release / f"{suite}.jsonl"
        if not (src_path.exists() and out_path.exists()):
            continue
        src = {}
        for line in src_path.read_text().splitlines():
            d = json.loads(line)
            src[(d["item_id"], d["condition"])] = d["prompt_text"]
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            key = (r["item_id"], r["condition"])
            assert key in src, f"{suite}: released row {key} has no source promptview"
            assert r["prompt_text"] == src[key], (
                f"{suite}: released prompt for {key} differs from the evaluated one"
            )


def test_history_release_reports_no_static_anchor():
    """History's anchor is the model's own Stage-1 answer, not an item property.

    The same item records anchor_value 80 for Qwen-7B and 81 for Llama-8B, so
    a single published value would misrepresent what the model saw. null is
    the honest answer; the realised value lives in the released results.
    """
    path = ROOT / "datasets" / "hf_release" / "history.jsonl"
    if not path.exists():
        pytest.skip("hf_release not generated")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows, "history release is empty"
    assert all(r["anchor_value"] is None for r in rows), (
        "history rows must not carry a static anchor_value"
    )
