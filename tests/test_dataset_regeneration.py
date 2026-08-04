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

# n_per_cell is the one recipe parameter that is not implied by --size core.
#
# The offset-grid suites get 6 domains x 2 difficulties x 3 anchor offsets x 10
# = 360 items from the `core` preset. History has no offset dimension
# (itemspec_gen.generate_history_itemspecs, anchor fixed at theta +/- 25), so
# the same preset yields 120 and it needs n_per_cell=30 to reach 360.
#
# The explicit 30 below records D1. When D1 is fixed so that `core` yields 360
# for every suite, this entry becomes None like the others -- and these
# assertions are what proves the fix did not disturb the data.
RECIPES = {
    "external": None,
    "history": 30,
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
