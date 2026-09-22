"""Views over :mod:`anchorbench.registry`, kept under their long-standing names.

These used to be hand-maintained literals that repeated what
``conf/model/*.yaml`` and ``conf/data/*.yaml`` already say. They are now
derived from the registry, so adding a model or suite is a config edit and
every importer of this module sees it.
"""

from __future__ import annotations

from typing import Any

from anchorbench import registry

# -- Suite dataset paths (relative to repo root) -------------------

SUITE_DATASETS: dict[str, str] = {s.key: s.dataset_dir for s in registry.suites()}

VARIANT_DATASETS: dict[str, str] = {s.key: s.dataset_dir for s in registry.variants()}

# -- Model display names (slug on disk -> short name in tables) ----

MODEL_SHORT: dict[str, str] = {m.slug: m.short for m in registry.models()}

# -- Model identifiers -------------------------------------------------

API_MODEL_IDS: list[str] = [m.hf_id for m in registry.api_models()]

OW_MODEL_IDS: list[str] = [m.hf_id for m in registry.open_weight_models()]

# -- Result discovery order (pinned; see registry.RESULTS_DISCOVERY_ORDER) --

SUITES = registry.RESULTS_DISCOVERY_ORDER

# -- Defaults ------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "epsilon": 3.0,
    "max_tokens": 512,
    "batch_size": 32,
    "promptviews_file": "promptviews_core.jsonl",
    "itemspecs_file": "itemspecs.jsonl",
}
