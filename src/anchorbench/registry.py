"""The model panel and the benchmark suites, read once from ``conf/``.

Facts about the fourteen models (short name, results slug, LaTeX macro,
family, parameter count) used to live in six modules, and the suites'
names, labels and dataset paths in about twenty, with three spellings of
"ICL" that the verifier had to map between. The documented invariant is
"one model, one config file", so the config files are the source of truth:
``conf/model/<key>.yaml`` and ``conf/data/<key>.yaml`` carry the metadata
and ``conf/panel.yaml`` carries the order the paper prints them in.

Everything else derives from here. :mod:`anchorbench.eval.constants` keeps
its old names as views over this module for its many importers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from anchorbench.paths import CONF_DIR

# Order in which result directories are scanned, and therefore the order of
# entries in unified_all_suites.json. It predates the paper order below and
# is pinned by tests/golden/unified.sha256, so it stays as it is.
RESULTS_DISCOVERY_ORDER: tuple[str, ...] = ("external", "icl", "rag", "tool", "history")

API_BACKENDS = frozenset({"openrouter", "api"})


@dataclass(frozen=True)
class Model:
    """One evaluated model, as declared in ``conf/model/<key>.yaml``."""

    key: str
    name: str
    hf_id: str
    slug: str
    short: str
    backend: str
    family: str | None = None
    family_latex: str | None = None
    params: str | None = None
    latex: str | None = None
    config: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def is_api(self) -> bool:
        return self.backend.lower() in API_BACKENDS


@dataclass(frozen=True)
class Suite:
    """One benchmark suite or variant, as declared in ``conf/data/<key>.yaml``.

    ``suite`` is the runner and renderer name; a variant such as ``icl_dist``
    keeps ``suite = "icl"`` and only swaps the dataset. ``unified_key`` is the
    spelling used in ``unified_all_suites.json`` (``Icl``, ``Rag``);
    ``label`` is the human one (``ICL``, ``RAG``).
    """

    key: str
    suite: str
    variant: str | None
    label: str
    unified_key: str
    latex: str
    dataset_dir: str
    promptviews_file: str
    itemspecs_file: str


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The registry reads the Hydra config tree; point "
            "ANCHORBENCH_CONF at it if the package is not run from a checkout."
        )
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


@cache
def load_model(key: str) -> Model:
    cfg = _read(CONF_DIR / "model" / f"{key}.yaml")
    return Model(
        key=key,
        name=str(cfg["name"]),
        hf_id=str(cfg["hf_id"]),
        slug=str(cfg["slug"]),
        short=str(cfg["short"]),
        backend=str(cfg.get("backend", "vllm")),
        family=cfg.get("family"),
        family_latex=cfg.get("family_latex"),
        params=None if cfg.get("params") is None else str(cfg["params"]),
        latex=cfg.get("latex"),
        config=dict(cfg),
    )


@cache
def load_suite(key: str) -> Suite:
    cfg = _read(CONF_DIR / "data" / f"{key}.yaml")
    suite = str(cfg["suite"])
    return Suite(
        key=key,
        suite=suite,
        variant=cfg.get("variant"),
        label=str(cfg.get("label", key.capitalize())),
        unified_key=str(cfg.get("unified_key", suite.capitalize())),
        latex=str(cfg.get("latex", f"\\{suite}suite")),
        dataset_dir=str(cfg["dataset_dir"]),
        promptviews_file=str(cfg.get("promptviews_file", "promptviews_core.jsonl")),
        itemspecs_file=str(cfg.get("itemspecs_file", "itemspecs.jsonl")),
    )


@cache
def _panel() -> dict[str, list[str]]:
    return _read(CONF_DIR / "panel.yaml")


def open_weight_models() -> tuple[Model, ...]:
    return tuple(load_model(k) for k in _panel()["open_weight"])


def api_models() -> tuple[Model, ...]:
    return tuple(load_model(k) for k in _panel()["api"])


def models() -> tuple[Model, ...]:
    """The published panel in table order: open-weight models, then API models."""
    return open_weight_models() + api_models()


def suites() -> tuple[Suite, ...]:
    """The five paper suites in paper order."""
    return tuple(load_suite(k) for k in _panel()["suites"])


def variants() -> tuple[Suite, ...]:
    return tuple(load_suite(k) for k in _panel().get("variants", []))


def suite(key: str) -> Suite:
    return load_suite(key)


def model_by_slug(slug: str) -> Model | None:
    return next((m for m in models() if m.slug == slug), None)


def short_for_slug(slug: str) -> str:
    """Table label for a results-directory slug, or the slug itself if unknown."""
    m = model_by_slug(slug)
    return m.short if m else slug


def all_model_keys() -> list[str]:
    """Every ``conf/model/*.yaml`` stem, whether or not it is in the panel."""
    return sorted(p.stem for p in (CONF_DIR / "model").glob("*.yaml"))
