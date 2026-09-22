"""Shared helpers for paper-artifact generators.

Centralises path resolution, model ordering, and a small set of
formatting helpers that are reused by every figure/table script.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from anchorbench import registry
from anchorbench.paths import OUTPUTS_DIR, RESULTS_DIR, ROOT

DEFAULT_OW_RESULTS = RESULTS_DIR / "full_benchmark"
DEFAULT_API_RESULTS = RESULTS_DIR / "api_benchmark"
DEFAULT_ICL_DIST_OW = RESULTS_DIR / "icl_dist_core" / "icl"
DEFAULT_ICL_DIST_API = RESULTS_DIR / "icl_dist_api"
DEFAULT_FIG_DIR = OUTPUTS_DIR / "figures"
DEFAULT_TABLE_DIR = OUTPUTS_DIR / "tables"

OW_MODELS_ORDER = [m.short for m in registry.open_weight_models()]
API_MODELS_ORDER = [m.short for m in registry.api_models()]
ALL_MODELS_ORDER = OW_MODELS_ORDER + API_MODELS_ORDER

OW_SLUGS = [m.slug for m in registry.open_weight_models()]
API_SLUGS = [m.slug for m in registry.api_models()]

# Suites by their unified_all_suites.json spelling, in paper order.
SUITES = [s.unified_key for s in registry.suites()]
SUITE_LATEX = {s.unified_key: s.latex for s in registry.suites()}

MODEL_LATEX = {m.short: m.latex for m in registry.models() if m.latex}


def slug_to_short(slug: str) -> str:
    """Map a model slug (e.g. 'Qwen_Qwen2.5-7B-Instruct') to its short name."""
    return registry.short_for_slug(slug)


def short_to_latex(name: str) -> str:
    """Return the LaTeX macro for a short-name model, or the literal name."""
    return MODEL_LATEX.get(name, name)


def fmt_num(v: float | None, decimals: int = 2, dash: str = "---") -> str:
    """Format a float with `decimals` precision, using LaTeX-safe minus."""
    if v is None:
        return dash
    try:
        if v != v:  # NaN
            return dash
    except TypeError:
        return dash
    s = f"{v:.{decimals}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    return s


def fmt_pct(v: float | None, dash: str = "---") -> str:
    """Format a 0-1 float as 'NN\\%' with LaTeX percent."""
    if v is None:
        return dash
    return f"{v * 100:.0f}\\%"


def fmt_pct1(v: float | None, dash: str = "---") -> str:
    """Format a 0-1 float as 'NN.N\\%'."""
    if v is None:
        return dash
    return f"{v * 100:.1f}\\%"


def rel_to_root(path: Path) -> Path:
    """Path relative to the repo for logging, or unchanged if outside it.

    Logging must never fail on an output path: writing to an --out_dir
    outside the repo made relative_to() raise *after* the file had already
    been written.
    """
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def write_table(path: Path, body: str) -> None:
    """Write a LaTeX table snippet to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body.rstrip() + "\n")
    print(f"  wrote {rel_to_root(path)}")


def collect_per_suite(unified: list[dict], suite: str) -> dict[str, dict]:
    """Return {short_model_name: row} for a given suite."""
    return {r["model"]: r for r in unified if r.get("suite") == suite}


def discover_jsonl(base: Path, suite: str) -> Iterable[tuple[str, str, Path]]:
    """Yield (short_model, slug, results_jsonl) for all models under base/suite/."""
    suite_dir = base / suite
    if not suite_dir.is_dir():
        return
    for model_dir in sorted(suite_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        slug = model_dir.name
        rpath = model_dir / "results.jsonl"
        if rpath.exists():
            yield slug_to_short(slug), slug, rpath
