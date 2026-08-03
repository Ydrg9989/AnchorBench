"""Shared helpers for paper-artifact generators.

Centralises path resolution, model ordering, and a small set of
formatting helpers that are reused by every figure/table script.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchorbench.eval.constants import MODEL_SHORT  # noqa: E402

DEFAULT_OW_RESULTS = ROOT / "results" / "full_benchmark"
DEFAULT_API_RESULTS = ROOT / "results" / "api_benchmark"
DEFAULT_ICL_DIST_OW = ROOT / "results" / "icl_dist_core" / "icl"
DEFAULT_ICL_DIST_API = ROOT / "results" / "icl_dist_api"
DEFAULT_FIG_DIR = ROOT / "COLM" / "figures"
DEFAULT_TABLE_DIR = ROOT / "outputs" / "tables"

OW_MODELS_ORDER = [
    "Qwen-1.5B", "Qwen-3B", "Qwen-7B",
    "Llama-1B", "Llama-3B", "Llama-8B",
    "Gemma-1B", "Gemma-4B",
    "OLMo-13B", "OLMo-32B",
]

API_MODELS_ORDER = [
    "GPT-5.4-mini", "Claude-H4.5", "Gemini-2.5-Flash", "Grok-3-mini",
]

ALL_MODELS_ORDER = OW_MODELS_ORDER + API_MODELS_ORDER

OW_SLUGS = [
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct",
    "meta-llama_Llama-3.2-1B-Instruct",
    "meta-llama_Llama-3.2-3B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "google_gemma-3-1b-it",
    "google_gemma-3-4b-it",
    "allenai_OLMo-2-1124-13B-Instruct",
    "allenai_OLMo-2-0325-32B-Instruct",
]

API_SLUGS = [
    "openai_gpt-5.4-mini",
    "anthropic_claude-haiku-4.5",
    "google_gemini-2.5-flash",
    "x-ai_grok-3-mini-beta",
]

SUITES = ["External", "History", "Icl", "Rag", "Tool"]

SUITE_LATEX = {
    "External": r"\externalsuite",
    "History": r"\historysuite",
    "Icl": r"\iclsuite",
    "Rag": r"\ragsuite",
    "Tool": r"\toolsuite",
}

MODEL_LATEX = {
    "Qwen-1.5B":      r"\qwenonefive",
    "Qwen-3B":        r"\qwenthree",
    "Qwen-7B":        r"\qwenseven",
    "Llama-1B":       r"\llamaone",
    "Llama-3B":       r"\llamathree",
    "Llama-8B":       r"\llamaeight",
    "Gemma-1B":       r"\gemmaone",
    "Gemma-4B":       r"\gemmafour",
    "OLMo-13B":       r"\olmothirteen",
    "OLMo-32B":       r"\olmonthirtytwo",
    "GPT-5.4-mini":   r"\gptfivefourmini",
    "Claude-H4.5":    r"\claudehaiku",
    "Gemini-2.5-Flash": r"\geminiflash",
    "Grok-3-mini":    r"\grokthreemini",
}


def slug_to_short(slug: str) -> str:
    """Map a model slug (e.g. 'Qwen_Qwen2.5-7B-Instruct') to its short name."""
    return MODEL_SHORT.get(slug, slug)


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


def write_table(path: Path, body: str) -> None:
    """Write a LaTeX table snippet to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body.rstrip() + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")


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
