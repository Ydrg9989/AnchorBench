"""Centralized constants for AnchorBench evaluation.

Loads suite dataset paths, model IDs, and display names from
``configs/benchmark.yaml`` so they are defined in one place.
Falls back to hardcoded defaults if the YAML is not found
(e.g. when the package is imported outside the repo tree).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "benchmark.yaml"


def _load_config() -> dict[str, Any]:
    if _CONFIG_PATH.is_file():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


_cfg = _load_config()

# ── Suite dataset paths (relative to repo root) ───────────────────

SUITE_DATASETS: dict[str, str] = _cfg.get("suite_datasets", {
    "external": "datasets/anchorbench_external_core",
    "history": "datasets/anchorbench_history_core",
    "icl": "datasets/anchorbench_icl_core",
    "rag": "datasets/anchorbench_rag_core",
    "tool": "datasets/anchorbench_tool_core",
})

VARIANT_DATASETS: dict[str, str] = _cfg.get("variant_datasets", {
    "icl_dist": "datasets/anchorbench_icl_dist_core",
})

# ── Model display names ───────────────────────────────────────────

MODEL_SHORT: dict[str, str] = _cfg.get("model_short_names", {
    "meta-llama_Llama-3.2-1B-Instruct": "Llama-1B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-8B",
    "Qwen_Qwen2.5-1.5B-Instruct": "Qwen-1.5B",
    "Qwen_Qwen2.5-3B-Instruct": "Qwen-3B",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen-7B",
    "google_gemma-3-1b-it": "Gemma-1B",
    "google_gemma-3-4b-it": "Gemma-4B",
    "allenai_OLMo-2-1124-13B-Instruct": "OLMo-13B",
    "allenai_OLMo-2-0325-32B-Instruct": "OLMo-32B",
    "openai_gpt-4o-mini": "GPT-4o-mini",
    "openai_gpt-5.4-mini": "GPT-5.4-mini",
    "google_gemini-2.5-flash": "Gemini-2.5-Flash",
    "anthropic_claude-haiku-4.5": "Claude-H4.5",
    "x-ai_grok-3-mini-beta": "Grok-3-mini",
})

# ── API model IDs ─────────────────────────────────────────────────

API_MODEL_IDS: list[str] = _cfg.get("api_models", [
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "x-ai/grok-3-mini-beta",
])

# ── Known suites for result discovery ─────────────────────────────

SUITES = ("external", "icl", "rag", "tool", "tool_agentic", "tool_read", "history")

# ── Defaults ──────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = _cfg.get("defaults", {
    "seed": 42,
    "epsilon": 3.0,
    "max_tokens": 512,
    "batch_size": 32,
    "promptviews_file": "promptviews_core.jsonl",
    "itemspecs_file": "itemspecs.jsonl",
})

# ── Open-weight HF Hub IDs ────────────────────────────────────────

OW_MODEL_IDS: list[str] = _cfg.get("ow_models", [])

# ── Model tiers (GPU scheduling) ──────────────────────────────────
# Each entry maps tier name -> {description, tensor_parallel, gpu_ids,
# models}. Used by ``scripts/run.py`` to schedule per-suite runners.

MODEL_TIERS: dict[str, dict[str, Any]] = _cfg.get("model_tiers", {})

# ── Named experiment recipes ──────────────────────────────────────
# Each recipe describes one paper run: which suites/models, which
# decoding settings, where to write outputs.

EXPERIMENTS: dict[str, dict[str, Any]] = _cfg.get("experiments", {})

# ── Frozen paper run pin ──────────────────────────────────────────

PAPER_RUN: dict[str, Any] = _cfg.get("paper_run", {})


def get_experiment(name: str) -> dict[str, Any]:
    """Return the named experiment recipe, raising if unknown."""
    if name not in EXPERIMENTS:
        known = ", ".join(sorted(EXPERIMENTS)) or "<none>"
        raise KeyError(f"Unknown experiment '{name}'. Known: {known}")
    return EXPERIMENTS[name]


def get_tier(name: str) -> dict[str, Any]:
    """Return the named model tier, raising if unknown."""
    if name not in MODEL_TIERS:
        known = ", ".join(sorted(MODEL_TIERS)) or "<none>"
        raise KeyError(f"Unknown model tier '{name}'. Known: {known}")
    return MODEL_TIERS[name]
