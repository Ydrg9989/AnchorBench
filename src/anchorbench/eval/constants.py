"""Centralized constants for AnchorBench evaluation.

This module is the single source of truth for the legacy Python API
that needs to be importable without touching Hydra (e.g. for tests,
analysis scripts, and the paper-artifact generators). The full Hydra
config tree lives at ``conf/`` in the repository root and supersedes
this module for CLI-driven runs.
"""

from __future__ import annotations

from typing import Any

# -- Suite dataset paths (relative to repo root) -------------------

SUITE_DATASETS: dict[str, str] = {
    "external": "datasets/anchorbench_external_core",
    "history": "datasets/anchorbench_history_core",
    "icl": "datasets/anchorbench_icl_core",
    "rag": "datasets/anchorbench_rag_core",
    "tool": "datasets/anchorbench_tool_core",
}

VARIANT_DATASETS: dict[str, str] = {
    "icl_dist": "datasets/anchorbench_icl_dist_core",
}

# -- Model display names (slug on disk -> short name in tables) ----

MODEL_SHORT: dict[str, str] = {
    "Qwen_Qwen2.5-1.5B-Instruct": "Qwen-1.5B",
    "Qwen_Qwen2.5-3B-Instruct": "Qwen-3B",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen-7B",
    "meta-llama_Llama-3.2-1B-Instruct": "Llama-1B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-8B",
    "google_gemma-3-1b-it": "Gemma-1B",
    "google_gemma-3-4b-it": "Gemma-4B",
    "allenai_OLMo-2-1124-13B-Instruct": "OLMo-13B",
    "allenai_OLMo-2-0325-32B-Instruct": "OLMo-32B",
    "openai_gpt-4o-mini": "GPT-4o-mini",
    "openai_gpt-5.4-mini": "GPT-5.4-mini",
    "google_gemini-2.5-flash": "Gemini-2.5-Flash",
    "anthropic_claude-haiku-4.5": "Claude-H4.5",
    "x-ai_grok-3-mini-beta": "Grok-3-mini",
}

# -- API model IDs (OpenRouter) ------------------------------------

API_MODEL_IDS: list[str] = [
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "x-ai/grok-3-mini-beta",
]

# -- Open-weight HF Hub IDs ----------------------------------------

OW_MODEL_IDS: list[str] = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "allenai/OLMo-2-1124-13B-Instruct",
    "allenai/OLMo-2-0325-32B-Instruct",
]

# -- Known suites for result discovery -----------------------------

SUITES = ("external", "icl", "rag", "tool", "history")

# -- Defaults ------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "epsilon": 3.0,
    "max_tokens": 512,
    "batch_size": 32,
    "promptviews_file": "promptviews_core.jsonl",
    "itemspecs_file": "itemspecs.jsonl",
}
