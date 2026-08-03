"""AnchorBench: a multi-paradigm benchmark for anchoring bias in LLMs.

The package is organized into clearly-scoped submodules:

    anchorbench.data        -- dataset generation (item specs, suites, validators)
    anchorbench.eval        -- metrics, evaluator, IO, vLLM/HF backends
    anchorbench.inference   -- shared inference helpers (OpenRouter async client)
    anchorbench.runners     -- per-suite and per-variant runners
    anchorbench.analysis    -- post-processing aggregators (unified, gold-shift, ...)
    anchorbench.paper       -- COLM 2026 figure + table generators and verifier
    anchorbench.cli         -- Hydra-driven entry points (`anchorbench` console script)

A frozen Hydra config tree lives at ``conf/`` in the repository root.
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__"]
