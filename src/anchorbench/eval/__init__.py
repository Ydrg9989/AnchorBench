"""AnchorBench shared evaluation package.

Provides parsing, metrics, backends, IO utilities, and the evaluator
orchestration used by all per-suite runners (``anchorbench.runners.*``).

Submodules:
    backends      -- vLLM and HuggingFace inference wrappers
    parsing       -- numeric extraction with optional LLM fallback
    metrics       -- TAR / UAI / Disc_Delta / MAE / Acc10
    evaluator     -- per-record evaluation loop
    io            -- JSONL + manifest helpers
    constants     -- shared model / suite / tier registries (configs/benchmark.yaml)
    runner_utils  -- argparse + backend factory helpers shared by runners
"""

from __future__ import annotations

from anchorbench import __version__

__all__ = ["__version__"]
