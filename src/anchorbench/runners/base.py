"""Shared scaffolding for per-suite runners.

Re-exports the helpers that previously lived in
``anchorbench_eval.runner_utils`` so all runners can import a single
``anchorbench.runners.base`` module instead of reaching across packages.
"""

from __future__ import annotations

from anchorbench.eval.runner_utils import (  # noqa: F401
    Backend,
    add_common_args,
    build_suffix,
    discover_results,
    make_backend,
    make_fallback,
    model_output_dir,
)

__all__ = [
    "Backend",
    "add_common_args",
    "build_suffix",
    "discover_results",
    "make_backend",
    "make_fallback",
    "model_output_dir",
]
