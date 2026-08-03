"""Per-suite and per-variant evaluation runners.

Each module exposes a ``main()`` entry point that the unified CLI
(``anchorbench eval``) dispatches to. Suite runners share argparse
scaffolding and backend construction via ``anchorbench.eval.runner_utils``
(re-exported here as :mod:`anchorbench.runners.base`).

Modules:
    external, history, icl, rag, tool      -- core paper suites
    api                                    -- OpenRouter API benchmark
    icl_dist_api                           -- ICL distribution variant (API)
    sampling                               -- temperature/sampling robustness
    mitigation_baseline, mitigation_headroom -- mitigation experiments
"""

from __future__ import annotations
