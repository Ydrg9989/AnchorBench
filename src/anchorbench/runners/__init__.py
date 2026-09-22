"""Per-suite and per-variant evaluation runners.

Each module exposes a ``main()`` entry point and is addressed as
``python -m anchorbench.runners.<name>``; ``anchorbench eval`` and
``anchorbench experiment`` build those commands via :mod:`anchorbench.cli.cells`.
Argparse scaffolding and backend construction are shared through
:mod:`anchorbench.eval.runner_utils`.

Modules:
    external, icl, rag                     -- single-stage suites (one shared body,
                                              :mod:`anchorbench.runners._single_stage`)
    history                                -- two-stage protocol (anchor = own Stage-1 answer)
    tool                                   -- chat-template tool messages, plaintext fallback
    api                                    -- OpenRouter API benchmark, all suites
    icl_dist_api                           -- ICL distribution variant on API models
    sampling                               -- temperature/sampling robustness
    mitigation_baseline, mitigation_headroom -- prompt-based mitigation probes
    rebuttal_*                             -- appendix experiments (see docs/REPRODUCIBILITY.md)
"""

from __future__ import annotations
