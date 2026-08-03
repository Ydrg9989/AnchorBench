"""Paper-artifact generators for AnchorBench (COLM 2026).

Each module reads existing per-record results JSONL files (no new
inference) and emits one or more paper figures or LaTeX tables.

Modules:
    fig4_dose_response  -- Figure 4 (UAI dose-response curves).
    fig5_acc_vs_disc    -- Figure 5 (accuracy vs discrimination scatter).
    tables_main         -- Table 1 (main results) and Table 2 (UAI by pathway).
    tables_appendix     -- Tables 4-16 plus boundary subtable and model details.
    verify              -- Data-driven verifier for every numeric paper claim.
"""

from __future__ import annotations
