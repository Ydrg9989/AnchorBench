"""`anchorbench tables` subcommand.

Regenerates COLM 2026 figures and LaTeX tables from already-computed
``unified_all_suites.json`` files. No new inference is run.

Usage::

    anchorbench tables --paper             # full umbrella (figures + tables)
    anchorbench tables --figures           # Fig 4 + Fig 5 only
    anchorbench tables --tables-only       # tables_main + tables_appendix
    anchorbench tables --skip-extensions   # skip slow gold/sampling/mitigation
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]


def _run(args: list[str]) -> int:
    log.info("$ %s", " ".join(args))
    return subprocess.run([sys.executable, *args]).returncode


def main() -> int:
    p = argparse.ArgumentParser(prog="anchorbench tables")
    p.add_argument("--paper", action="store_true",
                   help="Run figures + tables + verifier (default).")
    p.add_argument("--figures", action="store_true",
                   help="Only regenerate Fig 4 and Fig 5.")
    p.add_argument("--tables-only", action="store_true",
                   help="Only regenerate LaTeX tables.")
    p.add_argument("--skip-extensions", action="store_true",
                   help="Skip slow gold-shift / sampling / mitigation analyses.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    figures = args.paper or args.figures or not (args.tables_only)
    tables = args.paper or args.tables_only or not args.figures

    rc = 0
    if figures:
        for mod in ("anchorbench.paper.fig4_dose_response",
                    "anchorbench.paper.fig5_acc_vs_disc"):
            cmd = ["-m", mod]
            if args.dry_run:
                print(sys.executable, *cmd)
            else:
                rc |= _run(cmd)
    if tables:
        for mod in ("anchorbench.paper.tables_main",
                    "anchorbench.paper.tables_appendix"):
            cmd = ["-m", mod]
            if args.dry_run:
                print(sys.executable, *cmd)
            else:
                rc |= _run(cmd)

    if not args.skip_extensions and not args.figures and not args.tables_only:
        for mod in ("anchorbench.analysis.gold_shift",
                    "anchorbench.analysis.sampling",
                    "anchorbench.analysis.mitigation"):
            cmd = ["-m", mod]
            if args.dry_run:
                print(sys.executable, *cmd)
            else:
                _run(cmd)

    return rc
