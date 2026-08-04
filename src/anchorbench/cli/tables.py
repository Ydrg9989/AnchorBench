"""`anchorbench tables` subcommand.

Regenerates COLM 2026 figures and LaTeX tables from already-computed
``unified_all_suites.json`` files. No new inference is run.

Usage::

    anchorbench tables --paper             # full umbrella (figures + tables)
    anchorbench tables --figures           # Fig 4 + Fig 5 only
    anchorbench tables --tables-only       # tables_main + tables_appendix
    anchorbench tables --appendix          # the 13 \\input-ed appendix tables
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

# The 13 tables COLM_camera_ready/sections/appendix.tex \input-s, in the order
# they appear there. Each module already knows its own input and output paths;
# they were reachable only from scripts/rebuttal/*.sh, so regenerating the
# appendix meant running them by hand. See docs/APPENDIX_TABLES.md for the
# table -> module -> input mapping.
#
# These read results/rebuttal/**, which is published on Zenodo rather than
# committed, so they only run where that tree is present.
APPENDIX_MODULES = (
    "anchorbench.analysis.bayesian_bound",          # implied_weight + excess_uai
    "anchorbench.analysis.spectrum",
    "anchorbench.analysis.uncertain",
    "anchorbench.analysis.intensity_pathway",
    "anchorbench.analysis.extension_pilot",
    "anchorbench.analysis.weighted_mean",
    "anchorbench.analysis.cot_reasoning_extended",
    "anchorbench.analysis.task_spec",
    "anchorbench.analysis.rag_realism",
    "anchorbench.analysis.tool_realism",
    "anchorbench.analysis.large_panel",
    "anchorbench.analysis.case_studies",
    "anchorbench.analysis.cohens_d",
)


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
    p.add_argument("--appendix", action="store_true",
                   help="Regenerate the 13 appendix tables the camera-ready "
                        "\\input-s (needs results/rebuttal/).")
    p.add_argument("--skip-extensions", action="store_true",
                   help="Skip slow gold-shift / sampling / mitigation analyses.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    only_one = args.figures or args.tables_only or args.appendix
    figures = args.paper or args.figures or not only_one
    tables = args.paper or args.tables_only or not only_one

    rc = 0
    if args.appendix:
        if not (ROOT / "results" / "rebuttal").is_dir():
            print("results/rebuttal/ is absent; fetch the Zenodo bundles first "
                  "(scripts/make_zenodo_bundles.sh).", file=sys.stderr)
            return 1
        for mod in APPENDIX_MODULES:
            cmd = ["-m", mod]
            if args.dry_run:
                print(sys.executable, *cmd)
            else:
                rc |= _run(cmd)
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

    if not args.skip_extensions and not only_one:
        for mod in ("anchorbench.analysis.gold_shift",
                    "anchorbench.analysis.sampling",
                    "anchorbench.analysis.mitigation"):
            cmd = ["-m", mod]
            if args.dry_run:
                print(sys.executable, *cmd)
            else:
                _run(cmd)

    # --paper is documented as "figures + tables + verifier", so actually run
    # the verifier and let its exit code gate the command.
    if args.paper:
        cmd = ["-m", "anchorbench.paper.verify"]
        if args.dry_run:
            print(sys.executable, *cmd)
        else:
            rc |= _run(cmd)

    return rc
