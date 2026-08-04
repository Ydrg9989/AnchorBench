#!/usr/bin/env python3
"""Copy generated table bodies into the paper source.

The camera-ready used to paste appendix tables inline in
sections/appendix.tex. Pasted numbers drift from the generator silently --
four tables had, by 0.01 to 0.06 -- while not one of the 13 \\input-ed tables
ever did. This closes that gap for the pasted ones.

Only the ``tabular`` body is copied. The ``\\begin{table}`` wrapper, the float
type, the ``\\caption`` and the ``\\label`` stay in appendix.tex, because the
paper's captions carry interpretation the generator has no business owning
("Hard items show stronger discrimination on External and RAG"). So:
generator owns the numbers, author owns the prose.

Usage:
    python scripts/sync_paper_tables.py            # copy, report what changed
    python scripts/sync_paper_tables.py --check    # exit 1 if out of date

Regenerate outputs/tables/ first (``anchorbench tables --tables-only``) or
this copies stale numbers.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = ROOT / "outputs" / "tables"
PAPER_DIR = ROOT / "COLM_camera_ready" / "tables"

# generator output -> body file \input-ed by appendix.tex.
#
# tab_history_matched and tab_tool_plaintext are deliberately absent: their
# input runs were never preserved, so the generator emits a single row and
# "---" placeholders. Syncing them would replace real published numbers with
# blanks. See docs/RECONCILIATION.md D5.
TABLES = {
    "tab_stats_inference.tex": "stats_inference_body.tex",
    "tab_uai_distribution.tex": "uai_distribution_body.tex",
    "tab_anchored_mae.tex": "anchored_mae_body.tex",
    "tab_difficulty.tex": "difficulty_body.tex",
}


def tabular_body(text: str) -> str:
    """The \\begin{tabular}...\\end{tabular} block, inclusive."""
    m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, re.S)
    if not m:
        raise ValueError("no tabular block found")
    return m.group(0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="report staleness without writing; exit 1 if stale")
    args = p.parse_args()

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for gen_name, body_name in TABLES.items():
        gen = GEN_DIR / gen_name
        if not gen.exists():
            print(f"  missing generator output: {gen.relative_to(ROOT)}", file=sys.stderr)
            return 1
        header = (f"% Generated from outputs/{gen_name} by "
                  f"scripts/sync_paper_tables.py -- do not edit by hand.\n")
        new = header + tabular_body(gen.read_text()) + "\n"
        dest = PAPER_DIR / body_name
        if dest.exists() and dest.read_text() == new:
            print(f"  up to date  {body_name}")
            continue
        stale.append(body_name)
        if args.check:
            print(f"  STALE       {body_name}")
        else:
            dest.write_text(new)
            print(f"  wrote       {body_name}")

    if args.check and stale:
        print(f"\n{len(stale)} table body/bodies are stale. Run "
              f"scripts/sync_paper_tables.py.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
