#!/usr/bin/env python3
"""Compare the numbers in the camera-ready against current generator output.

The paper pastes most appendix tables inline in sections/appendix.tex rather
than \\input-ing them, so the two can drift apart silently. This pulls each
labelled table's tabular body out of the .tex, does the same for the matching
outputs/tables/*.tex, and reports cell-by-cell differences.

This is what produces the D6 rows in docs/RECONCILIATION.md. It is a
measurement tool, not a gate: it never edits anything, and a difference it
reports is a fact to be recorded and explained, not something to suppress.

Usage:
    python scripts/measure_paper_drift.py            # summary
    python scripts/measure_paper_drift.py --verbose  # every differing cell

Regenerate outputs/tables/ first (`anchorbench tables --tables-only`) or the
comparison is against stale output.
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# As in sync_paper_tables.py: the LaTeX source lives with the arXiv
# submission, not here. ANCHORBENCH_PAPER_DIR points at a local checkout.
APPENDIX = Path(os.environ.get("ANCHORBENCH_PAPER_DIR",
                               ROOT / "COLM_camera_ready")) / "sections/appendix.tex"
GEN_DIR = ROOT / "outputs/tables"

# Paper label -> generator output. Only tables pasted inline in appendix.tex
# appear here; the 13 \input-ed ones cannot drift by construction.
MAPPING = {
    "tab:model-details": "tab_model_details.tex",
    "tab:app-external": "tab_app_external.tex",
    "tab:app-history": "tab_app_history.tex",
    "tab:app-icl": "tab_app_icl.tex",
    "tab:app-rag": "tab_app_rag.tex",
    "tab:app-tool": "tab_app_tool.tex",
    "tab:uai_summary": "tab_uai_summary.tex",
    "tab:stats_inference": "tab_stats_inference.tex",
    "tab:uai_distribution": "tab_uai_distribution.tex",
    "tab:anchored_mae": "tab_anchored_mae.tex",
    "tab:history_matched": "tab_history_matched.tex",
    "tab:icl_dist": "tab_icl_dist.tex",
    "tab:tool_plaintext": "tab_tool_plaintext.tex",
    "tab:difficulty": "tab_difficulty.tex",
}

NUMBER = re.compile(r"-?\d+\.\d+|-?\d+\\?%|-?\d+")


def table_block(text: str, label: str) -> str | None:
    """The \\begin{table...}...\\end{table...} block carrying `label`."""
    m = re.search(r"\\label\{" + re.escape(label) + r"\}", text)
    if not m:
        return None
    start = text.rfind("\\begin{table", 0, m.start())
    end = text.find("\\end{table", m.end())
    if start < 0 or end < 0:
        return None
    return text[start : text.find("}", end) + 1]


def numeric_rows(block: str) -> list[list[str]]:
    """Numeric cells per row of the tabular body, macros stripped."""
    body = re.search(r"\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}", block, re.S)
    if not body:
        return []
    rows = []
    for raw in body.group(1).split(r"\\"):
        raw = re.sub(r"%.*", "", raw)
        raw = re.sub(r"\\(midrule|toprule|bottomrule|cmidrule)(\(.*?\))?(\{.*?\})?", "", raw)
        raw = raw.replace("$-$", "-").replace("$", "").replace(r"\,", "")
        raw = re.sub(r"\\text[a-z]*\{(.*?)\}", r"\1", raw)
        raw = re.sub(r"\\[a-zA-Z]+", " ", raw)
        nums = []
        for cell in raw.split("&"):
            nums.extend(NUMBER.findall(cell.replace("\\%", "%")))
        if nums:
            rows.append(nums)
    return rows


def as_float(token: str) -> float | None:
    try:
        return float(token.replace("%", "").replace("\\", ""))
    except ValueError:
        return None


def compare(label: str, gen_name: str, appendix: str) -> tuple[str, str, list[str]]:
    """Return (status, detail, per-cell differences)."""
    block = table_block(appendix, label)
    gen_path = GEN_DIR / gen_name
    if block is None:
        return "paper table not found", "", []
    if not gen_path.exists():
        return "generator output missing", "", []

    paper_rows = numeric_rows(block)
    gen_rows = numeric_rows(gen_path.read_text())
    if len(paper_rows) != len(gen_rows):
        return "STRUCTURE", f"rows {len(paper_rows)} vs {len(gen_rows)}", []

    diffs, max_delta, n_cells = [], 0.0, 0
    for i, (pr, gr) in enumerate(zip(paper_rows, gen_rows)):
        if len(pr) != len(gr):
            return "STRUCTURE", f"row {i}: cells {len(pr)} vs {len(gr)}", []
        for j, (pv, gv) in enumerate(zip(pr, gr)):
            n_cells += 1
            if pv == gv:
                continue
            a, b = as_float(pv), as_float(gv)
            if a is None or b is None:
                diffs.append(f"row {i} col {j}: {pv!r} -> {gv!r}")
            else:
                max_delta = max(max_delta, abs(a - b))
                diffs.append(f"row {i} col {j}: {a} -> {b}  (delta {abs(a - b):.4g})")
    if not n_cells:
        # tabularx tables of model names and the like. Reporting these as
        # "identical" would be vacuous, so say so plainly.
        return "no numbers", "nothing to compare", []
    if not diffs:
        return "identical", f"{n_cells} cells", []
    return "DRIFT", f"{len(diffs)}/{n_cells} cells, max |delta| {max_delta:.4g}", diffs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verbose", action="store_true", help="list every differing cell")
    args = p.parse_args()

    if not APPENDIX.is_file():
        print(f"{APPENDIX} not found; set ANCHORBENCH_PAPER_DIR to a "
              "checkout of the paper source.", file=sys.stderr)
        return 1

    appendix = APPENDIX.read_text()
    print(f"{'paper table':26s} {'status':12s} detail")
    print("-" * 78)
    n_drift = 0
    for label, gen_name in MAPPING.items():
        status, detail, diffs = compare(label, gen_name, appendix)
        print(f"{label:26s} {status:12s} {detail}")
        if status in ("DRIFT", "STRUCTURE"):
            n_drift += 1
        if args.verbose and diffs:
            for d in diffs:
                print(f"    {d}")
    print("-" * 78)
    print(f"{len(MAPPING) - n_drift} of {len(MAPPING)} inline tables match current output.")
    print("Differences are recorded in docs/RECONCILIATION.md, not suppressed here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
