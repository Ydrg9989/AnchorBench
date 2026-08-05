#!/usr/bin/env python3
"""Export the benchmark prompts for public / Hugging Face release.

Joins ``promptviews_core.jsonl`` (the rendered prompts) with
``itemspecs.jsonl`` (ground truth and anchor design) and writes one
self-contained row per prompt, so a user can run the benchmark and compute
UAI without needing any internal file.

Internal fields are dropped: ``prompt_hash`` and ``prompt_components`` are
generation bookkeeping, and ``anchor_string`` is the rendered sentence, which
is already visible inside ``prompt_text``.

Usage:
    python scripts/export_public_promptviews.py                      # all suites
    python scripts/export_public_promptviews.py --suites external    # one suite
    python scripts/export_public_promptviews.py --check              # staleness only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
DEFAULT_OUT = DATASETS / "hf_release"

# suite -> (dataset directory, promptviews file)
SUITES = {
    "external": ("anchorbench_external_core", "promptviews_core.jsonl"),
    "history": ("anchorbench_history_core", "promptviews_core.jsonl"),
    "icl": ("anchorbench_icl_core", "promptviews_core.jsonl"),
    "rag": ("anchorbench_rag_core", "promptviews_core.jsonl"),
    "tool": ("anchorbench_tool_core", "promptviews_core.jsonl"),
    # Backs Table 2 in the main paper. Its promptviews are derived from the
    # external itemspecs by runners/rebuttal_uncertain.py, so ground truth is
    # looked up in the external core directory.
    "external_uncertain": ("anchorbench_external_uncertain", "promptviews_uncertain.jsonl"),
}
ITEMSPEC_DIR = {"external_uncertain": "anchorbench_external_core"}


def anchor_value(itemspec: dict, condition: str, suite: str) -> int | None:
    """The numeric anchor placed in the prompt, or None where there isn't one.

    None in two distinct cases:

    * control conditions, which carry no anchor at all;
    * every History row. History's anchor is the model's *own* Stage-1 answer,
      so it is a property of the (item, model) run rather than of the item.
      Verified: the same item records anchor_value 80 for Qwen-7B and 81 for
      Llama-8B. The itemspec's anchors.low/high are the elicitation targets
      used to steer Stage 1, not the realised anchor, so publishing them here
      would misrepresent what the model actually saw. Read the realised value
      from the released results instead.
    """
    if suite == "history" or condition.startswith("control"):
        return None
    polarity = condition.rsplit("_", 1)[-1]
    return itemspec.get("anchors", {}).get(polarity)


def public_row(pv: dict, spec: dict) -> dict:
    condition = pv.get("condition", "")
    polarity = condition.rsplit("_", 1)[-1] if condition.endswith(("_low", "_high")) else None
    suite = pv.get("suite", "")
    anchors = spec.get("anchors", {})
    return {
        "item_id": pv.get("item_id"),
        "suite": suite,
        "condition": condition,
        "relevance": pv.get("anchor_relevance"),
        "anchor_polarity": polarity,
        "anchor_value": anchor_value(spec, condition, suite),
        "offset": anchors.get("offset"),
        "difficulty": spec.get("difficulty"),
        "domain": pv.get("domain"),
        "prompt_text": pv.get("prompt_text"),
        "y_star_evidence": spec.get("y_star_evidence"),
        # Eval-only benchmark: there is no train or validation portion. The
        # itemspecs carry tags.split = "core"/"pilot", which is a *size* label
        # and would be rendered by HF as a data split; it is deliberately not
        # propagated here.
        "split": "test",
    }


def build(suite: str) -> list[dict]:
    data_dir, pv_name = SUITES[suite]
    spec_dir = DATASETS / ITEMSPEC_DIR.get(suite, data_dir)
    specs = {}
    with open(spec_dir / "itemspecs.jsonl") as f:
        for line in f:
            d = json.loads(line)
            specs[d["item_id"]] = d
    rows = []
    with open(DATASETS / data_dir / pv_name) as f:
        for line in f:
            pv = json.loads(line)
            spec = specs.get(pv["item_id"])
            if spec is None:
                raise KeyError(f"{suite}: no itemspec for {pv['item_id']}")
            rows.append(public_row(pv, spec))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="*", default=sorted(SUITES),
                    help="Suites to export (default: all).")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="Report staleness without writing; exit 1 if stale.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stale = []
    for suite in args.suites:
        if suite not in SUITES:
            print(f"unknown suite: {suite}", file=sys.stderr)
            return 1
        rows = build(suite)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        dest = args.out_dir / f"{suite}.jsonl"
        if dest.exists() and dest.read_text() == text:
            print(f"  up to date  {suite:20s} {len(rows)} rows")
            continue
        stale.append(suite)
        if args.check:
            print(f"  STALE       {suite:20s} {len(rows)} rows")
        else:
            dest.write_text(text)
            print(f"  wrote       {suite:20s} {len(rows)} rows -> {dest.relative_to(ROOT)}")

    if args.check and stale:
        print(f"\n{len(stale)} export(s) stale. Run scripts/export_public_promptviews.py.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
