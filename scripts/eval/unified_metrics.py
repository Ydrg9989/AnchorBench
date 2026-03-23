#!/usr/bin/env python3
"""Unified metrics for AnchorBench — thin re-export from anchorbench_eval.

All logic now lives in src/anchorbench_eval/metrics.py and
src/anchorbench_eval/io.py. This file is kept for backward compatibility
so that existing scripts importing from unified_metrics still work.

Usage (unchanged):
    python scripts/eval/unified_metrics.py results/external_pilot/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench_eval.io import load_records
from anchorbench_eval.metrics import (
    CONDITIONS,
    EPSILON,
    bh_correction,
    bootstrap_ci,
    compute_by_difficulty,
    compute_by_offset,
    compute_extended_metrics,
    compute_unified_metrics,
    group_by_item,
    paired_wilcoxon,
    print_summary,
)

__all__ = [
    "CONDITIONS",
    "EPSILON",
    "load_records",
    "group_by_item",
    "compute_unified_metrics",
    "compute_extended_metrics",
    "compute_by_offset",
    "compute_by_difficulty",
    "print_summary",
    "bootstrap_ci",
    "paired_wilcoxon",
    "bh_correction",
]


def main() -> None:
    """CLI entry point: compute and optionally save unified metrics."""
    p = argparse.ArgumentParser(
        description="Compute unified AnchorBench metrics from results.jsonl",
    )
    p.add_argument("results", type=Path, help="Path to results.jsonl")
    p.add_argument("--epsilon", type=float, default=EPSILON)
    p.add_argument("--out", type=Path, default=None, help="Write unified summary JSON")
    p.add_argument("--label", type=str, default="")
    args = p.parse_args()

    records = load_records(args.results)
    metrics = compute_unified_metrics(records, epsilon=args.epsilon)
    print_summary(metrics, label=args.label or str(args.results))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Written to {args.out}")


if __name__ == "__main__":
    main()
