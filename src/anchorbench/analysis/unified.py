#!/usr/bin/env python3
"""Recompute all AnchorBench metrics using the unified framework.

Reads existing results.jsonl files (NO re-running of inference) and
produces a single comparison table across all suites and models.

Auto-discovers results from the directory structure:
  <results_dir>/<suite>/<model_slug>/results.jsonl
  <results_dir>/<suite>/<model_slug>/<model_slug>/results.jsonl  (icl/tool)

Usage:
    python -m anchorbench.analysis.unified \
        --results_dir results/full_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anchorbench.analysis._io import fmt
from anchorbench.eval.io import load_records
from anchorbench.eval.metrics import (
    baseline_condition,
    compute_by_difficulty,
    compute_by_offset,
    compute_extended_metrics,
)
from anchorbench.eval.runner_utils import discover_results
from anchorbench.paths import RESULTS_DIR


def _pct(v: float | None) -> str:
    return "---" if v is None else f"{v * 100:.1f}%"


def unified_rows(runs: list[tuple[str, str, str, Path]], epsilon: float) -> list[tuple[Path, dict]]:
    """One metrics row per discovered run, in discovery order: the extended
    metrics of Sec. 3.4 plus the by-offset and by-difficulty breakdowns,
    tagged with the capitalised suite name and the model's short name.
    Returns (results path, row) pairs; an empty results file is an error."""
    rows = []
    for suite, slug, short, path in runs:
        records = load_records(path)
        if not records:
            raise ValueError(f"empty results file: {path}")
        bc = baseline_condition(records)
        metrics = compute_extended_metrics(records, epsilon=epsilon, baseline_condition=bc)
        metrics["suite"] = suite.capitalize()
        metrics["model"] = short
        metrics["model_slug"] = slug
        metrics["n_records"] = len(records)
        metrics["by_offset"] = compute_by_offset(records, epsilon=epsilon, baseline_condition=bc)
        metrics["by_difficulty"] = compute_by_difficulty(records, epsilon=epsilon, baseline_condition=bc)
        rows.append((path, metrics))
    return rows


def write_unified(rows: list[tuple[Path, dict]], results_dir: Path) -> Path:
    """unified_summary.json next to each results.jsonl, and the one
    unified_all_suites.json every paper table reads, under results_dir."""
    for path, metrics in rows:
        with open(path.parent / "unified_summary.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    out = results_dir / "unified_all_suites.json"
    with open(out, "w") as f:
        json.dump([m for _, m in rows], f, indent=2, default=str)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Recompute unified metrics from existing results",
    )
    p.add_argument(
        "--results_dir", type=Path,
        default=RESULTS_DIR / "full_benchmark",
    )
    p.add_argument("--epsilon", type=float, default=3.0)
    args = p.parse_args()

    runs = discover_results(args.results_dir)
    if not runs:
        print(f"No results found in {args.results_dir}")
        print("Expected structure: <results_dir>/<suite>/<model_slug>/results.jsonl")
        sys.exit(1)

    print(f"Found {len(runs)} result files in {args.results_dir}")
    for suite, _slug, short, path in runs:
        print(f"  {suite:<10} {short:<16} {path}")
    print()

    rows = unified_rows(runs, args.epsilon)
    all_results = [m for _, m in rows]
    out_all = write_unified(rows, args.results_dir)

    # Per-suite tables
    for suite_name in ("External", "Icl", "Rag", "Tool", "History"):
        rows = [r for r in all_results if r["suite"] == suite_name]
        if not rows:
            continue
        rows.sort(key=lambda r: r.get("uai_plaus") or 0, reverse=True)

        print(f"\n{'=' * 105}")
        print(f"  {suite_name} Suite — Unified Metrics")
        print(f"{'=' * 105}")
        header = (
            f"{'Model':<16} {'N':>5} {'MAE_c':>6} {'Acc10_c':>8} {'UAI_irr':>8} "
            f"{'UAI_pls':>8} {'TAR_irr':>8} {'TAR_pls':>8} {'Disc_Δ':>8} {'Parse':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['model']:<16} "
                f"{r['n_records']:>5} "
                f"{fmt(r['mae_control']):>6} "
                f"{_pct(r['acc10_control']):>8} "
                f"{fmt(r['uai_irr'], 3):>8} "
                f"{fmt(r['uai_plaus'], 3):>8} "
                f"{fmt(r['tar_irr'], 3):>8} "
                f"{fmt(r['tar_plaus'], 3):>8} "
                f"{fmt(r['disc_delta'], 3):>8} "
                f"{_pct(r['parse_rate']):>7}"
            )
        print()

    # Cross-suite summary
    print("\n" + "=" * 105)
    print("  Cross-suite Summary (all suites, all models)")
    print("=" * 105)
    header = (
        f"{'Suite':<10} {'Model':<16} {'N':>5} {'MAE_c':>6} {'Acc10_c':>8} "
        f"{'UAI_irr':>8} {'UAI_pls':>8} {'Disc_Δ':>8} {'Parse':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in sorted(all_results, key=lambda r: (r["suite"], r["model"])):
        print(
            f"{r['suite']:<10} "
            f"{r['model']:<16} "
            f"{r['n_records']:>5} "
            f"{fmt(r['mae_control']):>6} "
            f"{_pct(r['acc10_control']):>8} "
            f"{fmt(r['uai_irr'], 3):>8} "
            f"{fmt(r['uai_plaus'], 3):>8} "
            f"{fmt(r['disc_delta'], 3):>8} "
            f"{_pct(r['parse_rate']):>7}"
        )

    # Per-offset breakdown table
    print("\n" + "=" * 115)
    print("  Per-offset breakdown (anchor distance analysis)")
    print("=" * 115)
    for r in all_results:
        by_off = r.get("by_offset", {})
        if not by_off:
            continue
        print(f"\n  {r['suite']} / {r['model']}:")
        print(f"    {'Offset':>8} {'UAI_irr':>8} {'UAI_pls':>8} {'TAR_irr':>8} {'TAR_pls':>8} {'N':>5}")
        for offset in sorted(by_off.keys(), key=int):
            m = by_off[offset]
            print(
                f"    {offset:>8} {fmt(m.get('uai_irr'), 3):>8} {fmt(m.get('uai_plaus'), 3):>8} "
                f"{fmt(m.get('tar_irr'), 3):>8} {fmt(m.get('tar_plaus'), 3):>8} {m.get('n_items', 0):>5}"
            )

    print(f"\n  Master JSON written to {out_all}")


if __name__ == "__main__":
    main()
