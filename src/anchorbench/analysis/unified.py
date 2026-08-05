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

from anchorbench.eval.io import load_records
from anchorbench.eval.metrics import (
    compute_by_difficulty,
    compute_by_offset,
    compute_extended_metrics,
)
from anchorbench.eval.runner_utils import (
    discover_results,
    fmt,
    fmt_pct,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Recompute unified metrics from existing results",
    )
    p.add_argument(
        "--results_dir", type=Path,
        default=Path(__file__).resolve().parents[3] / "results" / "full_benchmark",
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

    all_results = []
    for suite, slug, short, path in runs:
        records = load_records(path)
        if not records:
            print(f"  WARNING: empty {path}")
            continue

        conds_present = {r.get("condition") for r in records}
        if "control_twostage" in conds_present and "control" not in conds_present:
            bc = "control_twostage"
        else:
            bc = "control"

        metrics = compute_extended_metrics(records, epsilon=args.epsilon, baseline_condition=bc)
        metrics["suite"] = suite.capitalize()
        metrics["model"] = short
        metrics["model_slug"] = slug
        metrics["n_records"] = len(records)

        metrics["by_offset"] = compute_by_offset(records, epsilon=args.epsilon, baseline_condition=bc)
        metrics["by_difficulty"] = compute_by_difficulty(records, epsilon=args.epsilon, baseline_condition=bc)

        all_results.append(metrics)

        out_path = path.parent / "unified_summary.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

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
                f"{fmt_pct(r['acc10_control']):>8} "
                f"{fmt(r['uai_irr'], 3):>8} "
                f"{fmt(r['uai_plaus'], 3):>8} "
                f"{fmt(r['tar_irr'], 3):>8} "
                f"{fmt(r['tar_plaus'], 3):>8} "
                f"{fmt(r['disc_delta'], 3):>8} "
                f"{fmt_pct(r['parse_rate']):>7}"
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
            f"{fmt_pct(r['acc10_control']):>8} "
            f"{fmt(r['uai_irr'], 3):>8} "
            f"{fmt(r['uai_plaus'], 3):>8} "
            f"{fmt(r['disc_delta'], 3):>8} "
            f"{fmt_pct(r['parse_rate']):>7}"
        )

    # LaTeX table with CIs and significance markers
    print("\n" + "=" * 115)
    print("  LaTeX table rows with CIs (copy-paste into paper)")
    print("=" * 115)

    def fmt_ci(r: dict, key: str, decimals: int = 2) -> str:
        """Format a metric with CI: '0.42 [0.31, 0.53]'."""
        val = r.get(key)
        ci = r.get(f"{key}_ci")
        if val is None:
            return "---"
        v = f"{val:.{decimals}f}"
        if ci and ci.get("lo") is not None:
            return f"{v} [{ci['lo']:.{decimals}f}, {ci['hi']:.{decimals}f}]"
        return v

    def sig_marker(r: dict, key: str) -> str:
        """Return significance marker based on BH-corrected p-value."""
        p = r.get(f"{key}_bh", r.get(key))
        if p is None or (isinstance(p, float) and (p != p)):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    for suite_name in ("External", "Icl", "Rag", "Tool", "History"):
        rows = [r for r in all_results if r["suite"] == suite_name]
        if not rows:
            continue
        rows.sort(key=lambda r: r["model"])
        print(f"\n% --- {suite_name} suite ---")
        for r in rows:
            model = r["model"]
            mae = fmt_ci(r, "mae_control")
            acc = (
                f"{r['acc10_control'] * 100:.1f}\\%"
                if r["acc10_control"] is not None
                else "---"
            )
            ui = fmt(r["uai_irr"], 2) + sig_marker(r, "p_irr_vs_zero")
            up = fmt(r["uai_plaus"], 2) + sig_marker(r, "p_plaus_vs_zero")
            ti = fmt(r["tar_irr"], 2)
            tp = fmt(r["tar_plaus"], 2)
            dd = fmt(r["disc_delta"], 2) + sig_marker(r, "p_plaus_vs_irr")
            pr = (
                f"{r['parse_rate'] * 100:.1f}\\%"
                if r["parse_rate"] is not None
                else "---"
            )
            print(
                f"{model:<16} & {mae} & {acc} & {ui} & {up} & {ti} & {tp} & {dd} & {pr} \\\\"
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

    # Write master JSON
    out = args.results_dir / "unified_all_suites.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Master JSON written to {out}")


if __name__ == "__main__":
    main()
