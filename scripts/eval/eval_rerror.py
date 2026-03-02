#!/usr/bin/env python3
"""Unified R-Error evaluation for both ICL-anchor and SynAnchors experiments.

Computes per-condition R-Error, paired t-test, Wilcoxon signed-rank test,
and occurrence criterion across models and conditions.

For ICL anchor data: compares each anchored condition against no_demo baseline.
For SynAnchors data: compares with_anchor against without_anchor.

Usage:
  # ICL anchor (full runs)
  python scripts/eval/eval_rerror.py \
    --generations results/runs/*/generations.jsonl \
    --experiment icl_anchor \
    --out_dir results/rerror_report

  # SynAnchors
  python scripts/eval/eval_rerror.py \
    --generations results/syn_anchors/*_generations.jsonl \
    --experiment syn_anchors \
    --out_dir results/rerror_report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R-Error evaluation")
    p.add_argument("--generations", type=Path, nargs="+", required=True)
    p.add_argument("--experiment", choices=["icl_anchor", "syn_anchors"], required=True)
    p.add_argument("--out_dir", type=Path, default=Path("results/rerror_report"))
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────

def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        source = p.parent.name if p.parent.name != "syn_anchors" else p.stem.replace("_generations", "")
        for line in p.read_text().strip().split("\n"):
            rec = json.loads(line)
            rec["_source"] = source
            records.append(rec)
    return records


def group_by_run(records: list[dict], experiment: str) -> dict[str, list[dict]]:
    """Group records by model run."""
    groups: defaultdict[str, list] = defaultdict(list)
    for r in records:
        key = r.get("_source", "unknown")
        groups[key].append(r)
    return dict(sorted(groups.items()))


# ── Core R-Error computation ─────────────────────────────────────────

def compute_rerror_paired(
    records: list[dict],
    experiment: str,
) -> dict[str, dict]:
    """Compute R-Error for each anchor condition vs baseline.

    Returns a dict keyed by condition with full stats.
    """
    if experiment == "icl_anchor":
        baseline_cond = "no_demo"
        anchor_conds = ["low_anchor_demo", "high_anchor_demo",
                        "outlier_anchor_demo", "placeholder_demo"]
        group_key = "base_id"
    else:
        baseline_cond = "without_anchor"
        anchor_conds = ["with_anchor"]
        group_key = "item_id"

    # Collect parsed values per (group_key, condition)
    vals: defaultdict[tuple, list] = defaultdict(list)
    for r in records:
        if not r.get("parse_ok"):
            continue
        if experiment == "icl_anchor":
            cond = r.get("icl_condition")
        else:
            cond = r.get("condition")
        gk = r.get(group_key)
        if cond and gk:
            vals[(gk, cond)].append(r["parsed_value"])

    # For each group, compute median per condition
    all_groups = sorted({k[0] for k in vals})

    results = {}
    for anchor_cond in anchor_conds:
        v_orig_list = []
        v_anchor_list = []
        r_errors = []
        per_group = []

        for gk in all_groups:
            baseline_vals = vals.get((gk, baseline_cond), [])
            anchor_vals = vals.get((gk, anchor_cond), [])
            if not baseline_vals or not anchor_vals:
                continue

            med_base = float(np.median(baseline_vals))
            med_anch = float(np.median(anchor_vals))
            v_orig_list.append(med_base)
            v_anchor_list.append(med_anch)

            denom = max(abs(med_base), 1e-9)
            r_err = abs(med_anch - med_base) / denom
            r_errors.append(r_err)

            shift = med_anch - med_base
            per_group.append({
                group_key: gk,
                "median_baseline": med_base,
                "median_anchor": med_anch,
                "r_error": r_err,
                "shift": shift,
                "n_baseline": len(baseline_vals),
                "n_anchor": len(anchor_vals),
            })

        n_pairs = len(v_orig_list)
        v_a = np.array(v_anchor_list)
        v_o = np.array(v_orig_list)

        # R-Error aggregates
        r_error_mean = float(np.mean(r_errors)) if r_errors else None
        r_error_median = float(np.median(r_errors)) if r_errors else None
        sorted_re = sorted(r_errors) if r_errors else []
        trim_n = max(1, int(len(sorted_re) * 0.95))
        r_error_trimmed = float(np.mean(sorted_re[:trim_n])) if sorted_re else None

        # Paired t-test
        if n_pairs >= 2:
            t_stat, p_t = stats.ttest_rel(v_a, v_o)
            t_stat, p_t = float(t_stat), float(p_t)
        else:
            t_stat, p_t = None, None

        # Wilcoxon signed-rank
        w_stat, p_w = None, None
        if n_pairs >= 2:
            diffs = v_a - v_o
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= 10:
                w_stat, p_w = stats.wilcoxon(nonzero)
                w_stat, p_w = float(w_stat), float(p_w)

        # Occurrence criterion
        occurrence = (
            p_t is not None and p_t < 0.05
            and r_error_trimmed is not None and r_error_trimmed > 0.2
        )

        # Direction: fraction of items where shift goes toward anchor
        # (only meaningful for ICL with known anchor values)
        n_positive_shift = sum(1 for pg in per_group if pg["shift"] > 0)

        results[anchor_cond] = {
            "n_paired_groups": n_pairs,
            "r_error_mean": r_error_mean,
            "r_error_median": r_error_median,
            "r_error_trimmed_95": r_error_trimmed,
            "paired_t_stat": t_stat,
            "paired_p_value": p_t,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p_value": p_w,
            "anchoring_present_ttest": p_t < 0.05 if p_t is not None else None,
            "anchoring_present_wilcoxon": p_w < 0.05 if p_w is not None else None,
            "occurrence": occurrence,
            "fraction_positive_shift": n_positive_shift / n_pairs if n_pairs else None,
            "per_group": per_group,
        }

    # Overall parse rate
    total = len(records)
    parsed = sum(1 for r in records if r.get("parse_ok"))

    return {
        "n_records": total,
        "parse_rate": parsed / total if total else 0,
        "conditions": results,
    }


# ── Report formatting ────────────────────────────────────────────────

def format_run_report(run_name: str, result: dict) -> str:
    lines = [
        f"{'=' * 80}",
        f"  {run_name}   (n={result['n_records']}  parse_rate={result['parse_rate']:.1%})",
        f"{'=' * 80}",
    ]
    for cond, s in result["conditions"].items():
        lines.append(f"\n  Condition: {cond}  (n_pairs={s['n_paired_groups']})")
        lines.append(f"  {'-' * 60}")
        if s["r_error_median"] is not None:
            lines.append(f"    R-Error  mean={s['r_error_mean']:.4f}  "
                         f"median={s['r_error_median']:.4f}  "
                         f"trimmed95={s['r_error_trimmed_95']:.4f}")
        if s["paired_t_stat"] is not None:
            lines.append(f"    t-test   t={s['paired_t_stat']:+.3f}   "
                         f"p={s['paired_p_value']:.4f}   "
                         f"sig={'YES' if s['anchoring_present_ttest'] else 'no'}")
        if s["wilcoxon_stat"] is not None:
            lines.append(f"    Wilcoxon W={s['wilcoxon_stat']:.1f}   "
                         f"p={s['wilcoxon_p_value']:.4f}   "
                         f"sig={'YES' if s['anchoring_present_wilcoxon'] else 'no'}")
        else:
            lines.append(f"    Wilcoxon: N/A")
        lines.append(f"    Occurrence (p<0.05 & R-Err>0.2): "
                     f"{'YES' if s['occurrence'] else 'NO'}")
        if s["fraction_positive_shift"] is not None:
            lines.append(f"    Positive shift rate: {s['fraction_positive_shift']:.1%}")
    lines.append("")
    return "\n".join(lines)


def format_grand_comparison(all_results: dict[str, dict], experiment: str) -> str:
    if experiment == "icl_anchor":
        conds = ["low_anchor_demo", "high_anchor_demo",
                 "outlier_anchor_demo", "placeholder_demo"]
    else:
        conds = ["with_anchor"]

    lines = [
        "",
        "=" * 100,
        f"GRAND COMPARISON — {experiment.upper()}",
        "=" * 100,
    ]

    for cond in conds:
        lines.append(f"\n  Condition: {cond}")
        hdr = (f"  {'Run':40s} {'R-Err(med)':>10s} {'R-Err(trim)':>11s} "
               f"{'t p-val':>8s} {'W p-val':>8s} {'Occur':>6s} {'Shift+':>7s}")
        lines.append(hdr)
        lines.append(f"  {'-'*40} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*6} {'-'*7}")

        for run_name, result in all_results.items():
            s = result["conditions"].get(cond)
            if s is None:
                continue
            med = f"{s['r_error_median']:.4f}" if s['r_error_median'] is not None else "N/A"
            trm = f"{s['r_error_trimmed_95']:.4f}" if s['r_error_trimmed_95'] is not None else "N/A"
            pt = f"{s['paired_p_value']:.4f}" if s['paired_p_value'] is not None else "N/A"
            pw = f"{s['wilcoxon_p_value']:.4f}" if s['wilcoxon_p_value'] is not None else "N/A"
            oc = "YES" if s["occurrence"] else "NO"
            sh = f"{s['fraction_positive_shift']:.0%}" if s['fraction_positive_shift'] is not None else "N/A"
            lines.append(f"  {run_name:40s} {med:>10s} {trm:>11s} {pt:>8s} {pw:>8s} {oc:>6s} {sh:>7s}")

    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.generations)
    print(f"Loaded {len(records)} records from {len(args.generations)} file(s)")

    groups = group_by_run(records, args.experiment)
    all_results: dict[str, dict] = {}

    for run_name, recs in groups.items():
        result = compute_rerror_paired(recs, args.experiment)
        all_results[run_name] = result

        # Save per-run summary (without bulky per_group data)
        summary = {run_name: {
            "n_records": result["n_records"],
            "parse_rate": result["parse_rate"],
            "conditions": {
                cond: {k: v for k, v in s.items() if k != "per_group"}
                for cond, s in result["conditions"].items()
            },
        }}
        run_path = args.out_dir / f"{run_name}_{args.experiment}.json"
        run_path.write_text(json.dumps(summary, indent=2))

        report = format_run_report(run_name, result)
        print(report)

    comparison = format_grand_comparison(all_results, args.experiment)
    print(comparison)

    # Save grand comparison
    comp = {}
    for run_name, result in all_results.items():
        comp[run_name] = {
            "n_records": result["n_records"],
            "parse_rate": result["parse_rate"],
            "conditions": {
                cond: {k: v for k, v in s.items() if k != "per_group"}
                for cond, s in result["conditions"].items()
            },
        }
    comp_path = args.out_dir / f"comparison_{args.experiment}.json"
    comp_path.write_text(json.dumps(comp, indent=2))
    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
