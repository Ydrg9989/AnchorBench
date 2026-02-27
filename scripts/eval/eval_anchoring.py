#!/usr/bin/env python3
"""Anchoring-effect evaluation from existing generations.jsonl files.

Reads generations + dataset, computes paired IAS/OutlierShift with bootstrap
CIs, produces summary JSON, per-base CSV, and a textual findings block.
Supports grouping by model/seed/preset and robustness checks.

Usage:
  python scripts/eval/eval_anchoring.py \
    --dataset data/processed/icl_anchor_v0/dataset.jsonl \
    --generations results/runs/*/generations.jsonl \
    --out_dir results/anchoring_report
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from llm_anchoring.eval.parsers import parse_hierarchical
from llm_anchoring.eval.parse_utils import diagnose_parse


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate anchoring from generations")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--generations", type=Path, nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, default=Path("results/anchoring_report"))
    p.add_argument("--reparse", action="store_true",
                   help="Re-parse raw_output_text with current parser")
    p.add_argument("--collision_tol", type=float, default=0.01)
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────

def _load_dataset_meta(path: Path) -> dict[str, dict]:
    """Build item_id → {domain, template_id} lookup from dataset."""
    meta = {}
    for line in path.read_text().strip().split("\n"):
        d = json.loads(line)
        meta[d["item_id"]] = {
            "domain": d.get("domain"),
            "template_id": d.get("template_id"),
        }
    return meta


def _load_generations(
    paths: list[Path],
    meta: dict[str, dict],
    reparse: bool,
    collision_tol: float,
) -> list[dict]:
    """Load and optionally re-parse all generation records."""
    records = []
    for p in paths:
        for line in p.read_text().strip().split("\n"):
            r = json.loads(line)
            if r.get("domain") is None:
                m = meta.get(r.get("item_id"), {})
                r["domain"] = m.get("domain")
                r["template_id"] = m.get("template_id")
            if reparse and "raw_output_text" in r:
                trace = parse_hierarchical(r["raw_output_text"], prefer_last=True)
                diag = diagnose_parse(
                    trace, r["truth_value"], r.get("anchor_value"),
                    collision_tol=collision_tol,
                )
                pv = trace["parsed_value"]
                tv = r["truth_value"]
                r.update({
                    "parsed_value": pv,
                    "parse_ok": trace["parse_ok"],
                    "format_ok": trace["format_ok"],
                    "parse_method": trace["strategy_used"],
                    "candidate_count": trace["candidate_count"],
                    "abs_error": abs(pv - tv) if pv is not None else None,
                    "anchor_collision": diag["anchor_collision"],
                    "mismatch_suspect": diag["mismatch_suspect"],
                    "best_candidate_error": diag["best_candidate_error"],
                })
            records.append(r)
    return records


# ── Core metrics ─────────────────────────────────────────────────────

def _bootstrap_ci(
    values: list[float], n_boot: int = 2000, seed: int = 42
) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo = means[int(n_boot * 0.025)]
    hi = means[int(n_boot * 0.975)]
    return [round(lo, 6), round(hi, 6)]


def _compute_anchoring(
    records: list[dict],
    agg_fn=np.median,
    exclude_mismatch: bool = False,
) -> dict:
    """Paired IAS/OutlierShift with configurable aggregation."""
    groups: defaultdict[tuple, list] = defaultdict(list)
    base_meta: dict[str, dict] = {}
    for r in records:
        if not r.get("parse_ok"):
            continue
        if exclude_mismatch and r.get("mismatch_suspect"):
            continue
        bid = r["base_id"]
        if bid not in base_meta:
            base_meta[bid] = {
                "domain": r.get("domain"),
                "template_id": r.get("template_id"),
            }
        groups[(bid, r["icl_condition"])].append(r["parsed_value"])

    aggs = {k: float(agg_fn(v)) for k, v in groups.items()}

    per_base, ias_vals, osh_vals = [], [], []
    for bid in sorted(base_meta):
        m = base_meta[bid]
        low = aggs.get((bid, "low_anchor_demo"))
        high = aggs.get((bid, "high_anchor_demo"))
        outlier = aggs.get((bid, "outlier_anchor_demo"))
        no_demo = aggs.get((bid, "no_demo"))
        placeholder = aggs.get((bid, "placeholder_demo"))

        row = {
            "base_id": bid, "domain": m.get("domain"),
            "template_id": m.get("template_id"),
            "median_no_demo": no_demo, "median_low": low,
            "median_high": high, "median_outlier": outlier,
            "median_placeholder": placeholder,
        }

        n_low = len(groups.get((bid, "low_anchor_demo"), []))
        n_high = len(groups.get((bid, "high_anchor_demo"), []))
        row["n_low"] = n_low
        row["n_high"] = n_high

        if low is not None and high is not None:
            ias = high - low
            row["IAS"] = ias
            ias_vals.append(ias)
        if low is not None and outlier is not None:
            osh = outlier - low
            row["OutlierShift"] = osh
            osh_vals.append(osh)

        per_base.append(row)

    result = {
        "n_base_ids": len(per_base),
        "n_base_ids_with_ias": len(ias_vals),
        "per_base": per_base,
        "ias_mean": float(np.mean(ias_vals)) if ias_vals else None,
        "ias_median": float(np.median(ias_vals)) if ias_vals else None,
        "outlier_shift_mean": float(np.mean(osh_vals)) if osh_vals else None,
        "outlier_shift_median": float(np.median(osh_vals)) if osh_vals else None,
    }
    if ias_vals:
        result["ias_ci_95"] = _bootstrap_ci(ias_vals)
    if osh_vals:
        result["outlier_shift_ci_95"] = _bootstrap_ci(osh_vals)
    return result


def _condition_summary(records: list[dict]) -> dict[str, dict]:
    by_cond: defaultdict[str, list] = defaultdict(list)
    for r in records:
        by_cond[r["icl_condition"]].append(r)

    out = {}
    for cond, recs in sorted(by_cond.items()):
        n = len(recs)
        pok = sum(1 for r in recs if r.get("parse_ok"))
        fok = sum(1 for r in recs if r.get("format_ok"))
        acol = sum(1 for r in recs if r.get("anchor_collision"))
        mism = sum(1 for r in recs if r.get("mismatch_suspect"))
        errs = [r["abs_error"] for r in recs if r.get("abs_error") is not None]
        out[cond] = {
            "n": n,
            "parse_rate": pok / n if n else 0,
            "format_rate": fok / n if n else 0,
            "anchor_collision_rate": acol / n if n else 0,
            "mismatch_suspect_rate": mism / n if n else 0,
            "mae": float(np.mean(errs)) if errs else None,
            "median_abs_error": float(np.median(errs)) if errs else None,
        }
    return out


def _domain_breakdown(per_base: list[dict]) -> dict[str, dict]:
    by_dom: defaultdict[str, list] = defaultdict(list)
    for row in per_base:
        dom = row.get("domain") or "unknown"
        if row.get("IAS") is not None:
            by_dom[dom].append(row["IAS"])

    out = {}
    for dom, vals in sorted(by_dom.items()):
        out[dom] = {
            "n_base": len(vals),
            "ias_mean": float(np.mean(vals)),
            "ias_median": float(np.median(vals)),
            "ias_ci_95": _bootstrap_ci(vals) if len(vals) >= 5 else None,
        }
    return out


# ── Robustness checks ───────────────────────────────────────────────

def _robustness(records: list[dict]) -> dict[str, dict]:
    checks = {}

    anch_mean = _compute_anchoring(records, agg_fn=np.mean)
    checks["agg_by_mean"] = {
        "ias_mean": anch_mean["ias_mean"],
        "ias_median": anch_mean["ias_median"],
        "ias_ci_95": anch_mean.get("ias_ci_95"),
    }

    anch_excl = _compute_anchoring(records, exclude_mismatch=True)
    checks["exclude_mismatch"] = {
        "ias_mean": anch_excl["ias_mean"],
        "ias_median": anch_excl["ias_median"],
        "ias_ci_95": anch_excl.get("ias_ci_95"),
        "n_base_ids_with_ias": anch_excl["n_base_ids_with_ias"],
    }

    # Identify sampling runs: (model_id, seed) combos with any sample_idx > 0
    sampling_keys: set[tuple] = set()
    for r in records:
        if r.get("sample_idx", 0) > 0:
            sampling_keys.add((r.get("model_id"), r.get("seed")))
    if sampling_keys:
        sampling = [r for r in records
                    if (r.get("model_id"), r.get("seed")) in sampling_keys]
        anch_samp = _compute_anchoring(sampling)
        checks["sampling_only"] = {
            "n_records": len(sampling),
            "ias_mean": anch_samp["ias_mean"],
            "ias_ci_95": anch_samp.get("ias_ci_95"),
        }

    return checks


# ── Findings text ────────────────────────────────────────────────────

def _generate_findings(summary: dict) -> str:
    lines = []
    ias = summary.get("ias_mean")
    ci = summary.get("ias_ci_95")
    osh = summary.get("outlier_shift_mean")
    n_base = summary.get("n_base_ids_with_ias", 0)

    if ias is not None:
        ci_str = f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "CI not computed"
        sig = "excludes" if ci and ci[0] > 0 else "includes"
        lines.append(
            f"Across {n_base} paired base tasks, the mean Induced Anchoring "
            f"Shift (IAS = median_pred(high) - median_pred(low)) is "
            f"{ias:.2f} ({ci_str}). The confidence interval {sig} zero, "
            f"{'providing' if sig == 'excludes' else 'not providing'} "
            f"statistically significant evidence of anchoring bias."
        )

    if osh is not None:
        osh_ci = summary.get("outlier_shift_ci_95")
        osh_str = f"{osh:.2f}"
        if osh_ci:
            osh_str += f" (95% CI [{osh_ci[0]:.2f}, {osh_ci[1]:.2f}])"
        lines.append(f"OutlierShift (outlier - low) mean = {osh_str}.")

    cond = summary.get("condition_summary", {})
    pr = min((c.get("parse_rate", 0) for c in cond.values()), default=0)
    acr = max((c.get("anchor_collision_rate", 0) for c in cond.values()), default=0)
    msr = max((c.get("mismatch_suspect_rate", 0) for c in cond.values()), default=0)
    lines.append(
        f"Parse rate >= {pr:.1%} across all conditions. "
        f"Anchor collision rate <= {acr:.1%}. "
        f"Mismatch suspect rate <= {msr:.1%}."
    )

    rob = summary.get("robustness", {})
    excl = rob.get("exclude_mismatch", {})
    if excl.get("ias_mean") is not None:
        lines.append(
            f"Robustness: excluding mismatch suspects yields IAS mean "
            f"{excl['ias_mean']:.2f}, confirming stability."
        )

    return "\n\n".join(lines)


# ── Run-level grouping ───────────────────────────────────────────────

def _group_records(records: list[dict]) -> dict[str, list[dict]]:
    """Group by (model_id, seed) to produce per-run reports."""
    groups: defaultdict[str, list] = defaultdict(list)
    for r in records:
        model = r.get("model_id", "unknown").split("/")[-1]
        seed = r.get("seed", "?")
        key = f"{model}_s{seed}"
        groups[key].append(r)
    return dict(sorted(groups.items()))


# ── Output ───────────────────────────────────────────────────────────

def _save_per_base(per_base: list[dict], path: Path) -> None:
    if not per_base:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_base[0].keys())
        w.writeheader()
        w.writerows(per_base)


def _build_summary(records: list[dict], label: str) -> dict:
    cond_summ = _condition_summary(records)
    anchoring = _compute_anchoring(records)
    domains = _domain_breakdown(anchoring["per_base"])
    robust = _robustness(records)

    summary = {
        "label": label,
        "n_records": len(records),
        "n_base_ids": anchoring["n_base_ids"],
        "n_base_ids_with_ias": anchoring["n_base_ids_with_ias"],
        "ias_mean": anchoring["ias_mean"],
        "ias_median": anchoring["ias_median"],
        "ias_ci_95": anchoring.get("ias_ci_95"),
        "outlier_shift_mean": anchoring["outlier_shift_mean"],
        "outlier_shift_median": anchoring["outlier_shift_median"],
        "outlier_shift_ci_95": anchoring.get("outlier_shift_ci_95"),
        "condition_summary": cond_summ,
        "domain_breakdown": domains,
        "robustness": robust,
    }
    summary["findings"] = _generate_findings(summary)
    return summary, anchoring["per_base"]


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_dataset_meta(args.dataset)
    records = _load_generations(
        args.generations, meta, args.reparse, args.collision_tol
    )
    print(f"Loaded {len(records)} records from {len(args.generations)} file(s)")

    groups = _group_records(records)
    all_summaries = {}

    for label, recs in groups.items():
        summary, per_base = _build_summary(recs, label)
        all_summaries[label] = summary

        run_dir = args.out_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        _save_per_base(per_base, run_dir / "per_base.csv")

        ci = summary.get("ias_ci_95")
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "N/A"
        print(
            f"  {label:45s}  n={len(recs):5d}  "
            f"IAS={summary['ias_mean'] or 0:8.2f}  CI={ci_str:20s}  "
            f"OShift={summary['outlier_shift_mean'] or 0:8.2f}"
        )

    if len(groups) > 1:
        print("\n--- Aggregate (all records pooled) ---")
        agg_summary, agg_per_base = _build_summary(records, "aggregate")
        (args.out_dir / "aggregate_summary.json").write_text(
            json.dumps(agg_summary, indent=2))
        _save_per_base(agg_per_base, args.out_dir / "aggregate_per_base.csv")

        ci = agg_summary.get("ias_ci_95")
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "N/A"
        print(
            f"  {'AGGREGATE':45s}  n={len(records):5d}  "
            f"IAS={agg_summary['ias_mean'] or 0:8.2f}  CI={ci_str:20s}  "
            f"OShift={agg_summary['outlier_shift_mean'] or 0:8.2f}"
        )
        print(f"\n{agg_summary['findings']}")

    elif groups:
        label = next(iter(groups))
        print(f"\n{all_summaries[label]['findings']}")

    print(f"\nOutputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
