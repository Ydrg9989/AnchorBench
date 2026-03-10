#!/usr/bin/env python3
"""Unified metrics for AnchorBench: External, History, and RAG suites.

Computes a single set of metrics from any suite's results.jsonl:
  - Parse rate
  - MAE_control, Acc10_control   (control-only task accuracy)
  - UAI_irr, UAI_plaus           (Unified Anchor Influence)
  - TAR_irr, TAR_plaus           (Toward-Anchor Rate)
  - Disc_delta = UAI_plaus - UAI_irr

UAI definition (per item i, relevance r):
  UAI_{i,r} = (y_anchor - y_control) / (a - y_control)
  where a = anchor_value (External/RAG predefined; History = stage1_answer)
  Excluded when |a - y_control| < epsilon.

TAR definition:
  item counts as toward-anchor when (y_anchor - y_control) * (a - y_control) > 0

History-specific secondary metrics (for appendix):
  ACR = (stage2 - stage1) / (y_star - stage1)   [plausible only]
  RR  = 1 - |stage2 - y_star| / |stage1 - y_star|  [plausible only]

Usage:
  python scripts/eval/unified_metrics.py results/external_v2_pilot_Qwen25_7B_512/results.jsonl
  python scripts/eval/unified_metrics.py results/history_v2_pilot_Qwen25_7B_512/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

CONDITIONS = [
    "control",
    "irrelevant_low", "irrelevant_high",
    "plausible_low", "plausible_high",
]

EPSILON = 3.0


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: skipping malformed line {i+1} in {path}", file=sys.stderr)
    return records


def group_by_item(records: list[dict]) -> dict[str, dict[str, dict]]:
    """Group records by item_id -> condition -> record."""
    items: dict[str, dict[str, dict]] = {}
    for r in records:
        if not r.get("parsed_ok"):
            continue
        iid = r["item_id"]
        cond = r["condition"]
        if iid not in items:
            items[iid] = {}
        items[iid][cond] = r
    return items


def compute_unified_metrics(
    records: list[dict],
    epsilon: float = EPSILON,
) -> dict:
    n_total = len(records)
    n_parsed = sum(1 for r in records if r.get("parsed_ok") and r.get("answer_int") is not None)
    parse_rate = n_parsed / n_total if n_total else 0.0

    items = group_by_item(records)

    # --- Control-only accuracy ---
    mae_control_vals = []
    for iid, conds in items.items():
        ctrl = conds.get("control")
        if ctrl is None:
            continue
        y = ctrl.get("answer_int")
        y_star = ctrl.get("y_star_evidence")
        if y is None or y_star is None:
            continue
        mae_control_vals.append(abs(y - y_star))

    mae_control = float(np.mean(mae_control_vals)) if mae_control_vals else None
    acc10_control = (
        sum(1 for e in mae_control_vals if e <= 10) / len(mae_control_vals)
        if mae_control_vals else None
    )

    # --- UAI and TAR per condition ---
    uai_by_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS if c != "control"}
    tar_by_cond: dict[str, list[int]] = {c: [] for c in CONDITIONS if c != "control"}

    for iid, conds in items.items():
        ctrl = conds.get("control")
        if ctrl is None:
            continue
        y_ctrl = ctrl.get("answer_int")
        if y_ctrl is None:
            continue

        for cond in CONDITIONS:
            if cond == "control":
                continue
            rec = conds.get(cond)
            if rec is None:
                continue
            y_anchor = rec.get("answer_int")
            a = rec.get("anchor_value")
            if y_anchor is None or a is None:
                continue

            denom = a - y_ctrl
            shift = y_anchor - y_ctrl

            # TAR: toward-anchor if shift and denom have same sign
            tar_by_cond[cond].append(1 if shift * denom > 0 else 0)

            # UAI: skip if denominator too small
            if abs(denom) < epsilon:
                continue
            uai_by_cond[cond].append(shift / denom)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else None

    uai_irr_low = safe_mean(uai_by_cond["irrelevant_low"])
    uai_irr_high = safe_mean(uai_by_cond["irrelevant_high"])
    uai_plaus_low = safe_mean(uai_by_cond["plausible_low"])
    uai_plaus_high = safe_mean(uai_by_cond["plausible_high"])

    all_irr = uai_by_cond["irrelevant_low"] + uai_by_cond["irrelevant_high"]
    all_plaus = uai_by_cond["plausible_low"] + uai_by_cond["plausible_high"]
    uai_irr = safe_mean(all_irr)
    uai_plaus = safe_mean(all_plaus)

    tar_irr_low = safe_mean(tar_by_cond["irrelevant_low"])
    tar_irr_high = safe_mean(tar_by_cond["irrelevant_high"])
    tar_plaus_low = safe_mean(tar_by_cond["plausible_low"])
    tar_plaus_high = safe_mean(tar_by_cond["plausible_high"])

    all_tar_irr = tar_by_cond["irrelevant_low"] + tar_by_cond["irrelevant_high"]
    all_tar_plaus = tar_by_cond["plausible_low"] + tar_by_cond["plausible_high"]
    tar_irr = safe_mean(all_tar_irr)
    tar_plaus = safe_mean(all_tar_plaus)

    disc_delta = None
    if uai_irr is not None and uai_plaus is not None:
        disc_delta = uai_plaus - uai_irr

    # --- History-specific secondary metrics (ACR, RR) ---
    has_stage1 = any(r.get("stage1_answer") is not None for r in records)
    acr_vals = []
    rr_vals = []
    if has_stage1:
        for iid, conds in items.items():
            for cond in ("plausible_low", "plausible_high"):
                rec = conds.get(cond)
                if rec is None:
                    continue
                s1 = rec.get("stage1_answer")
                s2 = rec.get("answer_int")
                y_star = rec.get("y_star_evidence")
                if s1 is None or s2 is None or y_star is None:
                    continue
                if abs(y_star - s1) >= 1:
                    acr_vals.append((s2 - s1) / (y_star - s1))
                if abs(s1 - y_star) >= 1:
                    rr_vals.append(1.0 - abs(s2 - y_star) / abs(s1 - y_star))

    result = {
        "n_items": len(items),
        "n_records": n_total,
        "parse_rate": round(parse_rate, 4),
        "mae_control": round(mae_control, 2) if mae_control is not None else None,
        "acc10_control": round(acc10_control, 4) if acc10_control is not None else None,
        "uai_irr_low": round(uai_irr_low, 4) if uai_irr_low is not None else None,
        "uai_irr_high": round(uai_irr_high, 4) if uai_irr_high is not None else None,
        "uai_plaus_low": round(uai_plaus_low, 4) if uai_plaus_low is not None else None,
        "uai_plaus_high": round(uai_plaus_high, 4) if uai_plaus_high is not None else None,
        "uai_irr": round(uai_irr, 4) if uai_irr is not None else None,
        "uai_plaus": round(uai_plaus, 4) if uai_plaus is not None else None,
        "tar_irr_low": round(tar_irr_low, 4) if tar_irr_low is not None else None,
        "tar_irr_high": round(tar_irr_high, 4) if tar_irr_high is not None else None,
        "tar_plaus_low": round(tar_plaus_low, 4) if tar_plaus_low is not None else None,
        "tar_plaus_high": round(tar_plaus_high, 4) if tar_plaus_high is not None else None,
        "tar_irr": round(tar_irr, 4) if tar_irr is not None else None,
        "tar_plaus": round(tar_plaus, 4) if tar_plaus is not None else None,
        "disc_delta": round(disc_delta, 4) if disc_delta is not None else None,
        "epsilon": epsilon,
        "n_uai_irr": len(all_irr),
        "n_uai_plaus": len(all_plaus),
    }

    if has_stage1:
        result["acr_mean"] = round(safe_mean(acr_vals), 4) if acr_vals else None
        result["rr_mean"] = round(safe_mean(rr_vals), 4) if rr_vals else None

    return result


def print_summary(metrics: dict, label: str = ""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  Items:          {metrics['n_items']}")
    print(f"  Records:        {metrics['n_records']}")
    print(f"  Parse rate:     {metrics['parse_rate']:.2%}")
    print(f"  MAE_control:    {metrics['mae_control']}")
    print(f"  Acc10_control:  {metrics['acc10_control']}")
    print(f"  UAI_irr:        {metrics['uai_irr']}")
    print(f"  UAI_plaus:      {metrics['uai_plaus']}")
    print(f"  TAR_irr:        {metrics['tar_irr']}")
    print(f"  TAR_plaus:      {metrics['tar_plaus']}")
    print(f"  Disc_delta:     {metrics['disc_delta']}")
    if "acr_mean" in metrics:
        print(f"  ACR (plaus):    {metrics['acr_mean']}")
        print(f"  RR (plaus):     {metrics['rr_mean']}")
    print(f"  (epsilon={metrics['epsilon']}, n_uai_irr={metrics['n_uai_irr']}, n_uai_plaus={metrics['n_uai_plaus']})")
    print()


def main():
    p = argparse.ArgumentParser(description="Compute unified AnchorBench metrics from results.jsonl")
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
