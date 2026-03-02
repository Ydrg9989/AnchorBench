#!/usr/bin/env python3
"""Compute PoC anchoring metrics from runner result JSONL files.

Usage:
  python runner/eval_poc.py \
      --results runner_outputs/poc_v0_hf.jsonl \
      --dataset poc_dataset/poc.jsonl

  python runner/eval_poc.py \
      --results runner_outputs/poc_v0_hf.jsonl runner_outputs/poc_v0_openrouter.jsonl \
      --dataset poc_dataset/poc.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: list[float], n_boot: int = 5000, seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    means = np.sort([float(arr[rng.randint(0, n, n)].mean()) for _ in range(n_boot)])
    lo = float(means[int(n_boot * 0.025)])
    hi = float(means[int(n_boot * 0.975)])
    return lo, hi


def _ols_slope(x: list[float], y: list[float]) -> tuple[float, float]:
    """Returns (alpha, beta) for y = alpha + beta * x."""
    xa, ya = np.array(x), np.array(y)
    xm, ym = xa.mean(), ya.mean()
    denom = ((xa - xm) ** 2).sum()
    if denom == 0:
        return float(ym), 0.0
    beta = float(((xa - xm) * (ya - ym)).sum() / denom)
    alpha = float(ym - beta * xm)
    return alpha, beta


# ── data loading ──────────────────────────────────────────────────────

def _load_results(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    log.info("Loaded %d result records from %d file(s)", len(records), len(paths))
    return records


def _build_gold_lookup(items: list[dict]) -> dict[str, dict]:
    return {it["item_id"]: it["gold"] for it in items}


# ── external suite metrics ────────────────────────────────────────────

def _eval_external(
    records: list[dict], gold: dict[str, dict],
) -> dict:
    """Compute per-item mean answers and anchoring shift for external suite."""
    by_item: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["suite"] != "external" or not r["parsed_ok"]:
            continue
        if r["condition"].endswith("_softEV"):
            continue
        by_item[r["item_id"]][r["condition"]].append(r["answer_int"])

    shifts, dev_highs, dev_lows = [], [], []
    per_item = []

    for iid in sorted(by_item):
        conds = by_item[iid]
        m_ctrl = np.mean(conds["control"]) if conds.get("control") else None
        m_low = np.mean(conds["low_anchor"]) if conds.get("low_anchor") else None
        m_high = np.mean(conds["high_anchor"]) if conds.get("high_anchor") else None
        g = gold.get(iid, {})
        y_star = g.get("y_star")

        row = {"item_id": iid, "y_star": y_star,
               "mean_control": _r(m_ctrl), "mean_low": _r(m_low), "mean_high": _r(m_high)}

        if m_low is not None and m_high is not None:
            shift = float(m_high - m_low)
            shifts.append(shift)
            row["shift_ext"] = round(shift, 4)
        if m_high is not None and m_ctrl is not None:
            dev_highs.append(float(m_high - m_ctrl))
        if m_ctrl is not None and m_low is not None:
            dev_lows.append(float(m_ctrl - m_low))
        per_item.append(row)

    ci = _bootstrap_ci(shifts) if shifts else (None, None)
    return {
        "n_items": len(per_item),
        "mean_shift_ext": _r(np.mean(shifts)) if shifts else None,
        "shift_ci_95": [round(ci[0], 4), round(ci[1], 4)] if ci[0] is not None else None,
        "mean_dev_high_minus_control": _r(np.mean(dev_highs)) if dev_highs else None,
        "mean_dev_control_minus_low": _r(np.mean(dev_lows)) if dev_lows else None,
        "per_item": per_item,
    }


# ── self-generated suite metrics ─────────────────────────────────────

def _eval_self(
    records: list[dict], gold: dict[str, dict],
) -> dict:
    """Compute delta_err = |err_hist| - |err_fresh| for self-generated suite."""
    by_item: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["suite"] != "self_generated" or not r["parsed_ok"]:
            continue
        if r["condition"].endswith("_softEV"):
            continue
        if r["condition"] in ("turn2_fresh", "turn2_history"):
            by_item[r["item_id"]][r["condition"]].append(r["answer_int"])

    delta_errs = []
    per_item = []

    for iid in sorted(by_item):
        g = gold.get(iid, {})
        y2 = g.get("y_star_stage2")
        if y2 is None:
            continue

        fresh = by_item[iid].get("turn2_fresh", [])
        hist = by_item[iid].get("turn2_history", [])
        if not fresh or not hist:
            continue

        m_fresh = np.mean(fresh)
        m_hist = np.mean(hist)
        err_fresh = abs(m_fresh - y2)
        err_hist = abs(m_hist - y2)
        delta = float(err_hist - err_fresh)
        delta_errs.append(delta)

        per_item.append({
            "item_id": iid, "y_star_stage2": y2,
            "mean_fresh": _r(m_fresh), "mean_hist": _r(m_hist),
            "err_fresh": round(err_fresh, 4), "err_hist": round(err_hist, 4),
            "delta_err": round(delta, 4),
        })

    ci = _bootstrap_ci(delta_errs) if delta_errs else (None, None)
    return {
        "n_items": len(per_item),
        "mean_delta_err": _r(np.mean(delta_errs)) if delta_errs else None,
        "delta_err_ci_95": [round(ci[0], 4), round(ci[1], 4)] if ci[0] is not None else None,
        "per_item": per_item,
    }


# ── controlled-history slope ─────────────────────────────────────────

def _eval_controlled(records: list[dict]) -> dict:
    """Fit answer = alpha + beta * anchor across controlled-history conditions."""
    ANCHORS = [10, 30, 50, 70, 90]
    by_item_anchor: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        if r["suite"] != "self_generated" or not r["parsed_ok"]:
            continue
        for a in ANCHORS:
            if r["condition"] == f"turn2_controlled_history_{a}":
                by_item_anchor[r["item_id"]][a].append(r["answer_int"])

    x_all, y_all = [], []
    per_item = []
    for iid in sorted(by_item_anchor):
        row = {"item_id": iid}
        for a in ANCHORS:
            vals = by_item_anchor[iid].get(a, [])
            if vals:
                m = float(np.mean(vals))
                row[f"mean_a{a}"] = round(m, 2)
                x_all.append(float(a))
                y_all.append(m)
        per_item.append(row)

    alpha, beta = _ols_slope(x_all, y_all) if x_all else (None, 0.0)

    # Per-field breakdown
    by_field: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for r in records:
        if r["suite"] != "self_generated" or not r["parsed_ok"]:
            continue
        for a in ANCHORS:
            if r["condition"] == f"turn2_controlled_history_{a}":
                xf, yf = by_field[r["field"]]
                xf.append(float(a))
                yf.append(float(r["answer_int"]))

    field_betas = {}
    for fld, (xf, yf) in sorted(by_field.items()):
        _, fb = _ols_slope(xf, yf) if xf else (None, 0.0)
        field_betas[fld] = round(fb, 6)

    return {
        "n_items": len(per_item),
        "alpha": round(alpha, 4) if alpha is not None else None,
        "beta": round(beta, 6),
        "field_betas": field_betas,
        "per_item": per_item,
    }


# ── per-model aggregation ────────────────────────────────────────────

def _split_by_model(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["model_id"]].append(r)
    return dict(sorted(groups.items()))


def _parse_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["parsed_ok"]) / len(records)


def _r(v) -> float | None:
    return round(float(v), 4) if v is not None else None


# ── output ────────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    all_keys = list(rows[0].keys())
    for row in rows[1:]:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--dataset", type=Path, default=Path("poc_dataset/poc.jsonl"))
    p.add_argument("--out_dir", type=Path, default=Path("runner_outputs"))
    args = p.parse_args()

    items = load_dataset(args.dataset)
    gold = _build_gold_lookup(items)
    records = _load_results(args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_model = _split_by_model(records)
    csv_rows = []
    full_summary: dict[str, dict] = {}

    for model_id, recs in by_model.items():
        model_short = model_id.split("/")[-1]
        log.info("=== %s  (%d records, parse_rate=%.1f%%) ===",
                 model_short, len(recs), _parse_rate(recs) * 100)

        ext = _eval_external(recs, gold)
        self_ = _eval_self(recs, gold)
        ctrl = _eval_controlled(recs)

        model_summary = {
            "model_id": model_id,
            "n_records": len(recs),
            "parse_rate": round(_parse_rate(recs), 4),
            "external": {k: v for k, v in ext.items() if k != "per_item"},
            "self_generated": {k: v for k, v in self_.items() if k != "per_item"},
            "controlled_history": ctrl,
        }
        full_summary[model_id] = model_summary

        # Console report
        log.info("  External:  shift=%.2f  CI=%s  (n=%d)",
                 ext["mean_shift_ext"] or 0,
                 ext.get("shift_ci_95", "N/A"),
                 ext["n_items"])
        log.info("  Self-gen:  delta_err=%.2f  CI=%s  (n=%d)",
                 self_["mean_delta_err"] or 0,
                 self_.get("delta_err_ci_95", "N/A"),
                 self_["n_items"])
        log.info("  Controlled: beta=%.4f  (n=%d)",
                 ctrl["beta"], ctrl["n_items"])
        if ctrl["field_betas"]:
            for fld, fb in ctrl["field_betas"].items():
                log.info("    %s: beta=%.4f", fld, fb)

        csv_rows.append({
            "model_id": model_id,
            "backend": recs[0].get("backend", ""),
            "n_records": len(recs),
            "parse_rate": round(_parse_rate(recs), 4),
            "ext_mean_shift": ext["mean_shift_ext"],
            "ext_shift_ci_lo": ext["shift_ci_95"][0] if ext["shift_ci_95"] else None,
            "ext_shift_ci_hi": ext["shift_ci_95"][1] if ext["shift_ci_95"] else None,
            "ext_mean_dev_high": ext["mean_dev_high_minus_control"],
            "ext_mean_dev_low": ext["mean_dev_control_minus_low"],
            "self_mean_delta_err": self_["mean_delta_err"],
            "self_delta_ci_lo": self_["delta_err_ci_95"][0] if self_["delta_err_ci_95"] else None,
            "self_delta_ci_hi": self_["delta_err_ci_95"][1] if self_["delta_err_ci_95"] else None,
            "ctrl_beta": ctrl["beta"],
            **{f"ctrl_beta_{fld}": fb for fld, fb in ctrl.get("field_betas", {}).items()},
        })

    # Determine run_name from first results file stem
    run_name = args.results[0].stem.replace("_hf", "").replace("_openrouter", "")

    summary_path = args.out_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(full_summary, indent=2, ensure_ascii=False))
    log.info("Summary → %s", summary_path)

    csv_path = args.out_dir / f"{run_name}_summary.csv"
    _write_csv(csv_rows, csv_path)
    log.info("CSV     → %s", csv_path)


if __name__ == "__main__":
    main()
