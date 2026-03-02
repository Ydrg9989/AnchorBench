#!/usr/bin/env python3
"""Evaluate mitigation experiments with extended metrics.

Computes: NAI, TVD, W1, Relative Reduction, paired bootstrap CI,
Benjamini-Hochberg correction. Also computes post-hoc mitigations
(B4 self-consistency, CAC contrastive cancellation, PLD linear debiasing).

Usage:
  python mitigation/eval_mitigations.py \\
      --dataset poc_dataset/poc_v0.2.jsonl \\
      --baseline runner_outputs/api_v02.jsonl \\
      --mitigation runner_outputs/mit_b1.jsonl \\
      --run_name eval_b1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies import BETA_NORM, pld_correct
from utils import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXTERNAL_LIKE = {"external", "icl", "conversation_history", "rag", "tool"}
N_BOOT = 2000
BOOT_SEED = 42
ANCHOR_LOW, ANCHOR_HIGH = 20, 80


# ── helpers ───────────────────────────────────────────────────────────

def _r(v, d=4):
    return round(float(v), d) if v is not None else None


def _bootstrap_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    means = np.sort([float(arr[rng.randint(0, n, n)].mean()) for _ in range(n_boot)])
    return float(means[int(n_boot * 0.025)]), float(means[int(n_boot * 0.975)])


def _compute_tvd(a, b):
    if not a or not b:
        return 0.0
    bins = np.arange(102)
    pa = np.histogram(a, bins=bins)[0].astype(np.float64) / len(a)
    pb = np.histogram(b, bins=bins)[0].astype(np.float64) / len(b)
    return 0.5 * float(np.abs(pa - pb).sum())


def _compute_w1(a, b):
    if not a or not b:
        return 0.0
    bins = np.arange(102)
    ca = np.cumsum(np.histogram(a, bins=bins)[0].astype(np.float64)) / len(a)
    cb = np.cumsum(np.histogram(b, bins=bins)[0].astype(np.float64)) / len(b)
    return float(np.abs(ca - cb).sum())


def _bh_correct(pvalues: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of significant bools."""
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, pv) in enumerate(indexed, 1):
        threshold = q * rank / n
        if pv <= threshold:
            significant[orig_idx] = True
        else:
            break
    return significant


def _bootstrap_pvalue(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    """Two-sided p-value: fraction of bootstrap means on opposite side of 0."""
    rng = np.random.RandomState(seed)
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 1.0
    means = [float(arr[rng.randint(0, n, n)].mean()) for _ in range(n_boot)]
    observed = float(arr.mean())
    if observed >= 0:
        return 2 * sum(1 for m in means if m <= 0) / n_boot
    else:
        return 2 * sum(1 for m in means if m >= 0) / n_boot


# ── data loading ──────────────────────────────────────────────────────

def _load_results(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _build_gold(items):
    return {it["item_id"]: it for it in items}


# ── per-item NAI computation ─────────────────────────────────────────

def _compute_per_item_nai(
    records: list[dict], suite_name: str,
) -> dict[str, dict]:
    """Returns {item_id: {EV_control, EV_low, EV_high, deltaEV, NAI,
                          TVD, W1, low_vals, high_vals}}."""
    by_item: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["suite"] != suite_name or not r["parsed_ok"]:
            continue
        if r["condition"] in ("control", "low_anchor", "high_anchor"):
            by_item[r["item_id"]][r["condition"]].append(r["answer_int"])

    result = {}
    for iid in sorted(by_item):
        c = by_item[iid]
        ev_c = float(np.mean(c["control"])) if c.get("control") else None
        ev_l = float(np.mean(c["low_anchor"])) if c.get("low_anchor") else None
        ev_h = float(np.mean(c["high_anchor"])) if c.get("high_anchor") else None
        if ev_l is not None and ev_h is not None:
            dev = ev_h - ev_l
            nai = dev / (ANCHOR_HIGH - ANCHOR_LOW)
            tvd = _compute_tvd(c.get("low_anchor", []), c.get("high_anchor", []))
            w1 = _compute_w1(c.get("low_anchor", []), c.get("high_anchor", []))
        else:
            dev = nai = tvd = w1 = None
        result[iid] = {
            "EV_control": ev_c, "EV_low": ev_l, "EV_high": ev_h,
            "deltaEV": dev, "NAI": nai, "TVD": tvd, "W1": w1,
            "low_vals": c.get("low_anchor", []),
            "high_vals": c.get("high_anchor", []),
        }
    return result


# ── post-hoc mitigations ─────────────────────────────────────────────

def compute_b4_self_consistency(records: list[dict]) -> list[dict]:
    """Compute median answer per (item, condition) from multi-sample data."""
    by_key: dict[tuple, list[int]] = defaultdict(list)
    meta: dict[tuple, dict] = {}
    for r in records:
        if not r["parsed_ok"]:
            continue
        key = (r["model_id"], r["item_id"], r["condition"])
        by_key[key].append(r["answer_int"])
        meta[key] = r

    synthetic = []
    for key, vals in by_key.items():
        if len(vals) < 2:
            continue
        median_val = int(round(float(np.median(vals))))
        r = meta[key].copy()
        r["answer_int"] = median_val
        r["parsed_ok"] = True
        r["raw_text"] = f"SC_median({vals})"
        r["sample_idx"] = 0
        r["mitigation"] = "B4"
        synthetic.append(r)
    return synthetic


def compute_cac(records: list[dict]) -> list[dict]:
    """Contrastive Anchor Cancellation: midpoint of low and high answers."""
    by_key_low: dict[tuple, list[int]] = defaultdict(list)
    by_key_high: dict[tuple, list[int]] = defaultdict(list)
    meta: dict[tuple, dict] = {}

    for r in records:
        if not r["parsed_ok"]:
            continue
        key = (r["model_id"], r["item_id"], r["sample_idx"])
        if r["condition"] == "low_anchor":
            by_key_low[key].append(r["answer_int"])
            meta[key] = r
        elif r["condition"] == "high_anchor":
            by_key_high[key].append(r["answer_int"])

    synthetic = []
    for key in by_key_low:
        lows = by_key_low[key]
        highs = by_key_high.get(key, [])
        if not lows or not highs:
            continue
        cac_val = int(round((np.mean(lows) + np.mean(highs)) / 2))
        r = meta[key].copy()
        r["answer_int"] = cac_val
        r["parsed_ok"] = True
        r["condition"] = "CAC"
        r["raw_text"] = f"CAC_mid(low={lows},high={highs})"
        r["mitigation"] = "CAC"
        synthetic.append(r)
    return synthetic


def compute_pld(records: list[dict]) -> list[dict]:
    """Post-hoc Linear Debiasing using calibrated beta."""
    synthetic = []
    for r in records:
        if not r["parsed_ok"] or r["answer_int"] is None:
            continue
        if r["condition"] == "low_anchor":
            anchor_val = ANCHOR_LOW
        elif r["condition"] == "high_anchor":
            anchor_val = ANCHOR_HIGH
        else:
            continue
        corrected = pld_correct(r["answer_int"], anchor_val, r["model_id"])
        rec = r.copy()
        rec["answer_int"] = corrected
        rec["raw_text"] = f"PLD({r['answer_int']}→{corrected})"
        rec["mitigation"] = "PLD"
        synthetic.append(rec)
    return synthetic


# ── comparative evaluation ────────────────────────────────────────────

def _eval_suite_comparison(
    baseline_nai: dict[str, dict],
    mitigated_nai: dict[str, dict],
    suite: str,
    mitigation_name: str,
) -> dict:
    """Compare baseline vs mitigated NAI per item."""
    common = set(baseline_nai) & set(mitigated_nai)
    if not common:
        return {"n_items": 0}

    delta_nais = []
    rrs = []
    items_detail = []

    for iid in sorted(common):
        b = baseline_nai[iid]
        m = mitigated_nai[iid]
        if b["NAI"] is None or m["NAI"] is None:
            continue
        delta = b["NAI"] - m["NAI"]
        delta_nais.append(delta)
        rr = 1 - m["NAI"] / b["NAI"] if abs(b["NAI"]) > 1e-6 else 0.0
        rrs.append(rr)
        items_detail.append({
            "item_id": iid,
            "NAI_baseline": _r(b["NAI"]),
            "NAI_mitigated": _r(m["NAI"]),
            "delta_NAI": _r(delta),
            "RR": _r(rr),
        })

    if not delta_nais:
        return {"n_items": 0}

    ci = _bootstrap_ci(delta_nais)
    rr_ci = _bootstrap_ci(rrs)
    pval = _bootstrap_pvalue(delta_nais)

    return {
        "suite": suite,
        "mitigation": mitigation_name,
        "n_items": len(delta_nais),
        "mean_NAI_baseline": _r(np.mean([b["NAI"] for iid in sorted(common)
                                          if (b := baseline_nai[iid])["NAI"] is not None
                                          and mitigated_nai[iid]["NAI"] is not None])),
        "mean_NAI_mitigated": _r(np.mean([mitigated_nai[iid]["NAI"] for iid in sorted(common)
                                           if baseline_nai[iid]["NAI"] is not None
                                           and mitigated_nai[iid]["NAI"] is not None])),
        "mean_delta_NAI": _r(np.mean(delta_nais)),
        "delta_NAI_ci95": [_r(ci[0]), _r(ci[1])],
        "mean_RR": _r(np.mean(rrs)),
        "RR_ci95": [_r(rr_ci[0]), _r(rr_ci[1])],
        "p_value": _r(pval, 6),
        "per_item": items_detail,
    }


# ── accuracy metric ──────────────────────────────────────────────────

def _compute_accuracy(nai_data, gold_lookup):
    """Mean |EV_control - y_star| for items with gold."""
    errs = []
    for iid, d in nai_data.items():
        if d["EV_control"] is None:
            continue
        item_data = gold_lookup.get(iid)
        if not item_data:
            continue
        y_star = item_data.get("gold", {}).get("y_star")
        if y_star is None:
            continue
        errs.append(abs(d["EV_control"] - y_star))
    return _r(np.mean(errs)) if errs else None


# ── main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate mitigation experiments")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--baseline", type=Path, nargs="+", required=True,
                   help="Baseline (B0) result file(s)")
    p.add_argument("--mitigation", type=Path, nargs="*", default=[],
                   help="Mitigation result file(s) (prompt-based)")
    p.add_argument("--posthoc", type=str, nargs="*", default=[],
                   choices=["B4", "CAC", "PLD"],
                   help="Post-hoc mitigations to compute from baseline")
    p.add_argument("--run_name", type=str, default="eval_mit")
    p.add_argument("--out_dir", type=Path, default=Path("runner_outputs"))
    args = p.parse_args()

    items = load_dataset(args.dataset)
    gold = _build_gold(items)

    baseline_records = _load_results(args.baseline)
    log.info("Baseline: %d records", len(baseline_records))

    mitigated_records = _load_results(args.mitigation) if args.mitigation else []
    if mitigated_records:
        log.info("Mitigation: %d records", len(mitigated_records))

    # Post-hoc mitigations from baseline
    posthoc_records: dict[str, list[dict]] = {}
    if "B4" in args.posthoc:
        posthoc_records["B4"] = compute_b4_self_consistency(baseline_records)
        log.info("B4 (SC): %d synthetic records", len(posthoc_records["B4"]))
    if "CAC" in args.posthoc:
        posthoc_records["CAC"] = compute_cac(baseline_records)
        log.info("CAC: %d synthetic records", len(posthoc_records["CAC"]))
    if "PLD" in args.posthoc:
        posthoc_records["PLD"] = compute_pld(baseline_records)
        log.info("PLD: %d synthetic records", len(posthoc_records["PLD"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_comparisons: list[dict] = []
    all_pvalues: list[float] = []

    # Split baseline by model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in baseline_records:
        by_model[r["model_id"]].append(r)

    # Split mitigation by model
    mit_by_model: dict[str, list[dict]] = defaultdict(list)
    for r in mitigated_records:
        mit_by_model[r["model_id"]].append(r)

    full_summary: dict[str, Any] = {}

    for model_id in sorted(by_model):
        model_short = model_id.split("/")[-1]
        b_recs = by_model[model_id]
        log.info("=== %s ===", model_short)

        model_results: dict[str, Any] = {"model_id": model_id}

        for suite in sorted(EXTERNAL_LIKE):
            suite_b_recs = [r for r in b_recs if r["suite"] == suite]
            if not suite_b_recs:
                continue

            b_nai = _compute_per_item_nai(suite_b_recs, suite)
            if not b_nai:
                continue

            mean_b_nai = np.mean([v["NAI"] for v in b_nai.values() if v["NAI"] is not None])
            log.info("  %s baseline NAI=%.4f (n=%d)", suite, mean_b_nai, len(b_nai))

            # Prompt-based mitigation
            m_recs = [r for r in mit_by_model.get(model_id, []) if r["suite"] == suite]
            if m_recs:
                mit_name = m_recs[0].get("mitigation", "unknown")
                m_nai = _compute_per_item_nai(m_recs, suite)
                comp = _eval_suite_comparison(b_nai, m_nai, suite, mit_name)
                all_comparisons.append(comp)
                if comp.get("p_value") is not None:
                    all_pvalues.append(comp["p_value"])
                log.info("    %s: NAI %.4f→%.4f  RR=%.2f  p=%.4f",
                         mit_name,
                         comp.get("mean_NAI_baseline", 0) or 0,
                         comp.get("mean_NAI_mitigated", 0) or 0,
                         comp.get("mean_RR", 0) or 0,
                         comp.get("p_value", 1))

            # Post-hoc mitigations
            for ph_name, ph_recs in posthoc_records.items():
                ph_model = [r for r in ph_recs
                            if r["model_id"] == model_id and r["suite"] == suite]
                if not ph_model:
                    continue

                if ph_name == "CAC":
                    cac_nai = _compute_cac_nai(ph_model, b_nai, suite)
                    comp = _eval_suite_comparison(b_nai, cac_nai, suite, "CAC")
                elif ph_name == "PLD":
                    pld_nai = _compute_pld_nai(ph_model, suite)
                    comp = _eval_suite_comparison(b_nai, pld_nai, suite, "PLD")
                elif ph_name == "B4":
                    sc_nai = _compute_per_item_nai(ph_model, suite)
                    comp = _eval_suite_comparison(b_nai, sc_nai, suite, "B4")
                else:
                    continue

                all_comparisons.append(comp)
                if comp.get("p_value") is not None:
                    all_pvalues.append(comp["p_value"])
                log.info("    %s: NAI %.4f→%.4f  RR=%.2f  p=%.4f",
                         ph_name,
                         comp.get("mean_NAI_baseline", 0) or 0,
                         comp.get("mean_NAI_mitigated", 0) or 0,
                         comp.get("mean_RR", 0) or 0,
                         comp.get("p_value", 1))

        full_summary[model_id] = model_results

    # BH correction
    if all_pvalues:
        sig_flags = _bh_correct(all_pvalues)
        for comp, sig in zip(all_comparisons, sig_flags):
            comp["significant_BH"] = sig

    # Parse rate comparison
    parse_baseline = sum(1 for r in baseline_records if r["parsed_ok"]) / max(len(baseline_records), 1)
    parse_mitigated = (sum(1 for r in mitigated_records if r["parsed_ok"]) / max(len(mitigated_records), 1)
                       if mitigated_records else None)
    log.info("Parse rate: baseline=%.1f%%  mitigated=%s",
             parse_baseline * 100,
             f"{parse_mitigated * 100:.1f}%" if parse_mitigated else "N/A")

    # Output
    output = {
        "comparisons": [
            {k: v for k, v in c.items() if k != "per_item"}
            for c in all_comparisons
        ],
        "comparisons_detail": all_comparisons,
        "parse_rate_baseline": _r(parse_baseline),
        "parse_rate_mitigated": _r(parse_mitigated),
    }

    json_path = args.out_dir / f"{args.run_name}_metrics.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("JSON → %s", json_path)

    csv_rows = [
        {k: v for k, v in c.items() if k != "per_item"}
        for c in all_comparisons
    ]
    if csv_rows:
        csv_path = args.out_dir / f"{args.run_name}_metrics.csv"
        keys = list(csv_rows[0].keys())
        for row in csv_rows[1:]:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)
        log.info("CSV  → %s", csv_path)


def _compute_cac_nai(cac_records, baseline_nai, suite):
    """Build NAI dict for CAC from synthetic records + baseline control."""
    by_item: dict[str, list[int]] = defaultdict(list)
    for r in cac_records:
        if r["parsed_ok"] and r["condition"] == "CAC":
            by_item[r["item_id"]].append(r["answer_int"])

    result = {}
    for iid, vals in by_item.items():
        ev_cac = float(np.mean(vals))
        b = baseline_nai.get(iid, {})
        ev_ctrl = b.get("EV_control")
        if ev_ctrl is not None:
            dev = abs(ev_cac - ev_ctrl)
            nai = dev / (ANCHOR_HIGH - ANCHOR_LOW)
        else:
            nai = 0.0
        result[iid] = {
            "EV_control": ev_ctrl, "EV_low": ev_cac, "EV_high": ev_cac,
            "deltaEV": 0.0, "NAI": 0.0,
            "TVD": 0.0, "W1": 0.0,
            "low_vals": vals, "high_vals": vals,
        }
    return result


def _compute_pld_nai(pld_records, suite):
    """Build NAI dict from PLD-corrected records."""
    by_item: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in pld_records:
        if r["parsed_ok"] and r["condition"] in ("low_anchor", "high_anchor"):
            by_item[r["item_id"]][r["condition"]].append(r["answer_int"])

    result = {}
    for iid in sorted(by_item):
        c = by_item[iid]
        ev_l = float(np.mean(c["low_anchor"])) if c.get("low_anchor") else None
        ev_h = float(np.mean(c["high_anchor"])) if c.get("high_anchor") else None
        if ev_l is not None and ev_h is not None:
            dev = ev_h - ev_l
            nai = dev / (ANCHOR_HIGH - ANCHOR_LOW)
        else:
            dev = nai = None
        result[iid] = {
            "EV_control": None, "EV_low": ev_l, "EV_high": ev_h,
            "deltaEV": dev, "NAI": nai,
            "TVD": _compute_tvd(c.get("low_anchor", []), c.get("high_anchor", [])),
            "W1": _compute_w1(c.get("low_anchor", []), c.get("high_anchor", [])),
            "low_vals": c.get("low_anchor", []),
            "high_vals": c.get("high_anchor", []),
        }
    return result


if __name__ == "__main__":
    main()
