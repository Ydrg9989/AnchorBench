#!/usr/bin/env python3
"""Generate Tables 1-3 and Figures 1-2 for the anchoring mitigation paper.

Reads all mitigation result files + baseline results.
Outputs: CSV tables and matplotlib figures.

Usage:
  python mitigation/generate_results.py --out_dir results/
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _r(v, d=4):
    return round(float(v), d) if v is not None and not np.isnan(v) else None


def _bootstrap_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    means = sorted(float(arr[rng.randint(0, n, n)].mean()) for _ in range(n_boot))
    return float(means[int(n_boot * 0.025)]), float(means[int(n_boot * 0.975)])


def _bootstrap_pvalue(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 1.0
    means = [float(arr[rng.randint(0, n, n)].mean()) for _ in range(n_boot)]
    obs = float(arr.mean())
    if obs >= 0:
        return min(1.0, 2 * sum(1 for m in means if m <= 0) / n_boot)
    return min(1.0, 2 * sum(1 for m in means if m >= 0) / n_boot)


def _bh_correct(pvalues, q=0.05):
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    sig = [False] * n
    for rank, (idx, pv) in enumerate(indexed, 1):
        if pv <= q * rank / n:
            sig[idx] = True
        else:
            break
    return sig


def _load_results(paths):
    records = []
    for p in paths:
        if not p.exists():
            log.warning("File not found: %s", p)
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _compute_nai_per_item(records, suite):
    by_item = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["suite"] != suite or not r["parsed_ok"]:
            continue
        if r["condition"] in ("control", "low_anchor", "high_anchor"):
            by_item[r["item_id"]][r["condition"]].append(r["answer_int"])

    result = {}
    for iid in sorted(by_item):
        c = by_item[iid]
        ev_l = float(np.mean(c["low_anchor"])) if c.get("low_anchor") else None
        ev_h = float(np.mean(c["high_anchor"])) if c.get("high_anchor") else None
        ev_c = float(np.mean(c["control"])) if c.get("control") else None
        if ev_l is not None and ev_h is not None:
            nai = (ev_h - ev_l) / (ANCHOR_HIGH - ANCHOR_LOW)
        else:
            nai = None
        result[iid] = {"NAI": nai, "EV_control": ev_c, "EV_low": ev_l, "EV_high": ev_h}
    return result


def _compute_posthoc_b4(records):
    by_key = defaultdict(list)
    meta = {}
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
        r = meta[key].copy()
        r["answer_int"] = int(round(float(np.median(vals))))
        r["parsed_ok"] = True
        r["raw_text"] = f"SC_median"
        r["sample_idx"] = 0
        r["mitigation"] = "B4"
        synthetic.append(r)
    return synthetic


def _compute_posthoc_cac(records):
    by_low = defaultdict(list)
    by_high = defaultdict(list)
    meta = {}
    for r in records:
        if not r["parsed_ok"]:
            continue
        key = (r["model_id"], r["item_id"], r["sample_idx"])
        if r["condition"] == "low_anchor":
            by_low[key].append(r["answer_int"])
            meta[key] = r
        elif r["condition"] == "high_anchor":
            by_high[key].append(r["answer_int"])

    synthetic = []
    for key in by_low:
        lows = by_low[key]
        highs = by_high.get(key, [])
        if not lows or not highs:
            continue
        r = meta[key].copy()
        r["answer_int"] = int(round((np.mean(lows) + np.mean(highs)) / 2))
        r["parsed_ok"] = True
        r["condition"] = "CAC"
        r["raw_text"] = "CAC_mid"
        r["mitigation"] = "CAC"
        synthetic.append(r)
    return synthetic


def _compute_posthoc_pld(records):
    synthetic = []
    for r in records:
        if not r["parsed_ok"] or r["answer_int"] is None:
            continue
        if r["condition"] == "low_anchor":
            av = ANCHOR_LOW
        elif r["condition"] == "high_anchor":
            av = ANCHOR_HIGH
        else:
            continue
        corrected = pld_correct(r["answer_int"], av, r["model_id"])
        rec = r.copy()
        rec["answer_int"] = corrected
        rec["raw_text"] = f"PLD"
        rec["mitigation"] = "PLD"
        synthetic.append(rec)
    return synthetic


# ── TABLE 1: NAI by mitigation × suite × model ──────────────────────

def generate_table1(all_data, out_dir):
    """NAI heatmap table."""
    rows = []
    for (model, mitigation), records in sorted(all_data.items()):
        row = {"model": model.split("/")[-1], "mitigation": mitigation}
        for suite in sorted(EXTERNAL_LIKE):
            nai_data = _compute_nai_per_item(records, suite)
            nais = [v["NAI"] for v in nai_data.values() if v["NAI"] is not None]
            row[f"{suite}_NAI"] = _r(np.mean(nais)) if nais else None
            row[f"{suite}_n"] = len(nais)
        rows.append(row)

    path = out_dir / "table1_nai.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    log.info("Table 1 → %s (%d rows)", path, len(rows))
    return rows


# ── TABLE 2: Relative Reduction with CI ──────────────────────────────

def generate_table2(all_data, out_dir):
    """RR with 95% CI and BH significance."""
    baseline_key_fn = lambda model: (model, "B0")
    rows = []
    all_pvals = []

    for (model, mitigation), records in sorted(all_data.items()):
        if mitigation == "B0":
            continue
        b0_records = all_data.get(baseline_key_fn(model), [])
        if not b0_records:
            continue

        for suite in sorted(EXTERNAL_LIKE):
            b_nai = _compute_nai_per_item(b0_records, suite)
            m_nai = _compute_nai_per_item(records, suite)
            common = set(b_nai) & set(m_nai)
            rrs = []
            for iid in common:
                bn = b_nai[iid]["NAI"]
                mn = m_nai[iid]["NAI"]
                if bn is not None and mn is not None and abs(bn) > 1e-6:
                    rrs.append(1 - mn / bn)
            if not rrs:
                continue

            ci = _bootstrap_ci(rrs)
            pv = _bootstrap_pvalue(rrs)
            all_pvals.append(pv)

            rows.append({
                "model": model.split("/")[-1],
                "mitigation": mitigation,
                "suite": suite,
                "mean_RR": _r(np.mean(rrs)),
                "ci_lo": _r(ci[0]),
                "ci_hi": _r(ci[1]),
                "p_value": _r(pv, 6),
                "n_items": len(rrs),
            })

    if all_pvals:
        sig_flags = _bh_correct(all_pvals)
        for row, sig in zip(rows, sig_flags):
            row["sig_BH"] = sig

    path = out_dir / "table2_rr.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    log.info("Table 2 → %s (%d rows)", path, len(rows))
    return rows


# ── TABLE 3: Stress test results ────────────────────────────────────

def generate_table3(stress_data, out_dir):
    """Stress test NAI by test type and model."""
    rows = []
    for (model, mitigation), records in sorted(stress_data.items()):
        for test_prefix in ("T1", "T2", "T3", "T4"):
            test_recs = [r for r in records if r["item_id"].startswith(test_prefix)]
            if not test_recs:
                continue
            suite = test_recs[0]["suite"]
            nai_data = _compute_nai_per_item(test_recs, suite)
            nais = [v["NAI"] for v in nai_data.values() if v["NAI"] is not None]
            rows.append({
                "model": model.split("/")[-1],
                "mitigation": mitigation,
                "stress_test": test_prefix,
                "mean_NAI": _r(np.mean(nais)) if nais else None,
                "n_items": len(nais),
            })

    path = out_dir / "table3_stress.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    log.info("Table 3 → %s (%d rows)", path, len(rows))
    return rows


# ── FIGURE 1: NAI vs Accuracy scatter ────────────────────────────────

def generate_figure1(all_data, gold_lookup, out_dir):
    """NAI vs Accuracy Pareto frontier."""
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"B0": "o", "B1": "s", "B2": "D", "B3": "^", "B4": "v",
               "B5": "p", "CAC": "*", "EFR": "h", "CxDP": "P", "PLD": "X"}
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, ((model, mitigation), records) in enumerate(sorted(all_data.items())):
        for suite in ["external"]:
            nai_data = _compute_nai_per_item(records, suite)
            nais = [v["NAI"] for v in nai_data.values() if v["NAI"] is not None]
            if not nais:
                continue
            mean_nai = abs(np.mean(nais))

            accs = []
            for iid, d in nai_data.items():
                if d["EV_control"] is None:
                    continue
                item = gold_lookup.get(iid)
                if item and item.get("gold", {}).get("y_star") is not None:
                    accs.append(abs(d["EV_control"] - item["gold"]["y_star"]))
            mean_err = np.mean(accs) if accs else None

            if mean_err is not None:
                marker = markers.get(mitigation, "o")
                ax.scatter(mean_nai, mean_err, marker=marker, s=80,
                           label=f"{model.split('/')[-1]}/{mitigation}")

    ax.set_xlabel("|NAI| (anchoring magnitude)")
    ax.set_ylabel("Mean |Error| vs Gold")
    ax.set_title("Figure 1: NAI vs Accuracy (External Suite)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    path = out_dir / "figure1_nai_vs_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Figure 1 → %s", path)


# ── FIGURE 2: Controlled-history dose-response ──────────────────────

def generate_figure2(baseline_records, out_dir):
    """EV vs anchor value under B0 for controlled-history."""
    import re
    anchors_list = [10, 30, 50, 70, 90]

    by_model = defaultdict(lambda: defaultdict(list))
    for r in baseline_records:
        if not r["parsed_ok"]:
            continue
        m = re.match(r"turn2_controlled_history_(\d+)", r.get("condition", ""))
        if m:
            a = int(m.group(1))
            by_model[r["model_id"]][a].append(r["answer_int"])

    if not by_model:
        log.info("Figure 2: No controlled-history data found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (model_id, anchor_data) in enumerate(sorted(by_model.items())):
        xs, ys = [], []
        for a in anchors_list:
            vals = anchor_data.get(a, [])
            if vals:
                xs.append(a)
                ys.append(np.mean(vals))
        if xs:
            ax.plot(xs, ys, "o-", label=model_id.split("/")[-1])
            coeffs = np.polyfit(xs, ys, 1)
            ax.plot([0, 100], [coeffs[1], coeffs[0] * 100 + coeffs[1]],
                    "--", alpha=0.3)

    ax.plot([0, 100], [0, 100], "k:", alpha=0.2, label="y=x")
    ax.set_xlabel("Anchor value")
    ax.set_ylabel("Mean EV")
    ax.set_title("Figure 2: Controlled-History Dose-Response (B0)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "figure2_dose_response.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Figure 2 → %s", path)


# ── main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path,
                   default=Path("poc_dataset/poc_v0.2.jsonl"))
    p.add_argument("--baseline_dir", type=Path,
                   default=Path("runner_outputs"))
    p.add_argument("--mitigation_dir", type=Path,
                   default=Path("runner_outputs"))
    p.add_argument("--stress_dataset", type=Path,
                   default=Path("poc_dataset/stress_test.jsonl"))
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = load_dataset(args.dataset)
    gold_lookup = {it["item_id"]: it for it in items}

    # Load baseline (B0)
    baseline_files = list(args.baseline_dir.glob("poc_v0_*.jsonl")) + \
                     list(args.baseline_dir.glob("api_v02.jsonl"))
    baseline_records = _load_results(baseline_files)
    log.info("Baseline (B0): %d records from %d files",
             len(baseline_records), len(baseline_files))

    # Group all data by (model, mitigation)
    all_data: dict[tuple[str, str], list[dict]] = {}

    # B0 from baseline
    by_model_b0 = defaultdict(list)
    for r in baseline_records:
        r_copy = r.copy()
        r_copy.setdefault("mitigation", "B0")
        by_model_b0[r["model_id"]].append(r_copy)
    for model_id, recs in by_model_b0.items():
        all_data[(model_id, "B0")] = recs

    # Post-hoc B4, CAC, PLD from baseline
    b4_recs = _compute_posthoc_b4(baseline_records)
    for r in b4_recs:
        key = (r["model_id"], "B4")
        all_data.setdefault(key, []).append(r)

    cac_recs = _compute_posthoc_cac(baseline_records)
    for r in cac_recs:
        key = (r["model_id"], "CAC")
        all_data.setdefault(key, []).append(r)

    pld_recs = _compute_posthoc_pld(baseline_records)
    for r in pld_recs:
        key = (r["model_id"], "PLD")
        all_data.setdefault(key, []).append(r)

    # Load mitigation result files
    mit_files = list(args.mitigation_dir.glob("mit_*.jsonl")) + \
                list(args.mitigation_dir.glob("hf_*b*.jsonl")) + \
                list(args.mitigation_dir.glob("hf_*efr*.jsonl")) + \
                list(args.mitigation_dir.glob("hf_*cxdp*.jsonl"))
    mit_records = _load_results(mit_files)
    log.info("Mitigation: %d records from %d files",
             len(mit_records), len(mit_files))

    for r in mit_records:
        mit = r.get("mitigation", "unknown")
        key = (r["model_id"], mit)
        all_data.setdefault(key, []).append(r)

    log.info("Total (model, mitigation) groups: %d", len(all_data))
    for k, v in sorted(all_data.items()):
        log.info("  %s/%s: %d records", k[0].split("/")[-1], k[1], len(v))

    # Generate tables
    t1 = generate_table1(all_data, args.out_dir)
    t2 = generate_table2(all_data, args.out_dir)

    # Stress tests
    stress_data: dict[tuple[str, str], list[dict]] = {}
    stress_files = list(args.mitigation_dir.glob("*stress*.jsonl"))
    stress_records = _load_results(stress_files)
    if stress_records:
        for r in stress_records:
            mit = r.get("mitigation", "B0")
            key = (r["model_id"], mit)
            stress_data.setdefault(key, []).append(r)
        t3 = generate_table3(stress_data, args.out_dir)
    else:
        log.info("No stress-test results found, skipping Table 3.")

    # Generate figures
    generate_figure1(all_data, gold_lookup, args.out_dir)
    generate_figure2(baseline_records, args.out_dir)

    # Summary JSON
    summary = {
        "n_model_mitigation_groups": len(all_data),
        "groups": {f"{k[0].split('/')[-1]}/{k[1]}": len(v)
                   for k, v in sorted(all_data.items())},
        "table1_rows": len(t1),
        "table2_rows": len(t2),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Summary → %s", args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
