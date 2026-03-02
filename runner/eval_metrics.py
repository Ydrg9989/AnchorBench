#!/usr/bin/env python3
"""Compute anchoring metrics from runner result JSONL files.

Handles mixed backends, all suites, and outputs JSON + CSV.

Usage:
  python runner/eval_metrics.py \\
      --dataset poc_dataset/poc_v0.2.jsonl \\
      --inputs runner_outputs/api_v02.jsonl runner_outputs/hf_v02.jsonl \\
      --run_name v02
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXTERNAL_LIKE = {"external", "icl", "conversation_history", "rag", "tool"}
CTRL_ANCHORS = [10, 30, 50, 70, 90]
N_BOOT = 2000
BOOT_SEED = 42


# ── statistics helpers ────────────────────────────────────────────────

def _r(v, decimals: int = 4):
    if v is None:
        return None
    return round(float(v), decimals)


def _bootstrap_ci(
    values: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    means = np.sort(
        [float(arr[rng.randint(0, n, n)].mean()) for _ in range(n_boot)]
    )
    return float(means[int(n_boot * 0.025)]), float(means[int(n_boot * 0.975)])


def _ols_slope(x, y):
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xm, ym = xa.mean(), ya.mean()
    denom = float(((xa - xm) ** 2).sum())
    if denom == 0:
        return float(ym), 0.0
    beta = float(((xa - xm) * (ya - ym)).sum() / denom)
    alpha = float(ym - beta * xm)
    return alpha, beta


def _compute_tvd(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    bins = np.arange(102)
    pmf_a = np.histogram(a, bins=bins)[0].astype(np.float64) / len(a)
    pmf_b = np.histogram(b, bins=bins)[0].astype(np.float64) / len(b)
    return 0.5 * float(np.abs(pmf_a - pmf_b).sum())


def _compute_w1(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    bins = np.arange(102)
    cdf_a = np.cumsum(np.histogram(a, bins=bins)[0].astype(np.float64)) / len(a)
    cdf_b = np.cumsum(np.histogram(b, bins=bins)[0].astype(np.float64)) / len(b)
    return float(np.abs(cdf_a - cdf_b).sum())


# ── data loading ──────────────────────────────────────────────────────

def _load_results(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    log.info("Loaded %d result records from %d file(s)", len(records), len(paths))
    return records


def _build_gold(items: list[dict]) -> dict[str, dict]:
    return {it["item_id"]: it["gold"] for it in items}


def _split_by_model_backend(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["model_id"], r.get("backend", "unknown"))].append(r)
    return dict(sorted(groups.items()))


# ── parse rate table ──────────────────────────────────────────────────

def _parse_rate_table(
    records: list[dict], model_id: str, backend: str,
) -> dict[str, dict[str, float]]:
    """Returns {suite: {condition: rate}}."""
    counts: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in records:
        counts[(r["suite"], r["condition"])].append(r["parsed_ok"])

    table: dict[str, dict[str, float]] = defaultdict(dict)
    for (suite, cond), oks in sorted(counts.items()):
        table[suite][cond] = sum(oks) / len(oks) if oks else 0.0
    return dict(table)


# ── external-like metrics ─────────────────────────────────────────────

def _eval_external_like(
    records: list[dict], suite_name: str, gold: dict[str, dict],
) -> dict:
    by_item: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["suite"] != suite_name or not r["parsed_ok"]:
            continue
        if r["condition"] in ("control", "low_anchor", "high_anchor"):
            by_item[r["item_id"]][r["condition"]].append(r["answer_int"])

    delta_evs, nais, tvds, w1s = [], [], [], []
    per_item = []

    for iid in sorted(by_item):
        c = by_item[iid]
        ev_ctrl = float(np.mean(c["control"])) if c.get("control") else None
        ev_low = float(np.mean(c["low_anchor"])) if c.get("low_anchor") else None
        ev_high = float(np.mean(c["high_anchor"])) if c.get("high_anchor") else None

        row: dict = {
            "item_id": iid,
            "EV_control": _r(ev_ctrl),
            "EV_low": _r(ev_low),
            "EV_high": _r(ev_high),
        }

        if ev_low is not None and ev_high is not None:
            dev = float(ev_high - ev_low)
            nai = dev / 60.0
            tvd = _compute_tvd(c["low_anchor"], c["high_anchor"])
            w1 = _compute_w1(c["low_anchor"], c["high_anchor"])
            delta_evs.append(dev)
            nais.append(nai)
            tvds.append(tvd)
            w1s.append(w1)
            row.update({"deltaEV": _r(dev), "NAI": _r(nai),
                        "TVD": _r(tvd), "W1": _r(w1)})
        per_item.append(row)

    def _agg(vals):
        if not vals:
            return {"mean": None, "ci95": None}
        ci = _bootstrap_ci(vals)
        return {"mean": _r(np.mean(vals)), "ci95": [_r(ci[0]), _r(ci[1])]}

    return {
        "n_items": len(per_item),
        "deltaEV": _agg(delta_evs),
        "NAI": _agg(nais),
        "TVD": _agg(tvds),
        "W1": _agg(w1s),
        "per_item": per_item,
    }


# ── self-generated metrics ────────────────────────────────────────────

def _eval_self(
    records: list[dict], gold: dict[str, dict],
) -> dict:
    by_item_cond: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    by_sample: dict[tuple[str, int], dict[str, int | None]] = {}

    for r in records:
        if r["suite"] != "self_generated" or not r["parsed_ok"]:
            continue
        if r["condition"] in ("turn1", "turn2_fresh", "turn2_history"):
            by_item_cond[r["item_id"]][r["condition"]].append(r["answer_int"])
            key = (r["item_id"], r["sample_idx"])
            by_sample.setdefault(key, {})[r["condition"]] = r["answer_int"]

    delta_errs, pulls, inertias = [], [], []
    per_item = []

    for iid in sorted({k[0] for k in by_sample}):
        g = gold.get(iid, {})
        y2 = g.get("y_star_stage2")

        fresh_vals = by_item_cond[iid].get("turn2_fresh", [])
        hist_vals = by_item_cond[iid].get("turn2_history", [])

        ev_fresh = float(np.mean(fresh_vals)) if fresh_vals else None
        ev_hist = float(np.mean(hist_vals)) if hist_vals else None

        row: dict = {"item_id": iid, "y_star_stage2": y2,
                     "EV_fresh": _r(ev_fresh), "EV_hist": _r(ev_hist)}

        if y2 is not None and ev_fresh is not None and ev_hist is not None:
            de = abs(ev_hist - y2) - abs(ev_fresh - y2)
            delta_errs.append(de)
            row["delta_err"] = _r(de)

        per_item.append(row)

    # per-sample Pull and Inertia
    for (iid, si), vals in sorted(by_sample.items()):
        a1 = vals.get("turn1")
        f = vals.get("turn2_fresh")
        h = vals.get("turn2_history")
        if a1 is not None and f is not None and h is not None:
            pull = (h - f) * (a1 - f)
            inertia = abs(h - a1) - abs(f - a1)
            pulls.append(pull)
            inertias.append(inertia)

    de_ci = _bootstrap_ci(delta_errs) if delta_errs else (None, None)

    pull_rate = (sum(1 for p in pulls if p > 0) / len(pulls)) if pulls else None
    inertia_rate = (sum(1 for i in inertias if i < 0) / len(inertias)) if inertias else None

    return {
        "n_items": len(per_item),
        "delta_err": {
            "mean": _r(np.mean(delta_errs)) if delta_errs else None,
            "ci95": [_r(de_ci[0]), _r(de_ci[1])] if de_ci[0] is not None else None,
        },
        "pull": {
            "n_samples": len(pulls),
            "rate": _r(pull_rate),
            "mean": _r(np.mean(pulls)) if pulls else None,
        },
        "inertia": {
            "n_samples": len(inertias),
            "rate": _r(inertia_rate),
            "mean": _r(np.mean(inertias)) if inertias else None,
        },
        "per_item": per_item,
    }


# ── controlled-history ────────────────────────────────────────────────

def _eval_controlled(records: list[dict]) -> dict:
    by_item_anchor: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["suite"] != "self_generated" or not r["parsed_ok"]:
            continue
        for a in CTRL_ANCHORS:
            if r["condition"] == f"turn2_controlled_history_{a}":
                by_item_anchor[r["item_id"]][a].append(r["answer_int"])

    x_all, y_all = [], []
    per_item = []
    for iid in sorted(by_item_anchor):
        row: dict = {"item_id": iid}
        for a in CTRL_ANCHORS:
            vals = by_item_anchor[iid].get(a, [])
            if vals:
                m = float(np.mean(vals))
                row[f"EV_a{a}"] = _r(m, 2)
                x_all.append(float(a))
                y_all.append(m)
        per_item.append(row)

    alpha, beta = _ols_slope(x_all, y_all) if x_all else (None, 0.0)
    beta_norm = beta * 60.0 / 80.0 if beta else 0.0

    return {
        "n_items": len(per_item),
        "alpha": _r(alpha),
        "beta": _r(beta, 6),
        "beta_norm": _r(beta_norm, 6),
        "per_item": per_item,
    }


# ── CSV output ────────────────────────────────────────────────────────

def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    all_keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compute anchoring metrics")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("runner_outputs"))
    args = p.parse_args()

    items = load_dataset(args.dataset)
    gold = _build_gold(items)
    records = _load_results(args.inputs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name
    if not run_name:
        run_name = args.inputs[0].stem.split("_")[0]

    by_mb = _split_by_model_backend(records)
    full_summary: dict[str, dict] = {}
    csv_rows: list[dict] = []

    for (model_id, backend), recs in by_mb.items():
        model_short = model_id.split("/")[-1]
        total = len(recs)
        ok = sum(1 for r in recs if r["parsed_ok"])
        parse_rate = ok / total if total else 0
        log.info("=== %s [%s]  %d records  parse_rate=%.1f%% ===",
                 model_short, backend, total, parse_rate * 100)

        pr_table = _parse_rate_table(recs, model_id, backend)
        for suite, conds in sorted(pr_table.items()):
            for cond, rate in sorted(conds.items()):
                log.info("  parse %s/%-30s %.1f%%", suite, cond, rate * 100)

        model_key = f"{model_id}|{backend}"
        summary: dict[str, Any] = {
            "model_id": model_id,
            "backend": backend,
            "n_records": total,
            "parse_rate": _r(parse_rate),
            "parse_rates": pr_table,
        }

        csv_row: dict[str, Any] = {
            "model_id": model_id,
            "backend": backend,
            "n_records": total,
            "parse_rate_total": _r(parse_rate),
        }

        for suite_name in sorted(EXTERNAL_LIKE):
            suite_recs = [r for r in recs if r["suite"] == suite_name]
            if not suite_recs:
                continue
            ext = _eval_external_like(suite_recs, suite_name, gold)
            summary[suite_name] = {k: v for k, v in ext.items() if k != "per_item"}

            prefix = {
                "external": "ext", "icl": "icl",
                "conversation_history": "hist",
                "rag": "rag", "tool": "tool",
            }.get(suite_name, suite_name[:4])

            csv_row[f"{prefix}_n"] = ext["n_items"]
            csv_row[f"{prefix}_deltaEV_mean"] = ext["deltaEV"]["mean"]
            csv_row[f"{prefix}_NAI_mean"] = ext["NAI"]["mean"]
            csv_row[f"{prefix}_TVD_mean"] = ext["TVD"]["mean"]
            csv_row[f"{prefix}_W1_mean"] = ext["W1"]["mean"]
            if ext["deltaEV"]["ci95"]:
                csv_row[f"{prefix}_deltaEV_ci_lo"] = ext["deltaEV"]["ci95"][0]
                csv_row[f"{prefix}_deltaEV_ci_hi"] = ext["deltaEV"]["ci95"][1]

            log.info("  %s: ΔEV=%.2f  NAI=%.3f  TVD=%.3f  W1=%.2f  CI=%s  (n=%d)",
                     suite_name,
                     ext["deltaEV"]["mean"] or 0,
                     ext["NAI"]["mean"] or 0,
                     ext["TVD"]["mean"] or 0,
                     ext["W1"]["mean"] or 0,
                     ext["deltaEV"]["ci95"] or "N/A",
                     ext["n_items"])

        self_recs = [r for r in recs if r["suite"] == "self_generated"]
        if self_recs:
            self_m = _eval_self(self_recs, gold)
            ctrl_m = _eval_controlled(self_recs)

            summary["self_generated"] = {
                k: v for k, v in self_m.items() if k != "per_item"
            }
            summary["controlled_history"] = ctrl_m

            csv_row["self_delta_err_mean"] = self_m["delta_err"]["mean"]
            csv_row["self_pullrate"] = self_m["pull"]["rate"]
            csv_row["self_meanpull"] = self_m["pull"]["mean"]
            csv_row["self_inertiarate"] = self_m["inertia"]["rate"]
            csv_row["self_meaninertia"] = self_m["inertia"]["mean"]
            csv_row["self_betanorm"] = ctrl_m["beta_norm"]
            csv_row["self_beta"] = ctrl_m["beta"]

            log.info("  self_gen: delta_err=%.2f  pullrate=%.2f  betanorm=%.4f  (n=%d)",
                     self_m["delta_err"]["mean"] or 0,
                     self_m["pull"]["rate"] or 0,
                     ctrl_m["beta_norm"] or 0,
                     self_m["n_items"])

        full_summary[model_key] = summary
        csv_rows.append(csv_row)

    summary_path = args.out_dir / f"{run_name}_metrics.json"
    summary_path.write_text(
        json.dumps(full_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("JSON → %s", summary_path)

    csv_path = args.out_dir / f"{run_name}_metrics.csv"
    _write_csv(csv_rows, csv_path)
    log.info("CSV  → %s", csv_path)


if __name__ == "__main__":
    main()
