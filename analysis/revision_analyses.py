#!/usr/bin/env python3
"""Targeted analyses for the COLM paper revision.

A1: Anchored-condition MAE (do anchors help or hurt prediction quality?)
A2: UAI exclusion rates + epsilon sensitivity
A3: Anchor-distance (offset) dose-response analysis
A4: History suite matched-control (two-stage baseline) analysis
A5: ICL distribution-matching comparison

All analyses use existing results.jsonl files — no inference required.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from anchorbench_eval.metrics import (
    EPSILON,
    compute_unified_metrics,
    group_by_item,
    bootstrap_ci,
)

OUT_DIR = ROOT / "outputs" / "revision_analyses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_DIR = ROOT / "results" / "full_benchmark"
API_DIR = ROOT / "results" / "api_benchmark"
HIST_TS_DIR = ROOT / "results" / "history_core_twostage_baseline"
ICL_DIST_DIR = ROOT / "results" / "icl_dist_core"

MODEL_SHORT = {
    "Qwen_Qwen2.5-1.5B-Instruct": "Qwen-1.5B",
    "Qwen_Qwen2.5-3B-Instruct": "Qwen-3B",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen-7B",
    "meta-llama_Llama-3.2-1B-Instruct": "Llama-1B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-8B",
    "google_gemma-3-1b-it": "Gemma-1B",
    "google_gemma-3-4b-it": "Gemma-4B",
    "allenai_OLMo-2-1124-13B-Instruct": "OLMo-13B",
    "allenai_OLMo-2-0325-32B-Instruct": "OLMo-32B",
    "openai_gpt-5.4-mini": "GPT-5.4-mini",
    "anthropic_claude-haiku-4.5": "Claude-H4.5",
    "google_gemini-2.5-flash": "Gemini-2.5-Flash",
    "x-ai_grok-3-mini-beta": "Grok-3-mini",
}

SUITES = ["external", "history", "icl", "rag", "tool"]


def load_records(path: Path) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def iter_model_suite(base_dir: Path, suites: list[str] | None = None):
    """Yield (suite, model_slug, records_path) for each model-suite run."""
    target_suites = suites or SUITES
    for suite in target_suites:
        suite_dir = base_dir / suite
        if not suite_dir.is_dir():
            continue
        for model_dir in sorted(suite_dir.iterdir()):
            rpath = model_dir / "results.jsonl"
            if rpath.exists():
                yield suite, model_dir.name, rpath


def iter_model_flat(base_dir: Path):
    """Yield (model_slug, records_path) for flat model dirs (no suite subdir)."""
    for model_dir in sorted(base_dir.iterdir()):
        rpath = model_dir / "results.jsonl"
        if rpath.exists():
            yield model_dir.name, rpath


def parse_offset_from_id(item_id: str) -> int | None:
    m = re.search(r"-off(\d+)-", item_id)
    return int(m.group(1)) if m else None


# ───────────────────────────────────────────────────────────────
# A1: Anchored-condition MAE
# ───────────────────────────────────────────────────────────────
def analysis_a1():
    """Compute MAE for each condition (not just control)."""
    print("\n=== A1: Anchored-condition MAE ===")
    rows = []

    for base_dir in [FULL_DIR, API_DIR]:
        for suite, slug, rpath in iter_model_suite(base_dir):
            records = load_records(rpath)
            short = MODEL_SHORT.get(slug, slug)

            mae_by_cond = defaultdict(list)
            for r in records:
                if not r.get("parsed_ok") or r.get("answer_int") is None:
                    continue
                y = r["answer_int"]
                y_star = r.get("y_star_evidence")
                if y_star is None:
                    continue
                mae_by_cond[r["condition"]].append(abs(y - y_star))

            row = {"model": short, "suite": suite}
            for cond in ["control", "irrelevant_low", "irrelevant_high",
                         "plausible_low", "plausible_high"]:
                vals = mae_by_cond.get(cond, [])
                row[f"mae_{cond}"] = round(float(np.mean(vals)), 2) if vals else None
                row[f"n_{cond}"] = len(vals)

            if row.get("mae_control") is not None:
                irr_vals = mae_by_cond.get("irrelevant_low", []) + mae_by_cond.get("irrelevant_high", [])
                pls_vals = mae_by_cond.get("plausible_low", []) + mae_by_cond.get("plausible_high", [])
                row["mae_irr"] = round(float(np.mean(irr_vals)), 2) if irr_vals else None
                row["mae_pls"] = round(float(np.mean(pls_vals)), 2) if pls_vals else None
                row["delta_mae_irr"] = round(row["mae_irr"] - row["mae_control"], 2) if row["mae_irr"] else None
                row["delta_mae_pls"] = round(row["mae_pls"] - row["mae_control"], 2) if row["mae_pls"] else None
            rows.append(row)

    out = OUT_DIR / "a1_anchored_mae.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} rows to {out}")

    print("\n  Summary (mean across models):")
    for suite in SUITES:
        suite_rows = [r for r in rows if r["suite"] == suite and r.get("delta_mae_irr") is not None]
        if not suite_rows:
            continue
        d_irr = np.mean([r["delta_mae_irr"] for r in suite_rows])
        d_pls = np.mean([r["delta_mae_pls"] for r in suite_rows])
        print(f"    {suite:10s}: delta_MAE_irr={d_irr:+.2f}, delta_MAE_pls={d_pls:+.2f}")

    return rows


# ───────────────────────────────────────────────────────────────
# A2: UAI exclusion rate + epsilon sensitivity
# ───────────────────────────────────────────────────────────────
def analysis_a2():
    """Report exclusion rates and test sensitivity to epsilon."""
    print("\n=== A2: UAI exclusion rate + epsilon sensitivity ===")
    rows = []

    for base_dir in [FULL_DIR, API_DIR]:
        for suite, slug, rpath in iter_model_suite(base_dir):
            records = load_records(rpath)
            short = MODEL_SHORT.get(slug, slug)
            row = {"model": short, "suite": suite}

            for eps in [1.0, 3.0, 5.0]:
                m = compute_unified_metrics(records, epsilon=eps)
                n_irr = m.get("n_uai_irr", 0)
                n_pls = m.get("n_uai_plaus", 0)

                items = group_by_item(records)
                n_total_anch = 0
                n_excluded = 0
                for iid, conds in items.items():
                    ctrl = conds.get("control")
                    if ctrl is None:
                        continue
                    y_ctrl = ctrl.get("answer_int")
                    if y_ctrl is None:
                        continue
                    for c in ["irrelevant_low", "irrelevant_high",
                              "plausible_low", "plausible_high"]:
                        rec = conds.get(c)
                        if rec is None:
                            continue
                        a = rec.get("anchor_value")
                        if a is None:
                            continue
                        n_total_anch += 1
                        if abs(a - y_ctrl) < eps:
                            n_excluded += 1

                excl_rate = n_excluded / n_total_anch if n_total_anch else 0
                eps_key = str(eps).replace(".", "")
                row[f"eps{eps_key}_uai_irr"] = m.get("uai_irr")
                row[f"eps{eps_key}_uai_pls"] = m.get("uai_plaus") or m.get("uai_plaus")
                row[f"eps{eps_key}_disc"] = m.get("disc_delta")
                row[f"eps{eps_key}_excl_rate"] = round(excl_rate, 4)
                row[f"eps{eps_key}_n_excl"] = n_excluded
                row[f"eps{eps_key}_n_total"] = n_total_anch

            rows.append(row)

    out = OUT_DIR / "a2_epsilon_sensitivity.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} rows to {out}")

    print("\n  Summary exclusion rates (eps=3, mean across models):")
    for suite in SUITES:
        suite_rows = [r for r in rows if r["suite"] == suite]
        if not suite_rows:
            continue
        rates = [r.get("eps30_excl_rate", 0) for r in suite_rows]
        print(f"    {suite:10s}: mean excl_rate={np.mean(rates):.1%}")

    print("\n  Disc_delta sign stability across epsilon:")
    sign_flips = 0
    total = 0
    for r in rows:
        d1 = r.get("eps10_disc")
        d3 = r.get("eps30_disc")
        d5 = r.get("eps50_disc")
        if d1 is not None and d3 is not None and d5 is not None:
            total += 1
            signs = [np.sign(d1), np.sign(d3), np.sign(d5)]
            if len(set(s for s in signs if s != 0)) > 1:
                sign_flips += 1
    print(f"    {sign_flips}/{total} model-suite cells have Disc_delta sign flip across eps")

    return rows


# ───────────────────────────────────────────────────────────────
# A3: Anchor-distance (offset) dose-response
# ───────────────────────────────────────────────────────────────
def analysis_a3():
    """Compute UAI grouped by anchor offset {15, 25, 40}."""
    print("\n=== A3: Anchor-distance dose-response ===")
    rows = []

    for base_dir in [FULL_DIR, API_DIR]:
        for suite, slug, rpath in iter_model_suite(base_dir):
            records = load_records(rpath)
            short = MODEL_SHORT.get(slug, slug)

            by_offset = defaultdict(list)
            for r in records:
                offset = parse_offset_from_id(r.get("item_id", ""))
                if offset is not None:
                    by_offset[offset].append(r)

            for offset in sorted(by_offset.keys()):
                m = compute_unified_metrics(by_offset[offset], epsilon=EPSILON)
                rows.append({
                    "model": short,
                    "suite": suite,
                    "offset": offset,
                    "uai_irr": m.get("uai_irr"),
                    "uai_plaus": m.get("uai_plaus"),
                    "disc_delta": m.get("disc_delta"),
                    "mae_control": m.get("mae_control"),
                    "acc10_control": m.get("acc10_control"),
                    "n_uai_irr": m.get("n_uai_irr", 0),
                    "n_uai_plaus": m.get("n_uai_plaus", 0),
                    "n_items": m.get("n_items", 0),
                })

    out = OUT_DIR / "a3_offset_dose_response.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} rows to {out}")

    print("\n  Suite-level dose-response (mean UAI_plaus across models):")
    for suite in SUITES:
        suite_rows = [r for r in rows if r["suite"] == suite]
        if not suite_rows:
            continue
        for offset in [15, 25, 40]:
            off_rows = [r for r in suite_rows if r["offset"] == offset and r.get("uai_plaus") is not None]
            if off_rows:
                mean_uai = np.mean([r["uai_plaus"] for r in off_rows])
                print(f"    {suite:10s} delta={offset:2d}: mean UAI_plaus={mean_uai:.3f} (n={len(off_rows)})")

    # Check boundary effects
    print("\n  Boundary check (anchors clamped near 0 or 100):")
    boundary_items = 0
    total_items = 0
    for base_dir in [FULL_DIR, API_DIR]:
        for suite, slug, rpath in iter_model_suite(base_dir, ["external"]):
            records = load_records(rpath)
            for r in records:
                if r.get("condition") == "control":
                    continue
                a = r.get("anchor_value")
                if a is not None:
                    total_items += 1
                    if a <= 5 or a >= 95:
                        boundary_items += 1
    print(f"    {boundary_items}/{total_items} anchored items have anchor in [0,5] or [95,100]")

    return rows


# ───────────────────────────────────────────────────────────────
# A4: History matched-control (two-stage baseline)
# ───────────────────────────────────────────────────────────────
def analysis_a4():
    """Re-analyze History with matched two-stage control baseline."""
    print("\n=== A4: History matched-control analysis ===")
    rows = []

    if not HIST_TS_DIR.is_dir():
        print("  SKIP: history_core_twostage_baseline not found")
        return rows

    for slug, rpath in iter_model_flat(HIST_TS_DIR):
        short = MODEL_SHORT.get(slug, slug)
        ts_records = load_records(rpath)

        # Also load standard History results for comparison
        std_path = FULL_DIR / "history" / slug / "results.jsonl"
        if not std_path.exists():
            continue
        std_records = load_records(std_path)

        # Standard metrics (single-stage control)
        m_std = compute_unified_metrics(std_records, epsilon=EPSILON, baseline_condition="control")

        # Merge: take anchored conditions from std, control_twostage from ts
        merged = []
        for r in ts_records:
            if r["condition"] == "control_twostage":
                ctrl_rec = dict(r)
                ctrl_rec["condition"] = "control"
                merged.append(ctrl_rec)
        for r in std_records:
            if r["condition"] != "control":
                merged.append(r)

        m_matched = compute_unified_metrics(merged, epsilon=EPSILON, baseline_condition="control")

        # MAE for two-stage control
        ts_mae_vals = []
        for r in ts_records:
            if r["condition"] == "control_twostage" and r.get("parsed_ok") and r.get("answer_int") is not None:
                y_star = r.get("y_star_evidence")
                if y_star is not None:
                    ts_mae_vals.append(abs(r["answer_int"] - y_star))
        mae_ts_ctrl = round(float(np.mean(ts_mae_vals)), 2) if ts_mae_vals else None

        rows.append({
            "model": short,
            "suite": "history",
            "std_mae_ctrl": m_std.get("mae_control"),
            "ts_mae_ctrl": mae_ts_ctrl,
            "std_uai_irr": m_std.get("uai_irr"),
            "std_uai_pls": m_std.get("uai_plaus"),
            "std_disc": m_std.get("disc_delta"),
            "matched_uai_irr": m_matched.get("uai_irr"),
            "matched_uai_pls": m_matched.get("uai_plaus"),
            "matched_disc": m_matched.get("disc_delta"),
        })

    out = OUT_DIR / "a4_history_matched_control.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} rows to {out}")

    print("\n  Standard vs matched-control History:")
    print(f"  {'Model':<12s} {'MAE_std':>8s} {'MAE_ts':>8s} {'Disc_std':>9s} {'Disc_match':>10s}")
    for r in rows:
        print(f"  {r['model']:<12s} {r['std_mae_ctrl']:8.2f} {r['ts_mae_ctrl']:8.2f} "
              f"{r['std_disc']:9.2f} {r['matched_disc']:10.2f}")

    return rows


# ───────────────────────────────────────────────────────────────
# A5: ICL distribution-matching comparison
# ───────────────────────────────────────────────────────────────
def analysis_a5():
    """Compare ICL-dist (numerically relevant demos) with standard ICL."""
    print("\n=== A5: ICL distribution-matching comparison ===")
    rows = []

    if not ICL_DIST_DIR.is_dir():
        print("  SKIP: icl_dist_core not found")
        return rows

    for slug, rpath in iter_model_flat(ICL_DIST_DIR):
        short = MODEL_SHORT.get(slug, slug)
        dist_records = load_records(rpath)

        std_path = FULL_DIR / "icl" / slug / "results.jsonl"
        if not std_path.exists():
            continue
        std_records = load_records(std_path)

        m_std = compute_unified_metrics(std_records, epsilon=EPSILON)
        m_dist = compute_unified_metrics(dist_records, epsilon=EPSILON)

        rows.append({
            "model": short,
            "std_uai_irr": m_std.get("uai_irr"),
            "std_uai_pls": m_std.get("uai_plaus"),
            "std_disc": m_std.get("disc_delta"),
            "std_mae": m_std.get("mae_control"),
            "dist_uai_irr": m_dist.get("uai_irr"),
            "dist_uai_pls": m_dist.get("uai_plaus"),
            "dist_disc": m_dist.get("disc_delta"),
            "dist_mae": m_dist.get("mae_control"),
        })

    out = OUT_DIR / "a5_icl_dist_comparison.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} rows to {out}")

    print("\n  Standard ICL vs ICL-dist:")
    print(f"  {'Model':<12s} {'UAI_pls_std':>12s} {'UAI_pls_dist':>13s} {'Disc_std':>9s} {'Disc_dist':>10s}")
    for r in rows:
        std_p = f"{r['std_uai_pls']:.3f}" if r['std_uai_pls'] is not None else "N/A"
        dist_p = f"{r['dist_uai_pls']:.3f}" if r['dist_uai_pls'] is not None else "N/A"
        std_d = f"{r['std_disc']:.3f}" if r['std_disc'] is not None else "N/A"
        dist_d = f"{r['dist_disc']:.3f}" if r['dist_disc'] is not None else "N/A"
        print(f"  {r['model']:<12s} {std_p:>12s} {dist_p:>13s} {std_d:>9s} {dist_d:>10s}")

    return rows


if __name__ == "__main__":
    print("=" * 60)
    print("  COLM Revision Analyses")
    print("=" * 60)

    a1 = analysis_a1()
    a2 = analysis_a2()
    a3 = analysis_a3()
    a4 = analysis_a4()
    a5 = analysis_a5()

    print("\n" + "=" * 60)
    print("  All analyses complete. Outputs in:")
    print(f"  {OUT_DIR}")
    print("=" * 60)
