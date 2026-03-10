#!/usr/bin/env python3
"""Recompute all AnchorBench metrics using the unified framework.

Reads existing results.jsonl files (NO re-running of inference) and
produces a single comparison table across External, History, and RAG.

Usage:
    python scripts/eval/recompute_all_unified.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from unified_metrics import load_records, compute_unified_metrics, print_summary

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"

MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
}

RUNS = [
    # (suite_label, model_short, results_path)
    ("External", "Qwen2.5-7B",  RESULTS_ROOT / "external_v2_pilot_Qwen25_7B_512/results.jsonl"),
    ("External", "Qwen2.5-3B",  RESULTS_ROOT / "external_v2_pilot_Qwen25_3B_512/results.jsonl"),
    ("External", "Llama-3.1-8B", RESULTS_ROOT / "external_v2_pilot_Llama31_8B_512/results.jsonl"),
    ("External", "Llama-3.2-3B", RESULTS_ROOT / "external_v2_pilot_Llama32_3B_512/results.jsonl"),

    ("History",  "Qwen2.5-7B",  RESULTS_ROOT / "history_v2_pilot_Qwen25_7B_512/results.jsonl"),
    ("History",  "Qwen2.5-3B",  RESULTS_ROOT / "history_v2_pilot_Qwen25_3B_512/results.jsonl"),
    ("History",  "Llama-3.1-8B", RESULTS_ROOT / "history_v2_pilot_Llama31_8B_512/results.jsonl"),
    ("History",  "Llama-3.2-3B", RESULTS_ROOT / "history_v2_pilot_Llama32_3B_512/results.jsonl"),

    ("RAG",      "Qwen2.5-7B",  RESULTS_ROOT / "rag_v2_pilot/Qwen_Qwen2.5-7B-Instruct/results.jsonl"),
    ("RAG",      "Qwen2.5-3B",  RESULTS_ROOT / "rag_v2_pilot/Qwen_Qwen2.5-3B-Instruct/results.jsonl"),
    ("RAG",      "Llama-3.1-8B", RESULTS_ROOT / "rag_v2_pilot/meta-llama_Llama-3.1-8B-Instruct/results.jsonl"),
    ("RAG",      "Llama-3.2-3B", RESULTS_ROOT / "rag_v2_pilot/meta-llama_Llama-3.2-3B-Instruct/results.jsonl"),
]


def fmt(v, decimals=2):
    if v is None:
        return "---"
    return f"{v:.{decimals}f}"


def fmt_pct(v):
    if v is None:
        return "---"
    return f"{v*100:.1f}%"


def main():
    all_results = []
    missing = []

    for suite, model, path in RUNS:
        if not path.exists():
            missing.append((suite, model, str(path)))
            continue
        records = load_records(path)
        metrics = compute_unified_metrics(records)
        metrics["suite"] = suite
        metrics["model"] = model
        all_results.append(metrics)

        # Write unified summary next to the original
        out_path = path.parent / "unified_summary.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

    if missing:
        print("WARNING: Missing result files:")
        for s, m, p in missing:
            print(f"  {s} / {m}: {p}")
        print()

    # --- Print per-suite tables ---
    for suite in ("External", "History", "RAG"):
        rows = [r for r in all_results if r["suite"] == suite]
        if not rows:
            continue

        print(f"\n{'='*90}")
        print(f"  {suite} Suite — Unified Metrics")
        print(f"{'='*90}")
        header = f"{'Model':<16} {'MAE_c':>6} {'Acc10_c':>8} {'UAI_irr':>8} {'UAI_pls':>8} {'TAR_irr':>8} {'TAR_pls':>8} {'Disc_Δ':>8} {'Parse':>7}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['model']:<16} "
                f"{fmt(r['mae_control']):>6} "
                f"{fmt_pct(r['acc10_control']):>8} "
                f"{fmt(r['uai_irr'], 3):>8} "
                f"{fmt(r['uai_plaus'], 3):>8} "
                f"{fmt(r['tar_irr'], 3):>8} "
                f"{fmt(r['tar_plaus'], 3):>8} "
                f"{fmt(r['disc_delta'], 3):>8} "
                f"{fmt_pct(r['parse_rate']):>7}"
            )

        # History appendix metrics
        if suite == "History":
            print(f"\n  History secondary (appendix): ACR / RR (plausible conditions)")
            print(f"  {'Model':<16} {'ACR':>8} {'RR':>8}")
            print(f"  {'-'*34}")
            for r in rows:
                print(f"  {r['model']:<16} {fmt(r.get('acr_mean'), 3):>8} {fmt(r.get('rr_mean'), 3):>8}")
        print()

    # --- LaTeX tables ---
    print("\n" + "="*90)
    print("  LaTeX table rows (copy-paste into paper)")
    print("="*90)

    for suite in ("External", "History", "RAG"):
        rows = [r for r in all_results if r["suite"] == suite]
        if not rows:
            continue
        print(f"\n% --- {suite} suite ---")
        for r in rows:
            model = r["model"]
            mae = fmt(r["mae_control"])
            acc = f"{r['acc10_control']*100:.1f}\\%" if r["acc10_control"] is not None else "---"
            ui = fmt(r["uai_irr"], 2)
            up = fmt(r["uai_plaus"], 2)
            ti = fmt(r["tar_irr"], 2)
            tp = fmt(r["tar_plaus"], 2)
            dd = fmt(r["disc_delta"], 2)
            print(f"{model:<16} & {mae} & {acc} & {ui} & {up} & {ti} & {tp} & {dd} \\\\")

    # --- Cross-suite comparison summary ---
    print("\n" + "="*90)
    print("  Cross-suite summary (all suites, all models)")
    print("="*90)
    header = f"{'Suite':<10} {'Model':<16} {'MAE_c':>6} {'Acc10_c':>8} {'UAI_irr':>8} {'UAI_pls':>8} {'Disc_Δ':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['suite']:<10} "
            f"{r['model']:<16} "
            f"{fmt(r['mae_control']):>6} "
            f"{fmt_pct(r['acc10_control']):>8} "
            f"{fmt(r['uai_irr'], 3):>8} "
            f"{fmt(r['uai_plaus'], 3):>8} "
            f"{fmt(r['disc_delta'], 3):>8}"
        )

    # Save master JSON
    out = RESULTS_ROOT / "unified_all_suites.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Master JSON written to {out}")


if __name__ == "__main__":
    main()
