#!/usr/bin/env python3
"""End-to-end pipeline for the Anchoring Bias in LLMs PoC.

Supports multiple models in a single run. Each model gets its own output
directory, and a comparative report is generated at the end.

Usage:
    python scripts/run_pipeline.py \
        --models Qwen/Qwen3-4B-Instruct-2507 \
                 meta-llama/Llama-3.2-1B-Instruct \
                 meta-llama/Llama-3.2-3B-Instruct \
        --n-per-bin 200 --k-scenarios 67 --p-perms 15
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data_loader import (
    load_regime_a,
    load_regime_b,
    subsample_regime_a,
    subsample_regime_b,
)
from inference import ModelInference
from analysis import (
    parse_regime_a,
    parse_regime_b,
    compute_regime_a_metrics,
    compute_regime_b_metrics,
    plot_regime_a,
    plot_regime_b,
    plot_comparison_regime_a,
    plot_comparison_regime_b,
    save_regime_a_csv,
    save_regime_b_csv,
    X1_TO_ANCHOR_PCT,
)

LOG_INTERVAL_A = 50
LOG_INTERVAL_B = 25


def run_regime_a(model, sample, results_dir, figures_dir):
    print(f"\n{'='*60}")
    print("REGIME A: Prompt-induced anchoring (tum-nlp)")
    print(f"{'='*60}")
    n = len(sample)
    print(f"  Rows to evaluate: {n}")

    results = []
    t0 = time.time()
    for i, row in enumerate(sample):
        ctrl_raw = model.query_regime_a(row["control"])
        treat_raw = model.query_regime_a(row["treatment"])

        ctrl_idx = parse_regime_a(ctrl_raw)
        treat_idx = parse_regime_a(treat_raw)

        results.append({
            "idx": row["idx"],
            "scenario": row["scenario"],
            "x_1": row["x_1"],
            "control_idx": ctrl_idx,
            "treatment_idx": treat_idx,
            "control_raw": ctrl_raw,
            "treatment_raw": treat_raw,
        })

        if (i + 1) % LOG_INTERVAL_A == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            errs = sum(1 for r in results if r["control_idx"] is None or r["treatment_idx"] is None)
            print(f"  [{i+1}/{n}] {rate:.1f} rows/s | parse_errs={errs}")

    valid, agg = compute_regime_a_metrics(results)

    csv_path = results_dir / "results_regime_a.csv"
    save_regime_a_csv(results, csv_path)
    print(f"  Results CSV: {csv_path}")

    fig_path = figures_dir / "figure_regime_a.png"
    if valid:
        plot_regime_a(valid, agg, fig_path)
        print(f"  Figure:      {fig_path}")
    else:
        print("  No valid results to plot.")

    print(f"  Regime A done in {time.time() - t0:.0f}s "
          f"(valid={agg['n_valid']}/{agg['n_total']}, "
          f"ASS={agg.get('mean_abs_shift', 0):.3f})")
    return results, agg


def run_regime_b(model, scenarios, results_dir, figures_dir):
    print(f"\n{'='*60}")
    print("REGIME B: Sequential/order anchoring (jecht)")
    print(f"{'='*60}")
    total_calls = sum(sc["n_permutations"] for sc in scenarios.values())
    print(f"  Scenarios: {len(scenarios)}, total model calls: {total_calls}")

    results = defaultdict(lambda: defaultdict(list))
    call_num = 0
    parse_errs = 0
    t0 = time.time()

    for sid in sorted(scenarios):
        sc = scenarios[sid]
        students = sc["students"]
        n_students = sc["n_students"]

        for perm_idx, perm in enumerate(sc["permutations"]):
            call_num += 1

            raw = model.query_regime_b(students, perm)
            parsed = parse_regime_b(raw, n_students)

            if parsed is None:
                parse_errs += 1
                continue

            for pos, (student_idx, entry) in enumerate(zip(perm, parsed)):
                results[sid][student_idx].append({
                    "perm_idx": perm_idx,
                    "position": pos,
                    "decision": entry["decision"],
                    "confidence": entry["confidence"],
                })

            if call_num % LOG_INTERVAL_B == 0 or call_num == total_calls:
                elapsed = time.time() - t0
                rate = call_num / elapsed if elapsed > 0 else 0
                print(f"  [{call_num}/{total_calls}] {rate:.1f} calls/s | "
                      f"parse_errs={parse_errs}")

    results = {sid: dict(sd) for sid, sd in results.items()}

    csv_path = results_dir / "results_regime_b.csv"
    save_regime_b_csv(results, csv_path)
    print(f"  Results CSV: {csv_path}")

    agg = compute_regime_b_metrics(results)

    fig_path = figures_dir / "figure_regime_b.png"
    if agg["n_students_total"] > 0:
        plot_regime_b(agg, fig_path)
        print(f"  Figure:      {fig_path}")
    else:
        print("  No valid results to plot.")

    print(f"  Regime B done in {time.time() - t0:.0f}s "
          f"(students={agg['n_students_total']}, "
          f"flip_rate={agg['overall_flip_rate']:.3f})")
    return results, agg


def generate_model_report(agg_a, agg_b, model_name, output_path):
    """Short per-model report."""
    lines = [
        f"# Anchoring Bias PoC -- {model_name}\n",
        "## Regime A: Prompt-induced anchoring\n",
    ]
    if agg_a["n_valid"] > 0:
        lines.append(f"- **Rows evaluated:** {agg_a['n_total']} "
                      f"(valid: {agg_a['n_valid']}, "
                      f"parse errors: {agg_a['parse_error_rate']:.1%})")
        lines.append(f"- **Mean Absolute Shift:** {agg_a['mean_abs_shift']:.3f}")
        lines.append(f"- **Mean Signed Shift:** {agg_a['mean_signed_shift']:.3f}")
        lines.append(f"- **Pearson r:** {agg_a['pearson_r']:.3f}\n")
        lines.append("| x_1 | Anchor % | Mean Shift | Std | N |")
        lines.append("|-----|----------|------------|-----|---|")
        for x1, stats in sorted(agg_a["per_bin"].items()):
            pct = X1_TO_ANCHOR_PCT.get(x1, x1 * 10)
            lines.append(
                f"| {x1} | {pct}% | {stats['mean_signed_shift']:+.2f} "
                f"| {stats['std']:.2f} | {stats['n']} |"
            )
        lines.append("")
        lines.append("![Regime A](figure_regime_a.png)\n")
    else:
        lines.append("No valid results obtained.\n")

    lines.append("## Regime B: Sequential/order anchoring\n")
    if agg_b["n_students_total"] > 0:
        lines.append(f"- **Students evaluated:** {agg_b['n_students_total']}")
        lines.append(f"- **Overall flip rate:** {agg_b['overall_flip_rate']:.3f}")
        lines.append(f"- **Overall confidence variance:** {agg_b['overall_conf_var']:.1f}\n")
        lines.append("![Regime B](figure_regime_b.png)\n")
    else:
        lines.append("No valid results obtained.\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_comparison_report(all_aggs, figures_dir, output_path):
    """Cross-model comparison report."""
    models = list(all_aggs.keys())
    lines = [
        "# Anchoring Bias in LLMs -- Multi-Model Comparison Report\n",
        f"**Models:** {', '.join(f'`{m}`' for m in models)}\n",
        "## 1. Objective\n",
        "This report compares anchoring bias across multiple LLMs in two regimes:\n",
        "- **Regime A (prompt-induced):** control vs treatment prompts from "
        "tum-nlp/cognitive-biases-in-llms.\n",
        "- **Regime B (order-induced):** permuted student profiles from "
        "jecht/cognitive_bias.\n",
        "",
        "## 2. Regime A Comparison\n",
        "| Model | N Valid | Parse Err % | Mean |Shift| | Mean Shift | Pearson r |",
        "|-------|--------|-------------|-------------|------------|-----------|",
    ]
    for m in models:
        a = all_aggs[m]["regime_a"]
        lines.append(
            f"| {m} | {a.get('n_valid', 0)} | "
            f"{a.get('parse_error_rate', 0):.1%} | "
            f"{a.get('mean_abs_shift', 0):.3f} | "
            f"{a.get('mean_signed_shift', 0):.3f} | "
            f"{a.get('pearson_r', float('nan')):.3f} |"
        )
    lines.append("")
    lines.append("![Regime A Comparison](comparison_regime_a.png)\n")

    lines.append("## 3. Regime B Comparison\n")
    lines.append("| Model | N Students | Flip Rate | Conf Variance |")
    lines.append("|-------|-----------|-----------|---------------|")
    for m in models:
        b = all_aggs[m]["regime_b"]
        lines.append(
            f"| {m} | {b.get('n_students_total', 0)} | "
            f"{b.get('overall_flip_rate', 0):.3f} | "
            f"{b.get('overall_conf_var', 0):.1f} |"
        )
    lines.append("")
    lines.append("![Regime B Comparison](comparison_regime_b.png)\n")

    lines.append("## 4. Key Findings\n")
    a_shifts = {m: all_aggs[m]["regime_a"].get("mean_abs_shift", 0) for m in models}
    most_anchored = max(a_shifts, key=a_shifts.get)
    least_anchored = min(a_shifts, key=a_shifts.get)
    lines.append(f"- **Most anchored (Regime A):** `{most_anchored}` "
                  f"(mean |shift| = {a_shifts[most_anchored]:.3f})")
    lines.append(f"- **Least anchored (Regime A):** `{least_anchored}` "
                  f"(mean |shift| = {a_shifts[least_anchored]:.3f})")

    b_flips = {m: all_aggs[m]["regime_b"].get("overall_flip_rate", 0) for m in models}
    most_inconsistent = max(b_flips, key=b_flips.get)
    least_inconsistent = min(b_flips, key=b_flips.get)
    lines.append(f"- **Most order-sensitive (Regime B):** `{most_inconsistent}` "
                  f"(flip rate = {b_flips[most_inconsistent]:.3f})")
    lines.append(f"- **Least order-sensitive (Regime B):** `{least_inconsistent}` "
                  f"(flip rate = {b_flips[least_inconsistent]:.3f})\n")

    lines.append("## 5. Limitations\n")
    lines.append(
        "- **Temperature = 0:** Deterministic decoding eliminates stochastic "
        "variation but may mask anchoring that surfaces with sampling.\n"
        "- **Parse-error rate:** Structured output parsing may miss edge cases, "
        "especially for smaller models.\n"
        "- **Anchor mapping:** The `x_1` -> anchor percentage mapping was "
        "empirically determined from treatment text.\n"
        "- **Model size range:** Comparing 1B--4B models; larger models may "
        "show different anchoring patterns.\n"
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nComparison report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Anchoring Bias PoC -- Multi-Model")
    parser.add_argument(
        "--models", nargs="+",
        default=[
            "Qwen/Qwen3-4B-Instruct-2507",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        help="HuggingFace model names",
    )
    parser.add_argument("--n-per-bin", type=int, default=200,
                        help="Regime A: rows per x_1 bin (200 = all ~1000 rows)")
    parser.add_argument("--k-scenarios", type=int, default=67,
                        help="Regime B: number of scenario IDs (67 = all)")
    parser.add_argument("--p-perms", type=int, default=15,
                        help="Regime B: max permutations per scenario (15 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_root = PROJECT_ROOT / "results"
    figures_root = PROJECT_ROOT / "figures"

    # ---- Load datasets once ----
    print("Loading datasets...")
    t0 = time.time()
    regime_a_all = load_regime_a()
    regime_b_all = load_regime_b()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Regime A: {len(regime_a_all)} anchoring rows")
    print(f"  Regime B: {len(regime_b_all)} scenarios")

    sample_a = subsample_regime_a(
        regime_a_all, n_per_bin=args.n_per_bin, seed=args.seed,
    )
    sample_b = subsample_regime_b(
        regime_b_all, k_scenarios=args.k_scenarios,
        p_perms=args.p_perms, seed=args.seed,
    )
    total_b_calls = sum(sc["n_permutations"] for sc in sample_b.values())
    print(f"  Subsampled: Regime A = {len(sample_a)} rows, "
          f"Regime B = {len(sample_b)} scenarios / {total_b_calls} calls")

    # ---- Run each model ----
    all_aggs = {}

    for model_idx, model_name in enumerate(args.models):
        short = model_name.split("/")[-1]
        print(f"\n{'#'*60}")
        print(f"# MODEL {model_idx+1}/{len(args.models)}: {model_name}")
        print(f"{'#'*60}")

        res_dir = results_root / short
        fig_dir = figures_root / short
        res_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Loading model: {model_name}")
        t_load = time.time()
        model = ModelInference(model_name)
        print(f"  Model loaded in {time.time() - t_load:.1f}s")

        _, agg_a = run_regime_a(model, sample_a, res_dir, fig_dir)
        _, agg_b = run_regime_b(model, sample_b, res_dir, fig_dir)

        generate_model_report(
            agg_a, agg_b, model_name,
            fig_dir / "report.md",
        )

        metrics_path = res_dir / "metrics.json"
        serialisable_a = {k: v for k, v in agg_a.items()}
        serialisable_b = {
            k: v for k, v in agg_b.items() if k != "scenario_stats"
        }
        serialisable_b["scenario_stats"] = {
            str(sid): {k: v for k, v in ss.items() if k != "flip_rates"}
            for sid, ss in agg_b.get("scenario_stats", {}).items()
        }
        with open(metrics_path, "w") as f:
            json.dump({"regime_a": serialisable_a, "regime_b": serialisable_b},
                      f, indent=2)
        print(f"  Metrics JSON: {metrics_path}")

        all_aggs[short] = {"regime_a": agg_a, "regime_b": agg_b}

        model.cleanup()
        print(f"  GPU memory freed for {short}.")

    # ---- Comparative analysis ----
    if len(all_aggs) > 1:
        print(f"\n{'#'*60}")
        print("# CROSS-MODEL COMPARISON")
        print(f"{'#'*60}")

        comp_fig_dir = figures_root / "comparison"
        comp_fig_dir.mkdir(parents=True, exist_ok=True)

        a_aggs = {m: v["regime_a"] for m, v in all_aggs.items()}
        b_aggs = {m: v["regime_b"] for m, v in all_aggs.items()}

        comp_a_path = comp_fig_dir / "comparison_regime_a.png"
        plot_comparison_regime_a(a_aggs, comp_a_path)
        print(f"  Comparison figure A: {comp_a_path}")

        comp_b_path = comp_fig_dir / "comparison_regime_b.png"
        plot_comparison_regime_b(b_aggs, comp_b_path)
        print(f"  Comparison figure B: {comp_b_path}")

        report_path = comp_fig_dir / "comparison_report.md"
        generate_comparison_report(all_aggs, comp_fig_dir, report_path)

        comp_metrics_path = results_root / "comparison_metrics.json"
        comp_data = {}
        for m in all_aggs:
            a = all_aggs[m]["regime_a"]
            b = all_aggs[m]["regime_b"]
            comp_data[m] = {
                "regime_a": {
                    k: v for k, v in a.items() if k != "per_bin"
                },
                "regime_b": {
                    k: v for k, v in b.items()
                    if k not in ("scenario_stats", "position_effect")
                },
            }
        with open(comp_metrics_path, "w") as f:
            json.dump(comp_data, f, indent=2)
        print(f"  Comparison metrics: {comp_metrics_path}")

    print("\n" + "=" * 60)
    print("ALL MODELS COMPLETE")
    print("=" * 60)
    for m in all_aggs:
        a = all_aggs[m]["regime_a"]
        b = all_aggs[m]["regime_b"]
        print(f"  {m}: ASS={a.get('mean_abs_shift',0):.3f}, "
              f"r={a.get('pearson_r',0):.3f}, "
              f"flip={b.get('overall_flip_rate',0):.3f}")


if __name__ == "__main__":
    main()
