#!/usr/bin/env python3
"""Generate clean figures and consolidated table from existing sampling robustness data.

Reads the already-computed CSV at results/extension_summary/sampling_robustness.csv
and the raw JSONL files, produces publication-quality figures and LaTeX.

Usage:
    python scripts/eval/sampling_robustness_figures.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from anchorbench.eval.constants import MODEL_SHORT
from anchorbench.eval.io import load_records
from anchorbench.eval.metrics import compute_unified_metrics

ROOT = Path(__file__).resolve().parents[3]
SUITES = ["external", "rag", "icl_dist"]
METRIC_KEYS = ["uai_irr", "uai_plaus", "disc_delta", "mae_control",
               "acc10_control", "tar_irr", "tar_plaus", "parse_rate"]


def load_all_results(samp_dir: Path) -> list[dict]:
    """Load and compute metrics for every greedy + sampled run."""
    rows = []
    for suite in SUITES:
        suite_dir = samp_dir / suite
        if not suite_dir.is_dir():
            continue
        for model_dir in sorted(suite_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            slug = model_dir.name
            short = MODEL_SHORT.get(slug, slug)

            for run_dir in sorted(model_dir.iterdir()):
                rpath = run_dir / "results.jsonl"
                if not rpath.exists():
                    continue
                recs = load_records(str(rpath))
                m = compute_unified_metrics(recs)
                decoding = run_dir.name
                is_greedy = decoding == "greedy"

                rows.append({
                    "suite": suite,
                    "model": short,
                    "model_slug": slug,
                    "decoding": decoding,
                    "is_greedy": is_greedy,
                    **{k: m.get(k) for k in METRIC_KEYS},
                })
    return rows


def build_summary_table(rows: list[dict], out_dir: Path):
    """Consolidated table: greedy metrics, sampled mean±std, delta."""
    table_rows = []

    for suite in SUITES:
        for model in sorted(set(r["model"] for r in rows)):
            greedy = [r for r in rows if r["suite"] == suite
                      and r["model"] == model and r["is_greedy"]]
            sampled = [r for r in rows if r["suite"] == suite
                       and r["model"] == model and not r["is_greedy"]]
            if not greedy or not sampled:
                continue
            g = greedy[0]
            row = {"suite": suite, "model": model}
            for k in METRIC_KEYS:
                gv = g.get(k)
                sv = [s.get(k) for s in sampled if s.get(k) is not None]
                row[f"{k}_greedy"] = _r(gv)
                if sv:
                    row[f"{k}_sample_mean"] = _r(float(np.mean(sv)))
                    row[f"{k}_sample_std"] = _r(float(np.std(sv)))
                    row[f"{k}_delta"] = _r(float(np.mean(sv)) - gv if gv is not None else None)
                else:
                    row[f"{k}_sample_mean"] = None
                    row[f"{k}_sample_std"] = None
                    row[f"{k}_delta"] = None
            table_rows.append(row)

    csv_path = out_dir / "sampling_robustness_summary.csv"
    _write_csv(table_rows, csv_path)
    print(f"Wrote {csv_path}")

    # LaTeX
    latex_path = out_dir / "sampling_robustness_table.tex"
    _write_latex(table_rows, latex_path)
    print(f"Wrote {latex_path}")

    return table_rows


def _r(v, d=4):
    if v is None:
        return None
    return round(v, d)


def _write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_latex(table_rows, path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sampling robustness: greedy vs.\ stochastic decoding "
        r"($T{=}0.7$, $p{=}0.9$, 3~seeds). "
        r"$\Delta$ = mean(sampled) $-$ greedy.}",
        r"\label{tab:sampling_robustness}",
        r"\small",
        r"\begin{tabular}{ll rr rr rr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Disc}$_\Delta$} "
        r"& \multicolumn{2}{c}{\textbf{UAI}\textsubscript{pls}} "
        r"& \multicolumn{2}{c}{\textbf{MAE}\textsubscript{ctrl}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"\textbf{Suite} & \textbf{Model} & Greedy & $\Delta$ "
        r"& Greedy & $\Delta$ & Greedy & $\Delta$ \\",
        r"\midrule",
    ]

    for row in table_rows:
        suite = row["suite"]
        model = row["model"]
        dg = _fmt(row.get("disc_delta_greedy"), 3)
        dd = _fmt_delta(row.get("disc_delta_delta"), 3)
        ug = _fmt(row.get("uai_plaus_greedy"), 3)
        ud = _fmt_delta(row.get("uai_plaus_delta"), 3)
        mg = _fmt(row.get("mae_control_greedy"), 1)
        md = _fmt_delta(row.get("mae_control_delta"), 1)
        lines.append(
            f"{suite} & {model} & {dg} & {dd} & {ug} & {ud} & {mg} & {md} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines))


def _fmt(v, prec=3):
    if v is None:
        return "---"
    return f"{v:.{prec}f}"


def _fmt_delta(v, prec=3):
    if v is None:
        return "---"
    return f"{v:+.{prec}f}"


def make_figure(rows: list[dict], fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = [("disc_delta", r"Disc$_\Delta$"), ("uai_plaus", r"UAI$_{pls}$")]
    models = sorted(set(r["model"] for r in rows))
    suites_present = [s for s in SUITES if any(r["suite"] == s for r in rows)]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(11, 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    colors = {"Qwen-7B": "#1f77b4", "Llama-8B": "#d62728"}

    for ax, (mkey, mlabel) in zip(axes, metrics_to_plot):
        x_labels = []
        for si, suite in enumerate(suites_present):
            for mi, model in enumerate(models):
                x_labels.append(f"{suite}\n{model}")
                x_pos = si * (len(models) + 1) + mi

                greedy = [r for r in rows if r["suite"] == suite
                          and r["model"] == model and r["is_greedy"]]
                sampled = [r for r in rows if r["suite"] == suite
                           and r["model"] == model and not r["is_greedy"]]

                c = colors.get(model, "#333333")
                if greedy:
                    gv = greedy[0].get(mkey)
                    if gv is not None:
                        ax.scatter(x_pos, gv, marker="o", s=80, color=c,
                                   edgecolors="black", linewidths=0.8, zorder=3)

                if sampled:
                    sv = [s.get(mkey) for s in sampled if s.get(mkey) is not None]
                    for v in sv:
                        ax.scatter(x_pos + 0.15, v, marker="x", s=50,
                                   color=c, alpha=0.6, zorder=2)

        ax.set_xticks([si * (len(models) + 1) + mi
                       for si in range(len(suites_present))
                       for mi in range(len(models))])
        ax.set_xticklabels([f"{s[:3].upper()}\n{m}" for s in suites_present for m in models],
                           fontsize=8)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, label="Greedy"),
        Line2D([0], [0], marker="x", color="gray", linestyle="None",
               markersize=8, label="Sampled (T=0.7)"),
    ]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.suptitle("Sampling Robustness: Greedy vs Stochastic Decoding",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ["pdf", "png"]:
        p = fig_dir / f"sampling_robustness_comparison.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Sampling robustness figures & tables")
    p.add_argument("--samp_dir", type=Path,
                   default=Path("results/decoding_sampling_robustness"))
    p.add_argument("--out_dir", type=Path,
                   default=Path("results/revision/sampling_robustness"))
    p.add_argument("--fig_dir", type=Path,
                   default=Path("figures/revision"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all_results(args.samp_dir)
    if not rows:
        print("ERROR: No sampling robustness results found.")
        return

    build_summary_table(rows, args.out_dir)
    make_figure(rows, args.fig_dir)

    # README
    readme = args.out_dir / "README.md"
    readme.write_text(
        "# Sampling Robustness Analysis\n\n"
        "Generated from existing results in `results/decoding_sampling_robustness/`.\n\n"
        "## Reproduction\n"
        "```bash\n"
        "# Generate figures/tables from existing data:\n"
        "python scripts/eval/sampling_robustness_figures.py\n\n"
        "# To rerun inference (requires GPU):\n"
        "bash scripts/run_with_env.sh python scripts/eval/run_sampling_robustness.py \\\n"
        "    --model_id Qwen/Qwen2.5-7B-Instruct --backend vllm\n"
        "```\n"
    )
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
