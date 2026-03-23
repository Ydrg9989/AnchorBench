#!/usr/bin/env python3
"""Generate all paper-quality figures from unified_all_suites.json.

Produces:
  1. Disc_Δ heatmap (model × suite)
  2. UAI plausible vs irrelevant dumbbell chart by suite
  3. Acc10 vs Disc_Δ scatter (bubble) plot
  4. UAI bar chart with CIs (grouped by suite)
  5. Parse rate heatmap
  6. Cross-suite radar chart per model family
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_PATH = Path("results/full_benchmark/unified_all_suites.json")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUITE_ORDER = ["External", "History", "Icl", "Rag", "Tool"]
SUITE_LABELS = {"External": "External", "History": "History", "Icl": "ICL", "Rag": "RAG", "Tool": "Tool"}

MODEL_ORDER = [
    "Gemma-1B", "Qwen-1.5B", "Llama-1B",
    "Llama-3B", "Qwen-3B", "Gemma-4B",
    "Qwen-7B", "Llama-8B",
    "OLMo-13B", "OLMo-32B",
]

FAMILY_COLORS = {
    "Gemma": "#E91E63",
    "Qwen": "#2196F3",
    "Llama": "#4CAF50",
    "OLMo": "#FF9800",
}

def get_family(model: str) -> str:
    for fam in FAMILY_COLORS:
        if fam.lower() in model.lower():
            return fam
    return "Other"

def get_param_b(model: str) -> float:
    import re
    m = re.search(r"(\d+\.?\d*)B", model)
    if m:
        return float(m.group(1))
    return 0

def load_data() -> pd.DataFrame:
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["suite_label"] = df["suite"].map(SUITE_LABELS)
    df["family"] = df["model"].apply(get_family)
    df["param_b"] = df["model"].apply(get_param_b)
    return df


def fig1_disc_delta_heatmap(df: pd.DataFrame):
    """Heatmap of Disc_Δ across models and suites."""
    pivot = df.pivot(index="model", columns="suite", values="disc_delta")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=SUITE_ORDER)
    pivot.columns = [SUITE_LABELS[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Disc$_{\\Delta}$"},
        vmin=-0.1, vmax=0.9,
    )
    ax.set_title("Discrimination Gap (Disc$_{\\Delta}$) by Model and Suite", fontsize=13, fontweight="bold")
    ax.set_xlabel("Suite")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_disc_delta_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_disc_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [1] Disc_Δ heatmap saved")


def fig2_uai_dumbbell(df: pd.DataFrame):
    """Dumbbell chart: UAI_irr vs UAI_pls per model, faceted by suite."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    for i, suite in enumerate(SUITE_ORDER):
        ax = axes[i]
        sub = df[df["suite"] == suite].set_index("model").reindex(MODEL_ORDER).reset_index()
        y = np.arange(len(sub))

        for j, row in sub.iterrows():
            color = FAMILY_COLORS.get(get_family(row["model"]), "gray")
            ax.plot([row["uai_irr"], row["uai_plaus"]], [j, j], color=color, linewidth=2, alpha=0.7)
            ax.scatter(row["uai_irr"], j, color=color, marker="o", s=60, zorder=5, edgecolors="white", linewidth=0.5)
            ax.scatter(row["uai_plaus"], j, color=color, marker="D", s=60, zorder=5, edgecolors="white", linewidth=0.5)

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(SUITE_LABELS[suite], fontsize=12, fontweight="bold")
        ax.set_xlabel("UAI")
        if i == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(sub["model"], fontsize=9)
        ax.set_xlim(-0.25, 1.0)
        ax.grid(axis="x", alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", label="UAI$_{irr}$", markersize=8, linestyle="None"),
        Line2D([0], [0], marker="D", color="gray", label="UAI$_{pls}$", markersize=8, linestyle="None"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Irrelevant vs Plausible Anchor Influence (UAI)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_uai_dumbbell.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_uai_dumbbell.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [2] UAI dumbbell chart saved")


def fig3_acc_vs_disc_bubble(df: pd.DataFrame):
    """Scatter/bubble: Acc10 (x) vs Disc_Δ (y), color by family, size by params."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    for i, suite in enumerate(SUITE_ORDER):
        ax = axes[i]
        sub = df[df["suite"] == suite].copy()
        sub["disc_delta_plot"] = sub["disc_delta"].fillna(0)

        for _, row in sub.iterrows():
            color = FAMILY_COLORS.get(row["family"], "gray")
            size = 40 + row["param_b"] * 12
            ax.scatter(
                row["acc10_control"] * 100, row["disc_delta_plot"],
                c=color, s=size, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=5,
            )
            ax.annotate(
                row["model"], (row["acc10_control"] * 100, row["disc_delta_plot"]),
                fontsize=6, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points",
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(SUITE_LABELS[suite], fontsize=12, fontweight="bold")
        ax.set_xlabel("Acc$_{10}$ (%)")
        if i == 0:
            ax.set_ylabel("Disc$_{\\Delta}$")
        ax.grid(alpha=0.3)
        ax.set_xlim(15, 105)
        ax.set_ylim(-0.15, 1.0)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Task Accuracy vs Discrimination Gap", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_acc_vs_disc_bubble.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_acc_vs_disc_bubble.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [3] Acc vs Disc bubble plot saved")


def fig4_uai_plaus_bar(df: pd.DataFrame):
    """Grouped bar chart of UAI_pls by suite, with CI whiskers."""
    fig, ax = plt.subplots(figsize=(14, 5))
    n_models = len(MODEL_ORDER)
    n_suites = len(SUITE_ORDER)
    bar_width = 0.15
    x = np.arange(n_models)

    suite_colors = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]
    for i, suite in enumerate(SUITE_ORDER):
        sub = df[df["suite"] == suite].set_index("model").reindex(MODEL_ORDER)
        vals = sub["uai_plaus"].fillna(0).values
        ci_lo = [sub.loc[m, "uai_plaus_ci"]["lo"] if pd.notna(sub.loc[m, "uai_plaus"]) and sub.loc[m, "uai_plaus_ci"] else 0 for m in MODEL_ORDER]
        ci_hi = [sub.loc[m, "uai_plaus_ci"]["hi"] if pd.notna(sub.loc[m, "uai_plaus"]) and sub.loc[m, "uai_plaus_ci"] else 0 for m in MODEL_ORDER]
        yerr_lo = [max(0, v - l) for v, l in zip(vals, ci_lo)]
        yerr_hi = [max(0, h - v) for v, h in zip(vals, ci_hi)]

        ax.bar(x + i * bar_width, vals, bar_width, label=SUITE_LABELS[suite],
               color=suite_colors[i], alpha=0.85, yerr=[yerr_lo, yerr_hi],
               capsize=2, error_kw={"linewidth": 0.8})

    ax.set_xticks(x + bar_width * (n_suites - 1) / 2)
    ax.set_xticklabels(MODEL_ORDER, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("UAI$_{pls}$")
    ax.set_title("Plausible Anchor Influence (UAI$_{pls}$) with 95% CI", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_uai_plaus_bar.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_uai_plaus_bar.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [4] UAI_pls grouped bar chart saved")


def fig5_parse_rate_heatmap(df: pd.DataFrame):
    """Heatmap of parse rates."""
    pivot = df.pivot(index="model", columns="suite", values="parse_rate")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=SUITE_ORDER)
    pivot.columns = [SUITE_LABELS[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot * 100, annot=True, fmt=".1f", cmap="YlGn",
        linewidths=0.5, ax=ax, vmin=75, vmax=100,
        cbar_kws={"label": "Parse Rate (%)"},
    )
    ax.set_title("Parse Rate (%) by Model and Suite", fontsize=13, fontweight="bold")
    ax.set_xlabel("Suite")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_parse_rate_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_parse_rate_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [5] Parse rate heatmap saved")


def fig6_acc10_heatmap(df: pd.DataFrame):
    """Heatmap of Acc10 control accuracy."""
    pivot = df.pivot(index="model", columns="suite", values="acc10_control")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=SUITE_ORDER)
    pivot.columns = [SUITE_LABELS[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot * 100, annot=True, fmt=".1f", cmap="Blues",
        linewidths=0.5, ax=ax, vmin=15, vmax=100,
        cbar_kws={"label": "Acc$_{10}$ (%)"},
    )
    ax.set_title("Task Accuracy (Acc$_{10}$, %) by Model and Suite", fontsize=13, fontweight="bold")
    ax.set_xlabel("Suite")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_acc10_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_acc10_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [6] Acc10 heatmap saved")


def fig7_combined_two_panel(df: pd.DataFrame):
    """Two-panel figure: Acc10 (left) vs Disc_Δ (right) heatmaps side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    pivot_acc = df.pivot(index="model", columns="suite", values="acc10_control")
    pivot_acc = pivot_acc.reindex(index=MODEL_ORDER, columns=SUITE_ORDER)
    pivot_acc.columns = [SUITE_LABELS[c] for c in pivot_acc.columns]

    pivot_disc = df.pivot(index="model", columns="suite", values="disc_delta")
    pivot_disc = pivot_disc.reindex(index=MODEL_ORDER, columns=SUITE_ORDER)
    pivot_disc.columns = [SUITE_LABELS[c] for c in pivot_disc.columns]

    sns.heatmap(pivot_acc * 100, annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.5, ax=ax1, vmin=15, vmax=100,
                cbar_kws={"label": "Acc$_{10}$ (%)"})
    ax1.set_title("(a) Task Accuracy", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Model")

    sns.heatmap(pivot_disc, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax2, vmin=-0.1, vmax=0.9,
                cbar_kws={"label": "Disc$_{\\Delta}$"})
    ax2.set_title("(b) Discrimination Gap", fontsize=12, fontweight="bold")
    ax2.set_ylabel("")

    fig.suptitle("Task Accuracy vs Discrimination: Orthogonal Dimensions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_acc_disc_two_panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig7_acc_disc_two_panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [7] Two-panel heatmap saved")


def fig8_suite_comparison_bar(df: pd.DataFrame):
    """Suite-level aggregated metrics (mean across models)."""
    suite_means = df.groupby("suite").agg({
        "uai_irr": "mean",
        "uai_plaus": "mean",
        "disc_delta": "mean",
        "acc10_control": "mean",
        "parse_rate": "mean",
    }).reindex(SUITE_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    suite_labels = [SUITE_LABELS[s] for s in SUITE_ORDER]
    colors = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]
    x = np.arange(len(SUITE_ORDER))

    axes[0].bar(x, suite_means["uai_plaus"], color=colors, alpha=0.85)
    axes[0].set_title("Mean UAI$_{pls}$", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(suite_labels, rotation=30)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, suite_means["disc_delta"], color=colors, alpha=0.85)
    axes[1].set_title("Mean Disc$_{\\Delta}$", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(suite_labels, rotation=30)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, suite_means["acc10_control"] * 100, color=colors, alpha=0.85)
    axes[2].set_title("Mean Acc$_{10}$ (%)", fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(suite_labels, rotation=30)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Suite-Level Aggregated Metrics (Mean Across 10 Models)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig8_suite_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig8_suite_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [8] Suite comparison bar chart saved")


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} entries: {df['model'].nunique()} models × {df['suite'].nunique()} suites")
    print()

    print("Generating figures...")
    fig1_disc_delta_heatmap(df)
    fig2_uai_dumbbell(df)
    fig3_acc_vs_disc_bubble(df)
    fig4_uai_plaus_bar(df)
    fig5_parse_rate_heatmap(df)
    fig6_acc10_heatmap(df)
    fig7_combined_two_panel(df)
    fig8_suite_comparison_bar(df)

    print()
    print(f"All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
