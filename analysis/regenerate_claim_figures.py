#!/usr/bin/env python3
"""Regenerate AnchorBench claim-facing figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "full_benchmark" / "unified_all_suites.json"
OUT = ROOT / "outputs" / "figures"
SUMMARY = OUT / "ANCHORBENCH_CLAIM_FIGURES_SUMMARY.md"

SUITE_ORDER = ["External", "History", "Icl", "Rag", "Tool"]
SUITE_LABELS = ["External", "History", "ICL", "RAG", "Tool"]
MODEL_ORDER = [
    "Qwen-1.5B",
    "Qwen-3B",
    "Qwen-7B",
    "Llama-1B",
    "Llama-3B",
    "Llama-8B",
    "Gemma-1B",
    "Gemma-4B",
    "OLMo-13B",
    "OLMo-32B",
]


def load_df() -> pd.DataFrame:
    with open(SRC, encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["suite"] = pd.Categorical(df["suite"], categories=SUITE_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(["model", "suite"]).reset_index(drop=True)


def _heatmap_with_na(
    pivot: pd.DataFrame,
    title: str,
    cbar_label: str,
    cmap: str,
    out_stem: str,
    fmt: str = ".2f",
) -> list[Path]:
    arr = pivot.values.astype(float)
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="#d9d9d9")
    im = ax.imshow(masked, cmap=cm, aspect="auto")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([{"Icl": "ICL", "Rag": "RAG"}.get(c, c) for c in pivot.columns], fontsize=10)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index), fontsize=9)
    ax.set_title(title, fontsize=12, pad=10)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7, color="black")
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{out_stem}.png"
    pdf = OUT / f"{out_stem}.pdf"
    fig.savefig(png, dpi=320)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_disc_heatmap(df: pd.DataFrame) -> list[Path]:
    pivot = df.pivot(index="model", columns="suite", values="disc_delta").reindex(index=MODEL_ORDER, columns=SUITE_ORDER)
    return _heatmap_with_na(
        pivot=pivot,
        title="DiscΔ by Model × Interface",
        cbar_label="DiscΔ",
        cmap="YlGnBu",
        out_stem="disc_delta_heatmap",
        fmt=".2f",
    )


def plot_acc_heatmap(df: pd.DataFrame) -> list[Path]:
    pivot = df.pivot(index="model", columns="suite", values="acc10_control").reindex(index=MODEL_ORDER, columns=SUITE_ORDER)
    return _heatmap_with_na(
        pivot=pivot,
        title="Acc10_c by Model × Interface",
        cbar_label="Acc10_c",
        cmap="Blues",
        out_stem="acc10_c_heatmap",
        fmt=".2f",
    )


def plot_combined_heatmap(df: pd.DataFrame) -> list[Path]:
    disc = df.pivot(index="model", columns="suite", values="disc_delta").reindex(index=MODEL_ORDER, columns=SUITE_ORDER)
    acc = df.pivot(index="model", columns="suite", values="acc10_control").reindex(index=MODEL_ORDER, columns=SUITE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8))
    for ax, pivot, title, cmap, label in [
        (axes[0], acc, "Acc10_c", "Blues", "Acc10_c"),
        (axes[1], disc, "DiscΔ", "YlGnBu", "DiscΔ"),
    ]:
        arr = pivot.values.astype(float)
        masked = np.ma.masked_invalid(arr)
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(color="#d9d9d9")
        im = ax.imshow(masked, cmap=cm, aspect="auto")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([{"Icl": "ICL", "Rag": "RAG"}.get(c, c) for c in pivot.columns], fontsize=9)
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(list(pivot.index), fontsize=8)
        ax.set_title(title, fontsize=11)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = arr[i, j]
                ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "NA", ha="center", va="center", fontsize=6.8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(label, fontsize=9)

    fig.suptitle("AnchorBench: Control Accuracy vs Anchoring Discrimination", fontsize=13, y=1.01)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "acc10_disc_two_panel_heatmap.png"
    pdf = OUT / "acc10_disc_two_panel_heatmap.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_uai_dumbbell(df: pd.DataFrame) -> list[Path]:
    # One point per model-suite for UAI_irr and UAI_plaus, grouped by suite.
    d = df[["suite", "model", "uai_irr", "uai_plaus"]].copy().sort_values(["suite", "model"])
    y_pos = np.arange(len(d))
    suite_codes = d["suite"].cat.codes.values

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    suite_color = {s: colors[i] for i, s in enumerate(SUITE_ORDER)}

    fig, ax = plt.subplots(figsize=(10.8, 8.6))
    for i, row in d.reset_index(drop=True).iterrows():
        c = suite_color[row["suite"]]
        ax.plot([row["uai_irr"], row["uai_plaus"]], [i, i], color=c, alpha=0.45, linewidth=1.4)
        ax.scatter(row["uai_irr"], i, color=c, edgecolors="black", linewidths=0.3, s=30, marker="o")
        ax.scatter(row["uai_plaus"], i, color=c, edgecolors="black", linewidths=0.3, s=30, marker="s")

    # suite separators
    starts = d.groupby("suite", observed=True).size().cumsum().values
    for s in starts[:-1]:
        ax.axhline(s - 0.5, color="gray", linewidth=0.8, alpha=0.4)

    ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.5)
    ax.set_xlabel("UAI value")
    ax.set_ylabel("Model-suite pairs (grouped by suite)")
    ax.set_title("Paired UAI_irr vs UAI_plaus by Interface (Dumbbell Plot)")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=suite_color[s], lw=2, label={"Icl": "ICL", "Rag": "RAG"}.get(s, s)) for s in SUITE_ORDER]
    handles += [
        Line2D([0], [0], marker="o", color="black", lw=0, label="UAI_irr", markersize=6),
        Line2D([0], [0], marker="s", color="black", lw=0, label="UAI_plaus", markersize=6),
    ]
    ax.legend(handles=handles, ncol=4, fontsize=8, loc="upper right")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "uai_irr_vs_uai_plaus_dumbbell_by_suite.png"
    pdf = OUT / "uai_irr_vs_uai_plaus_dumbbell_by_suite.pdf"
    fig.savefig(png, dpi=320)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_suite_small_multiples(df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, suite in enumerate(SUITE_ORDER):
        ax = axes[i]
        d = df[df["suite"] == suite]
        ax.scatter(d["acc10_control"], d["disc_delta"], s=42, color="#1f77b4", edgecolors="black", linewidths=0.3, alpha=0.85)
        # spearman
        corr = d[["acc10_control", "disc_delta"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{ {'Icl':'ICL','Rag':'RAG'}.get(suite,suite) }  ρ={corr:.2f}", fontsize=10)
        ax.grid(alpha=0.25)
        for _, r in d.iterrows():
            ax.annotate(str(r["model"]), (r["acc10_control"], r["disc_delta"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    # hide 6th axis
    axes[-1].axis("off")
    fig.supxlabel("Acc10_c")
    fig.supylabel("DiscΔ")
    fig.suptitle("Appendix: Acc10_c vs DiscΔ by Suite (Spearman shown)", fontsize=12, y=0.98)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "appendix_small_multiples_acc10_vs_disc_by_suite.png"
    pdf = OUT / "appendix_small_multiples_acc10_vs_disc_by_suite.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def write_summary(df: pd.DataFrame, artifacts: list[Path]) -> None:
    # Tool partial-coverage caveat: check parse-rate spread in Tool suite
    tool = df[df["suite"] == "Tool"]
    tool_parse_min = float(tool["parse_rate"].min()) if not tool.empty else float("nan")
    tool_parse_max = float(tool["parse_rate"].max()) if not tool.empty else float("nan")
    tool_missing = int(df[df["suite"] == "Tool"]["disc_delta"].isna().sum())

    lines: list[str] = []
    lines.append("# AnchorBench Claim-Facing Figures Summary")
    lines.append("")
    lines.append("## Source files used")
    lines.append(f"- `{SRC.relative_to(ROOT)}` (model × suite machine-readable metrics)")
    lines.append("")
    lines.append("## Figures and claim mapping")
    lines.append("- `disc_delta_heatmap`: supports claim (1), interface dependence of anchoring strength.")
    lines.append("- `acc10_c_heatmap`: supports claim (3), capability differences independent from anchoring.")
    lines.append("- `acc10_disc_two_panel_heatmap`: directly contrasts capability vs anchoring across interfaces.")
    lines.append("- `uai_irr_vs_uai_plaus_dumbbell_by_suite`: supports claim (2), plausible vs irrelevant within interfaces.")
    lines.append("- `appendix_small_multiples_acc10_vs_disc_by_suite` (optional): per-suite association diagnostics; avoids overclaiming a single global trend.")
    lines.append("")
    lines.append("## Limitations / caveats")
    lines.append("- No missing values were fabricated; any absent model-suite cells are shown as `NA` (gray).")
    lines.append(
        f"- Tool suite metric cells missing: {tool_missing}. Tool parse-rate range across models: [{tool_parse_min:.4f}, {tool_parse_max:.4f}]."
    )
    lines.append("- Tool should be interpreted with protocol-compliance caveats; parse/protocol issues may contribute to observed behavior.")
    lines.append("")
    lines.append("## Generated artifacts")
    for p in artifacts:
        lines.append(f"- `{p.relative_to(ROOT)}`")

    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_df()
    OUT.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []
    artifacts.extend(plot_disc_heatmap(df))
    artifacts.extend(plot_acc_heatmap(df))
    artifacts.extend(plot_combined_heatmap(df))
    artifacts.extend(plot_uai_dumbbell(df))
    artifacts.extend(plot_suite_small_multiples(df))
    write_summary(df, artifacts + [SUMMARY])

    print("Generated claim-facing figures:")
    for p in artifacts:
        print(" -", p.relative_to(ROOT))
    print("Summary:")
    print(" -", SUMMARY.relative_to(ROOT))


if __name__ == "__main__":
    main()

