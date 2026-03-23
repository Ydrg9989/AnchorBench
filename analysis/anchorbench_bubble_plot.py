#!/usr/bin/env python3
"""Generate supplementary bubble plots for AnchorBench."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "full_benchmark" / "unified_all_suites.json"
FIG_DIR = ROOT / "outputs" / "figures"
STATS_DIR = ROOT / "outputs" / "stats"
TABLE_DIR = ROOT / "outputs" / "tables"
SUMMARY_MD = STATS_DIR / "BUBBLE_PLOT_SUMMARY.md"

SUITE_ORDER = ["External", "History", "Icl", "Rag", "Tool"]
COLORS = {
    "Qwen": "#1f77b4",
    "Llama": "#ff7f0e",
    "Gemma": "#2ca02c",
    "OLMo": "#d62728",
    "Other": "#9467bd",
}


def family_from_slug(slug: str) -> str:
    s = slug.lower()
    if "qwen" in s:
        return "Qwen"
    if "llama" in s:
        return "Llama"
    if "gemma" in s:
        return "Gemma"
    if "olmo" in s:
        return "OLMo"
    return "Other"


def params_from_model(model: str, slug: str) -> float | None:
    # Extract B-scale from model text first (e.g., Qwen-1.5B, OLMo-32B).
    for text in [model, slug]:
        m = re.search(r"(\d+(?:\.\d+)?)B", text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def load_data() -> pd.DataFrame:
    with open(SRC, encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["suite"] = pd.Categorical(df["suite"], categories=SUITE_ORDER, ordered=True)
    return df


def aggregate(df: pd.DataFrame, include_tool: bool = True) -> pd.DataFrame:
    d = df.copy()
    if not include_tool:
        d = d[d["suite"] != "Tool"]

    out = (
        d.groupby(["model", "model_slug"], as_index=False)
        .agg(
            mean_abs_uai_irr=("uai_irr", lambda s: s.abs().mean()),
            mean_abs_uai_pls=("uai_plaus", lambda s: s.abs().mean()),
            mean_disc=("disc_delta", "mean"),
            mean_abs_disc=("disc_delta", lambda s: s.abs().mean()),
            mean_acc10_c=("acc10_control", "mean"),
            n_suites=("suite", "nunique"),
        )
    )
    out["mean_abs_uai"] = (out["mean_abs_uai_irr"] + out["mean_abs_uai_pls"]) / 2.0
    out["family"] = out["model_slug"].map(family_from_slug)
    out["params_b"] = [params_from_model(m, s) for m, s in zip(out["model"], out["model_slug"])]
    out["capability"] = out["mean_acc10_c"]  # fallback since Arena not available
    out["capability_source"] = "mean_acc10_c (Arena unavailable)"
    # Bubble area scale for readability
    out["bubble_area"] = out["params_b"].fillna(out["params_b"].median()).pow(1.35) * 120
    return out.sort_values("model").reset_index(drop=True)


def _bubble_plot(df: pd.DataFrame, y_col: str, y_label: str, title: str, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    for fam, g in df.groupby("family"):
        ax.scatter(
            g["capability"],
            g[y_col],
            s=g["bubble_area"],
            alpha=0.72,
            label=fam,
            color=COLORS.get(fam, COLORS["Other"]),
            edgecolors="black",
            linewidths=0.6,
        )
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["capability"], r[y_col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Capability (mean Acc10_c)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(title="Family", fontsize=8)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=320)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    df = load_data()
    agg_all = aggregate(df, include_tool=True)
    agg_no_tool = aggregate(df, include_tool=False)

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Save aggregates
    agg_all.to_csv(STATS_DIR / "bubble_model_aggregate_all_suites.csv", index=False)
    agg_no_tool.to_csv(STATS_DIR / "bubble_model_aggregate_excl_tool.csv", index=False)
    agg_all.to_latex(TABLE_DIR / "bubble_model_aggregate_all_suites.tex", index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    agg_no_tool.to_latex(TABLE_DIR / "bubble_model_aggregate_excl_tool.tex", index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))

    figs: list[Path] = []
    figs += _bubble_plot(
        agg_all,
        y_col="mean_abs_uai",
        y_label="Mean Absolute UAI",
        title="Capability vs Aggregate Anchoring Strength (All Suites)",
        stem="bubble_capability_vs_mean_abs_uai_all",
    )
    figs += _bubble_plot(
        agg_all,
        y_col="mean_disc",
        y_label="Mean DiscΔ",
        title="Capability vs Mean DiscΔ (All Suites)",
        stem="bubble_capability_vs_mean_disc_all",
    )
    figs += _bubble_plot(
        agg_no_tool,
        y_col="mean_abs_uai",
        y_label="Mean Absolute UAI",
        title="Capability vs Aggregate Anchoring Strength (Excluding Tool)",
        stem="bubble_capability_vs_mean_abs_uai_excl_tool",
    )
    figs += _bubble_plot(
        agg_no_tool,
        y_col="mean_disc",
        y_label="Mean DiscΔ",
        title="Capability vs Mean DiscΔ (Excluding Tool)",
        stem="bubble_capability_vs_mean_disc_excl_tool",
    )

    md = []
    md.append("# AnchorBench Supplementary Bubble Plot Summary")
    md.append("")
    md.append("## Source file")
    md.append(f"- `{SRC.relative_to(ROOT)}`")
    md.append("")
    md.append("## Metadata availability")
    md.append("- Chatbot Arena score: not found in repository; used fallback `mean_acc10_c` for x-axis.")
    md.append("- Parameter count: parsed from model names/slugs (B-scale values).")
    md.append("- Model family/developer: inferred from model slug (`Qwen`, `Llama`, `Gemma`, `OLMo`).")
    md.append("")
    md.append("## Aggregation formulas (one row per model)")
    md.append("- `mean_abs_uai = average_over_suites( mean(|UAI_irr|, |UAI_pls|) )`")
    md.append("- `mean_disc = average_over_suites(DiscΔ)`")
    md.append("- `mean_abs_disc = average_over_suites(|DiscΔ|)`")
    md.append("- `mean_acc10_c = average_over_suites(Acc10_c)`")
    md.append("")
    md.append("## Artifacts")
    for p in figs:
        md.append(f"- `{p.relative_to(ROOT)}`")
    md.append(f"- `{(STATS_DIR / 'bubble_model_aggregate_all_suites.csv').relative_to(ROOT)}`")
    md.append(f"- `{(STATS_DIR / 'bubble_model_aggregate_excl_tool.csv').relative_to(ROOT)}`")
    md.append(f"- `{(TABLE_DIR / 'bubble_model_aggregate_all_suites.tex').relative_to(ROOT)}`")
    md.append(f"- `{(TABLE_DIR / 'bubble_model_aggregate_excl_tool.tex').relative_to(ROOT)}`")
    md.append("")
    md.append("## Interpretation caution")
    md.append("- This is a global supplementary summary, not a primary causal/inferential figure.")
    md.append("- Tool-inclusive and Tool-excluded versions are both provided to avoid overclaiming under protocol-sensitive settings.")
    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")

    print("Generated bubble plot package:")
    for p in figs:
        print(" -", p.relative_to(ROOT))
    print(" -", (STATS_DIR / "bubble_model_aggregate_all_suites.csv").relative_to(ROOT))
    print(" -", (STATS_DIR / "bubble_model_aggregate_excl_tool.csv").relative_to(ROOT))
    print(" -", SUMMARY_MD.relative_to(ROOT))


if __name__ == "__main__":
    main()

