#!/usr/bin/env python3
"""Generate paper-ready AnchorBench figures/tables from source-of-truth results."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results" / "full_benchmark"
UNIFIED_PATH = RESULTS_ROOT / "unified_all_suites.json"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
SUMMARY_MD = ROOT / "outputs" / "ANCHORBENCH_RESULT_PACKAGE.md"

SUITE_ORDER = ["External", "History", "Icl", "Rag", "Tool"]
SUITE_LABEL = {"External": "External", "History": "History", "Icl": "ICL", "Rag": "RAG", "Tool": "Tool"}
SUITE_COLOR = {
    "External": "#1f77b4",
    "History": "#ff7f0e",
    "Icl": "#2ca02c",
    "Rag": "#d62728",
    "Tool": "#9467bd",
}
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


def load_unified() -> pd.DataFrame:
    with open(UNIFIED_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["suite"] = pd.Categorical(df["suite"], categories=SUITE_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(["model", "suite"]).reset_index(drop=True)


def save_df(df: pd.DataFrame, stem: str) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TAB_DIR / f"{stem}.csv"
    tex_path = TAB_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(
        tex_path,
        index=False,
        na_rep="---",
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
    )


def plot_disc_heatmap(df: pd.DataFrame) -> list[Path]:
    pivot = (
        df.pivot(index="model", columns="suite", values="disc_delta")
        .reindex(index=MODEL_ORDER, columns=SUITE_ORDER)
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(SUITE_ORDER)))
    ax.set_xticklabels([SUITE_LABEL[s] for s in SUITE_ORDER], rotation=0)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_title("DiscΔ Heatmap by Model and Suite")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("DiscΔ")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "disc_delta_heatmap.png"
    pdf = FIG_DIR / "disc_delta_heatmap.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_acc_vs_disc(df: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for suite in SUITE_ORDER:
        d = df[df["suite"] == suite]
        ax.scatter(
            d["acc10_control"],
            d["disc_delta"],
            label=SUITE_LABEL[suite],
            s=46,
            alpha=0.9,
            color=SUITE_COLOR[suite],
            edgecolors="black",
            linewidths=0.3,
        )
    ax.set_xlabel("Acc10_c")
    ax.set_ylabel("DiscΔ")
    ax.set_title("Control Accuracy vs Anchoring Discrimination")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "acc10_vs_disc_delta_scatter.png"
    pdf = FIG_DIR / "acc10_vs_disc_delta_scatter.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_uai_scatter(df: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    for suite in SUITE_ORDER:
        d = df[df["suite"] == suite]
        ax.scatter(
            d["uai_irr"],
            d["uai_plaus"],
            label=SUITE_LABEL[suite],
            s=44,
            alpha=0.9,
            color=SUITE_COLOR[suite],
            edgecolors="black",
            linewidths=0.3,
        )
    ax.plot([df["uai_irr"].min() - 0.05, df["uai_irr"].max() + 0.05], [df["uai_irr"].min() - 0.05, df["uai_irr"].max() + 0.05], "--", linewidth=1)
    ax.set_xlabel("UAI_irr")
    ax.set_ylabel("UAI_plaus")
    ax.set_title("UAI Irrelevant vs Plausible by Suite")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "uai_irr_vs_uai_plaus_scatter.png"
    pdf = FIG_DIR / "uai_irr_vs_uai_plaus_scatter.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def parse_failure_category(raw: str) -> str:
    if not raw or len(raw.strip()) < 2:
        return "empty"
    t = raw.strip()
    if t.startswith("ERROR:"):
        return "error_prefix"
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and ("tool_calls" in obj or "name" in obj or "function" in obj):
            return "tool_protocol"
    except Exception:
        pass
    ints = [int(m.group()) for m in re.finditer(r"(?<![\d.])-?\d+(?!\d)(?!\.\d)", t)]
    if ints and all(v < 0 or v > 100 for v in ints):
        return "out_of_range"
    if not ints:
        return "no_integer"
    return "other_parser"


def build_parse_failure_summary() -> pd.DataFrame:
    rows = []
    for p in sorted((RESULTS_ROOT).glob("*/*/results.jsonl")):
        suite = p.parent.parent.name
        model_slug = p.parent.name
        with open(p, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        fails = [r for r in recs if not r.get("parsed_ok")]
        cats = {}
        for r in fails:
            c = parse_failure_category(r.get("raw_text", "") or "")
            cats[c] = cats.get(c, 0) + 1
        rows.append(
            {
                "suite": suite.capitalize() if suite != "icl" else "Icl",
                "model_slug": model_slug,
                "n_records": len(recs),
                "n_failures": len(fails),
                "parse_rate": (len(recs) - len(fails)) / len(recs) if recs else 1.0,
                "tool_protocol": cats.get("tool_protocol", 0),
                "out_of_range": cats.get("out_of_range", 0),
                "no_integer": cats.get("no_integer", 0),
                "other_parser": cats.get("other_parser", 0),
                "empty": cats.get("empty", 0),
                "error_prefix": cats.get("error_prefix", 0),
            }
        )
    out = pd.DataFrame(rows)
    out["suite"] = pd.Categorical(out["suite"], categories=SUITE_ORDER, ordered=True)
    return out.sort_values(["suite", "model_slug"]).reset_index(drop=True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    df = load_unified()

    # Main-paper compact table
    compact = df[["suite", "model", "acc10_control", "disc_delta", "parse_rate"]].copy()
    compact.columns = ["suite", "model", "Acc10_c", "DiscDelta", "Parse"]
    save_df(compact, "main_condensed_benchmark_summary")

    # Optional claim/effect table from computable model-suite metrics
    effect = df[
        [
            "suite",
            "model",
            "uai_irr",
            "uai_plaus",
            "disc_delta",
            "disc_delta_ci",
            "p_plaus_vs_irr_bh",
            "parse_rate",
        ]
    ].copy()
    effect["disc_delta_ci_lo"] = effect["disc_delta_ci"].apply(lambda x: x.get("lo") if isinstance(x, dict) else None)
    effect["disc_delta_ci_hi"] = effect["disc_delta_ci"].apply(lambda x: x.get("hi") if isinstance(x, dict) else None)
    effect["plausible_gt_irrelevant"] = effect["uai_plaus"] > effect["uai_irr"]
    effect = effect.drop(columns=["disc_delta_ci"])
    save_df(effect, "optional_headline_effects_model_suite")

    # Appendix full metrics and UAI decomposition
    appendix_full = df[
        [
            "suite",
            "model",
            "mae_control",
            "acc10_control",
            "uai_irr",
            "uai_plaus",
            "tar_irr",
            "tar_plaus",
            "disc_delta",
            "parse_rate",
            "p_plaus_vs_irr_bh",
            "p_irr_vs_zero_bh",
            "p_plaus_vs_zero_bh",
        ]
    ].copy()
    save_df(appendix_full, "appendix_full_metric_table")

    appendix_uai = df[
        ["suite", "model", "uai_irr_low", "uai_irr_high", "uai_plaus_low", "uai_plaus_high", "uai_irr", "uai_plaus", "disc_delta"]
    ].copy()
    save_df(appendix_uai, "appendix_uai_decomposition")

    parse_summary = build_parse_failure_summary()
    save_df(parse_summary, "appendix_parse_failure_summary")

    # Figures
    fig_paths = []
    fig_paths.extend(plot_disc_heatmap(df))
    fig_paths.extend(plot_acc_vs_disc(df))
    fig_paths.extend(plot_uai_scatter(df))

    # markdown summary
    md = []
    md.append("# AnchorBench Main-Paper Result Package")
    md.append("")
    md.append("## Source of truth")
    md.append(f"- `{UNIFIED_PATH.relative_to(ROOT)}` (machine-readable unified metrics)")
    md.append(f"- `{RESULTS_ROOT.relative_to(ROOT)}/**/results.jsonl` (raw outputs for parse summaries)")
    md.append("")
    md.append("## Main figures")
    md.append("- `outputs/figures/disc_delta_heatmap.(png|pdf)` supports claim (1): interface dependence.")
    md.append("- `outputs/figures/acc10_vs_disc_delta_scatter.(png|pdf)` supports claim (3): control accuracy vs robustness dissociation.")
    md.append("- `outputs/figures/uai_irr_vs_uai_plaus_scatter.(png|pdf)` supports claim (2): plausible > irrelevant trends.")
    md.append("")
    md.append("## Main tables")
    md.append("- `outputs/tables/main_condensed_benchmark_summary.(csv|tex)` compact COLM-style table (`Acc10_c`, `DiscΔ`, `Parse`).")
    md.append("- `outputs/tables/optional_headline_effects_model_suite.(csv|tex)` optional effects with CI / BH-adjusted p-value at model-suite granularity.")
    md.append("- `outputs/tables/tab_benchmark_disc_delta_wide.tex` — model × suite **DiscΔ** (booktabs; missing → `---`).")
    md.append("- `outputs/tables/tab_benchmark_parse_rate_wide.tex` — model × suite **parse rate** (\\%).")
    md.append("- `outputs/tables/tab_benchmark_condensed_pct.tex` — long-form Acc$_{10}^c$ / DiscΔ / Parse as percentages.")
    md.append("- Regenerate wide/percent tables: `python scripts/eval/export_latex_tables.py`.")
    md.append("")
    md.append("## Appendix")
    md.append("- `outputs/tables/appendix_full_metric_table.(csv|tex)` full metric matrix.")
    md.append("- `outputs/tables/appendix_uai_decomposition.(csv|tex)` low/high UAI decomposition.")
    md.append("- `outputs/tables/appendix_parse_failure_summary.(csv|tex)` parse/protocol failure counts.")
    md.append("")
    md.append("## Notes")
    md.append("- Tool suite has full metric coverage in current source-of-truth files, but parse/protocol failures remain non-negligible for some model-suite runs; this is explicitly surfaced via `Parse` and appendix parse-failure summary.")
    md.append("- No fabricated metrics were used.")

    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")

    print("Generated figures:")
    for p in fig_paths:
        print(" -", p.relative_to(ROOT))
    print("Generated tables in outputs/tables")
    print("Generated summary:", SUMMARY_MD.relative_to(ROOT))


if __name__ == "__main__":
    main()

