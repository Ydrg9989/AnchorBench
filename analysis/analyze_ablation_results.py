#!/usr/bin/env python3
"""Summarize and visualize AnchorBench ablation study results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ABL_ROOT = ROOT / "results" / "ablation_full_benchmark"
CORE_ROOT = ROOT / "results" / "full_benchmark"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
STATS_DIR = ROOT / "outputs" / "stats"
SUMMARY_MD = STATS_DIR / "ABLATION_STUDY_SUMMARY.md"

SUITES = ["external", "icl", "rag", "history"]
MODEL_ORDER = [
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct",
    "meta-llama_Llama-3.2-1B-Instruct",
    "meta-llama_Llama-3.2-3B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "google_gemma-3-1b-it",
    "google_gemma-3-4b-it",
    "allenai_OLMo-2-1124-13B-Instruct",
    "allenai_OLMo-2-0325-32B-Instruct",
]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_core_control_answer(suite: str, model_slug: str) -> dict[str, float]:
    p = CORE_ROOT / suite / model_slug / "results.jsonl"
    recs = load_jsonl(p)
    out = {}
    for r in recs:
        if r.get("condition") != "control":
            continue
        y = r.get("answer_int")
        if r.get("parsed_ok") and y is not None:
            out[r["item_id"]] = float(y)
    return out


def build_joined_ablation_df() -> pd.DataFrame:
    rows = []
    for suite in SUITES:
        for p in sorted((ABL_ROOT / suite).glob("*/results.jsonl")):
            model_slug = p.parent.name
            core_ctrl = get_core_control_answer(suite, model_slug)
            for r in load_jsonl(p):
                ans = r.get("answer_int")
                parsed = bool(r.get("parsed_ok")) and ans is not None
                ctrl = core_ctrl.get(r["item_id"])
                anchor = r.get("anchor_value")
                shift = None
                uai = None
                if parsed and ctrl is not None:
                    shift = float(ans) - float(ctrl)
                    if anchor is not None:
                        denom = float(anchor) - float(ctrl)
                        if abs(denom) >= 3:
                            uai = shift / denom
                rows.append(
                    {
                        "suite": suite,
                        "model_slug": model_slug,
                        "item_id": r["item_id"],
                        "condition": r.get("condition"),
                        "anchor_relevance": r.get("anchor_relevance", "none"),
                        "anchor_value": anchor,
                        "parsed_ok": parsed,
                        "answer_int": ans if parsed else None,
                        "core_control_answer": ctrl,
                        "shift_vs_core_control": shift,
                        "uai_vs_core_control": uai,
                    }
                )
    return pd.DataFrame(rows)


def save_df(df: pd.DataFrame, stem: str) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{stem}.csv", index=False)
    df.to_latex(
        TAB_DIR / f"{stem}.tex",
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
    )


def make_figures(df: pd.DataFrame) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # 1) parse-rate heatmap (suite x model)
    pr = (
        df.groupby(["suite", "model_slug"], as_index=False)["parsed_ok"].mean()
        .pivot(index="model_slug", columns="suite", values="parsed_ok")
        .reindex(index=MODEL_ORDER, columns=SUITES)
    )
    arr = np.ma.masked_invalid(pr.values.astype(float))
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    cm = plt.get_cmap("Blues").copy()
    cm.set_bad("#d9d9d9")
    im = ax.imshow(arr, cmap=cm, aspect="auto", vmin=0.8, vmax=1.0)
    ax.set_xticks(range(len(SUITES)))
    ax.set_xticklabels([s.upper() for s in SUITES])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=8)
    ax.set_title("Ablation Parse Rate by Model × Suite")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = pr.values[i, j]
            ax.text(j, i, "NA" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Parse rate")
    fig.tight_layout()
    p1 = FIG_DIR / "ablation_parse_rate_heatmap.png"
    p2 = FIG_DIR / "ablation_parse_rate_heatmap.pdf"
    fig.savefig(p1, dpi=300)
    fig.savefig(p2)
    plt.close(fig)
    paths += [p1, p2]

    # 2) suite-level UAI bars for ablation conditions
    ab = df[df["uai_vs_core_control"].notna()].copy()
    ab["suite_cond"] = ab["suite"] + " | " + ab["condition"]
    g = ab.groupby(["suite", "condition"], as_index=False)["uai_vs_core_control"].mean()
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    x = np.arange(len(g))
    ax.bar(x, g["uai_vs_core_control"], color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}:{c}" for s, c in zip(g["suite"], g["condition"])], rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Mean UAI vs Core Control")
    ax.set_title("Ablation Condition Effects (UAI)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p3 = FIG_DIR / "ablation_condition_uai_bar.png"
    p4 = FIG_DIR / "ablation_condition_uai_bar.pdf"
    fig.savefig(p3, dpi=300)
    fig.savefig(p4)
    plt.close(fig)
    paths += [p3, p4]

    # 3) history control_twostage delta vs core control
    h = df[(df["suite"] == "history") & (df["condition"] == "control_twostage") & (df["shift_vs_core_control"].notna())]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    vals = h["shift_vs_core_control"].to_numpy(float)
    ax.hist(vals, bins=30, color="#59a14f", alpha=0.85)
    ax.axvline(np.mean(vals), color="black", linestyle="--", linewidth=1.2, label=f"mean={np.mean(vals):.2f}")
    ax.axvline(0.0, color="gray", linewidth=1.0)
    ax.set_title("History Ablation: control_twostage - core control")
    ax.set_xlabel("Answer shift")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    p5 = FIG_DIR / "ablation_history_control_twostage_shift_hist.png"
    p6 = FIG_DIR / "ablation_history_control_twostage_shift_hist.pdf"
    fig.savefig(p5, dpi=300)
    fig.savefig(p6)
    plt.close(fig)
    paths += [p5, p6]

    return paths


def main() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    df = build_joined_ablation_df()
    df.to_csv(STATS_DIR / "ablation_joined_item_level.csv", index=False)

    # Coverage and parse summary
    coverage = (
        df.groupby(["suite", "model_slug"], as_index=False)
        .agg(
            n_records=("item_id", "size"),
            parse_rate=("parsed_ok", "mean"),
            n_uai=("uai_vs_core_control", lambda s: s.notna().sum()),
        )
        .sort_values(["suite", "model_slug"])
    )
    save_df(coverage, "ablation_coverage_and_parse")

    # Condition-level summary
    cond = (
        df.groupby(["suite", "condition", "anchor_relevance"], as_index=False)
        .agg(
            n_records=("item_id", "size"),
            parse_rate=("parsed_ok", "mean"),
            mean_shift=("shift_vs_core_control", "mean"),
            mean_uai=("uai_vs_core_control", "mean"),
            median_uai=("uai_vs_core_control", "median"),
        )
        .sort_values(["suite", "condition"])
    )
    save_df(cond, "ablation_condition_summary")

    # Model-level ablation summary
    model = (
        df.groupby(["suite", "model_slug"], as_index=False)
        .agg(
            parse_rate=("parsed_ok", "mean"),
            mean_abs_shift=("shift_vs_core_control", lambda s: s.abs().mean()),
            mean_abs_uai=("uai_vs_core_control", lambda s: s.abs().mean()),
            mean_uai=("uai_vs_core_control", "mean"),
        )
        .sort_values(["suite", "model_slug"])
    )
    save_df(model, "ablation_model_summary")

    figs = make_figures(df)

    # Short analysis bullets
    ext = cond[cond["suite"] == "external"]
    ext_placebo = ext[ext["anchor_relevance"] == "placebo"]["mean_uai"].mean()
    ext_authority = ext[ext["anchor_relevance"].isin(["authority", "plausible"])]["mean_uai"].mean()
    icl_neutral = cond[(cond["suite"] == "icl") & (cond["anchor_relevance"] == "neutral")]["mean_uai"].mean()
    h_shift = cond[(cond["suite"] == "history") & (cond["condition"] == "control_twostage")]["mean_shift"].mean()

    rag = cond[cond["suite"] == "rag"].copy()
    rag_first = rag[rag["condition"].str.contains("order_first", na=False)]["mean_uai"].mean()
    rag_last = rag[rag["condition"].str.contains("order_last", na=False)]["mean_uai"].mean()
    rag_nodiscl = rag[rag["condition"].str.contains("nodiscl", na=False)]["mean_uai"].mean()
    rag_irrel_order = rag[
        (rag["anchor_relevance"] == "irrelevant")
        & (rag["condition"].str.contains("order_", na=False))
    ]["mean_uai"].mean()

    lines = []
    lines.append("# Ablation Study Results Summary")
    lines.append("")
    lines.append("## Data sources")
    lines.append("- `results/ablation_full_benchmark/*/*/results.jsonl` (ablation runs)")
    lines.append("- `results/full_benchmark/*/*/results.jsonl` (core controls for baseline)")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Suites analyzed: {', '.join(SUITES)}")
    lines.append(f"- Models per suite: {coverage.groupby('suite')['model_slug'].nunique().min()} (all complete)")
    lines.append("- Tool ablations are not included because no ablation promptviews exist for tool suites.")
    lines.append("")
    lines.append("## Key observations (descriptive)")
    lines.append(f"- External: authority/plausible-style ablations show stronger UAI than placebo on average ({ext_authority:.3f} vs {ext_placebo:.3f}).")
    lines.append(f"- ICL neutral-header ablation mean UAI: {icl_neutral:.3f}.")
    lines.append(
        f"- History control_twostage mean shift vs core control: {h_shift:.3f} "
        "(non-trivial; two-stage format can shift responses)."
    )
    lines.append(f"- RAG order effect (first vs last): {rag_first:.3f} vs {rag_last:.3f} mean UAI.")
    lines.append(
        f"- RAG disclaimer removal (nodiscl) mean UAI: {rag_nodiscl:.3f}; "
        f"irrelevant order-based baseline mean UAI: {rag_irrel_order:.3f}."
    )
    lines.append("")
    lines.append("## Generated artifacts")
    lines.append("- `outputs/stats/ablation_joined_item_level.csv`")
    lines.append("- `outputs/tables/ablation_coverage_and_parse.(csv|tex)`")
    lines.append("- `outputs/tables/ablation_condition_summary.(csv|tex)`")
    lines.append("- `outputs/tables/ablation_model_summary.(csv|tex)`")
    for p in figs:
        lines.append(f"- `{p.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Analysis is descriptive and anchored to core-control baselines from the same model/suite/item.")
    lines.append("- No missing values were imputed.")
    lines.append("- Parse failures are preserved in parse-rate reporting.")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    print("Ablation analysis complete.")
    print("Summary:", SUMMARY_MD)


if __name__ == "__main__":
    main()

