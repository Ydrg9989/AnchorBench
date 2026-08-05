#!/usr/bin/env python3
"""Gold-referenced error decomposition analysis.

Classifies each anchored response relative to control and gold:
  - harmful_shift:  |y_anchor - y_gold| > |y_ctrl - y_gold|
  - helpful_shift:  |y_anchor - y_gold| < |y_ctrl - y_gold|
  - neutral_shift:  within tolerance

Stratifies by:
  - anchor relevance (irrelevant vs plausible)
  - anchor direction relative to gold/control:
      anchor_closer_to_gold:  |anchor - gold| < |control - gold|
      anchor_farther_from_gold: |anchor - gold| > |control - gold|
  - suite, model, low/high anchor

Outputs:
  - Per-model per-suite summary table (CSV + JSON)
  - Aggregated cross-model table
  - LaTeX table
  - Stacked bar figure (PDF + PNG)

Usage:
    python -m anchorbench.analysis.gold_shift [--results_dirs ...] [--tolerance 0]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from anchorbench.eval.constants import MODEL_SHORT
from anchorbench.eval.io import load_records

ROOT = Path(__file__).resolve().parents[3]

SUITES = ["external", "history", "icl", "rag", "tool"]
ANCHORED_CONDITIONS = [
    "irrelevant_low", "irrelevant_high", "plausible_low", "plausible_high",
]


def classify_shift(y_anchor: int, y_ctrl: int, y_gold: int,
                   tolerance: float = 0) -> str:
    err_anchor = abs(y_anchor - y_gold)
    err_ctrl = abs(y_ctrl - y_gold)
    diff = err_anchor - err_ctrl
    if diff > tolerance:
        return "harmful"
    elif diff < -tolerance:
        return "helpful"
    return "neutral"


def anchor_proximity(anchor_val: float, y_ctrl: int, y_gold: int) -> str:
    dist_anchor_gold = abs(anchor_val - y_gold)
    dist_ctrl_gold = abs(y_ctrl - y_gold)
    if dist_anchor_gold < dist_ctrl_gold:
        return "anchor_closer"
    elif dist_anchor_gold > dist_ctrl_gold:
        return "anchor_farther"
    return "anchor_equidistant"


def analyze_one_run(records: list[dict], tolerance: float = 0) -> list[dict]:
    """Decompose one model×suite run into per-item shift classifications."""
    by_item: dict[str, dict[str, dict]] = {}
    for r in records:
        if not r.get("parsed_ok") or r.get("answer_int") is None:
            continue
        iid = r["item_id"]
        by_item.setdefault(iid, {})[r["condition"]] = r

    rows = []
    for iid, conds in by_item.items():
        ctrl = conds.get("control")
        if ctrl is None or ctrl.get("answer_int") is None:
            continue
        y_ctrl = ctrl["answer_int"]
        y_gold = ctrl.get("y_star_evidence")
        if y_gold is None:
            continue

        for cond_name in ANCHORED_CONDITIONS:
            rec = conds.get(cond_name)
            if rec is None or rec.get("answer_int") is None:
                continue
            y_anchor = rec["answer_int"]
            anchor_val = rec.get("anchor_value")
            if anchor_val is None:
                continue

            relevance = "irrelevant" if "irrelevant" in cond_name else "plausible"
            direction = "low" if "low" in cond_name else "high"

            shift_type = classify_shift(y_anchor, y_ctrl, y_gold, tolerance)
            prox = anchor_proximity(anchor_val, y_ctrl, y_gold)

            rows.append({
                "item_id": iid,
                "condition": cond_name,
                "relevance": relevance,
                "direction": direction,
                "y_gold": y_gold,
                "y_ctrl": y_ctrl,
                "y_anchor": y_anchor,
                "anchor_value": anchor_val,
                "err_ctrl": abs(y_ctrl - y_gold),
                "err_anchor": abs(y_anchor - y_gold),
                "shift_type": shift_type,
                "anchor_proximity": prox,
            })
    return rows


def aggregate_shares(item_rows: list[dict]) -> dict:
    """Compute harmful/helpful/neutral shares from item-level rows."""
    n = len(item_rows)
    if n == 0:
        return {"n": 0, "harmful": 0, "helpful": 0, "neutral": 0,
                "harmful_pct": 0, "helpful_pct": 0, "neutral_pct": 0,
                "mean_delta_err": 0}
    counts = defaultdict(int)
    deltas = []
    for r in item_rows:
        counts[r["shift_type"]] += 1
        deltas.append(r["err_anchor"] - r["err_ctrl"])
    return {
        "n": n,
        "harmful": counts["harmful"],
        "helpful": counts["helpful"],
        "neutral": counts["neutral"],
        "harmful_pct": round(100 * counts["harmful"] / n, 1),
        "helpful_pct": round(100 * counts["helpful"] / n, 1),
        "neutral_pct": round(100 * counts["neutral"] / n, 1),
        "mean_delta_err": round(float(np.mean(deltas)), 2),
    }


def run_analysis(results_dirs: list[Path], tolerance: float,
                 out_dir: Path, fig_dir: Path):
    all_item_rows = []
    per_model_suite = []

    for base_dir in results_dirs:
        for suite in SUITES:
            suite_dir = base_dir / suite
            if not suite_dir.is_dir():
                continue
            for model_dir in sorted(suite_dir.iterdir()):
                rpath = model_dir / "results.jsonl"
                if not rpath.exists():
                    continue
                slug = model_dir.name
                short = MODEL_SHORT.get(slug, slug)

                records = load_records(str(rpath))
                item_rows = analyze_one_run(records, tolerance)

                for r in item_rows:
                    r["model"] = short
                    r["model_slug"] = slug
                    r["suite"] = suite

                all_item_rows.extend(item_rows)

                shares = aggregate_shares(item_rows)
                shares["model"] = short
                shares["suite"] = suite

                for rel in ["irrelevant", "plausible"]:
                    sub = [r for r in item_rows if r["relevance"] == rel]
                    s = aggregate_shares(sub)
                    shares[f"{rel}_harmful_pct"] = s["harmful_pct"]
                    shares[f"{rel}_helpful_pct"] = s["helpful_pct"]
                    shares[f"{rel}_neutral_pct"] = s["neutral_pct"]
                    shares[f"{rel}_n"] = s["n"]

                for prox in ["anchor_closer", "anchor_farther", "anchor_equidistant"]:
                    sub = [r for r in item_rows if r["anchor_proximity"] == prox]
                    s = aggregate_shares(sub)
                    shares[f"{prox}_harmful_pct"] = s["harmful_pct"]
                    shares[f"{prox}_helpful_pct"] = s["helpful_pct"]
                    shares[f"{prox}_n"] = s["n"]

                per_model_suite.append(shares)

    if not all_item_rows:
        print("WARNING: No item-level data found. Check results_dirs.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-model-suite CSV ---
    csv_path = out_dir / "gold_shift_per_model_suite.csv"
    _write_csv(per_model_suite, csv_path)
    print(f"Wrote {csv_path} ({len(per_model_suite)} rows)")

    # --- Item-level CSV (for reproducibility) ---
    item_csv = out_dir / "gold_shift_item_level.csv"
    _write_csv(all_item_rows, item_csv)
    print(f"Wrote {item_csv} ({len(all_item_rows)} rows)")

    # --- Aggregated cross-model table ---
    agg_rows = []
    for suite in SUITES:
        suite_items = [r for r in all_item_rows if r["suite"] == suite]
        if not suite_items:
            continue
        row = {"suite": suite, **aggregate_shares(suite_items)}

        for rel in ["irrelevant", "plausible"]:
            sub = [r for r in suite_items if r["relevance"] == rel]
            s = aggregate_shares(sub)
            row[f"{rel}_harmful_pct"] = s["harmful_pct"]
            row[f"{rel}_helpful_pct"] = s["helpful_pct"]
            row[f"{rel}_neutral_pct"] = s["neutral_pct"]
            row[f"{rel}_n"] = s["n"]

        for prox in ["anchor_closer", "anchor_farther"]:
            sub = [r for r in suite_items if r["anchor_proximity"] == prox]
            s = aggregate_shares(sub)
            row[f"{prox}_harmful_pct"] = s["harmful_pct"]
            row[f"{prox}_helpful_pct"] = s["helpful_pct"]
            row[f"{prox}_n"] = s["n"]

        agg_rows.append(row)

    # Grand total
    grand = {"suite": "ALL", **aggregate_shares(all_item_rows)}
    for rel in ["irrelevant", "plausible"]:
        sub = [r for r in all_item_rows if r["relevance"] == rel]
        s = aggregate_shares(sub)
        grand[f"{rel}_harmful_pct"] = s["harmful_pct"]
        grand[f"{rel}_helpful_pct"] = s["helpful_pct"]
        grand[f"{rel}_neutral_pct"] = s["neutral_pct"]
        grand[f"{rel}_n"] = s["n"]
    for prox in ["anchor_closer", "anchor_farther"]:
        sub = [r for r in all_item_rows if r["anchor_proximity"] == prox]
        s = aggregate_shares(sub)
        grand[f"{prox}_harmful_pct"] = s["harmful_pct"]
        grand[f"{prox}_helpful_pct"] = s["helpful_pct"]
        grand[f"{prox}_n"] = s["n"]
    agg_rows.append(grand)

    agg_csv = out_dir / "gold_shift_aggregated.csv"
    _write_csv(agg_rows, agg_csv)
    print(f"Wrote {agg_csv}")

    # --- JSON export ---
    json_path = out_dir / "gold_shift_decomposition.json"
    with open(json_path, "w") as f:
        json.dump({
            "per_model_suite": per_model_suite,
            "aggregated": agg_rows,
            "tolerance": tolerance,
            "n_total_items": len(all_item_rows),
        }, f, indent=2)
    print(f"Wrote {json_path}")

    # --- LaTeX table ---
    latex_path = out_dir / "gold_shift_table.tex"
    _write_latex_table(agg_rows, latex_path)
    print(f"Wrote {latex_path}")

    # --- Figures ---
    _plot_stacked_bar(per_model_suite, all_item_rows, fig_dir)

    # --- README ---
    readme = out_dir / "README.md"
    readme.write_text(
        "# Gold-Referenced Error Decomposition\n\n"
        "Generated by `anchorbench.analysis.gold_shift`.\n\n"
        "## Files\n"
        "- `gold_shift_per_model_suite.csv`: per-model per-suite harmful/helpful/neutral shares\n"
        "- `gold_shift_item_level.csv`: item-level classifications\n"
        "- `gold_shift_aggregated.csv`: cross-model summary by suite\n"
        "- `gold_shift_decomposition.json`: full JSON export\n"
        "- `gold_shift_table.tex`: LaTeX table for paper\n\n"
        "## Reproduction\n"
        "```bash\n"
        "python -m anchorbench.analysis.gold_shift\n"
        "```\n"
    )
    print(f"Wrote {readme}")


def _write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_latex_table(agg_rows: list[dict], path: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Gold-referenced error decomposition by suite. "
        r"\textit{Harmful}: anchor increases error relative to control; "
        r"\textit{Helpful}: anchor decreases error; "
        r"\textit{Neutral}: no change. "
        r"Columns split by anchor proximity: closer to gold than control, "
        r"or farther.}",
        r"\label{tab:gold_shift}",
        r"\small",
        r"\begin{tabular}{l rrr rrr rr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Overall (\%)}} "
        r"& \multicolumn{3}{c}{\textbf{By Relevance (\%)}} "
        r"& \multicolumn{2}{c}{\textbf{By Proximity (\%)}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-9}",
        r"\textbf{Suite} & Harm & Help & Neut "
        r"& Irr\textsubscript{harm} & Pls\textsubscript{harm} & $\Delta$\textsubscript{err} "
        r"& Closer\textsubscript{harm} & Farther\textsubscript{harm} \\",
        r"\midrule",
    ]

    for row in agg_rows:
        suite = row["suite"]
        if suite == "ALL":
            lines.append(r"\midrule")
            suite = r"\textbf{All}"

        irr_harm = row.get("irrelevant_harmful_pct", 0)
        pls_harm = row.get("plausible_harmful_pct", 0)
        delta = row.get("mean_delta_err", 0)
        closer_harm = row.get("anchor_closer_harmful_pct", 0)
        farther_harm = row.get("anchor_farther_harmful_pct", 0)

        lines.append(
            f"{suite} & {row['harmful_pct']:.1f} & {row['helpful_pct']:.1f} "
            f"& {row['neutral_pct']:.1f} & {irr_harm:.1f} & {pls_harm:.1f} "
            f"& {delta:+.1f} & {closer_harm:.1f} & {farther_harm:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path.write_text("\n".join(lines))


def _plot_stacked_bar(per_model_suite: list[dict], all_items: list[dict],
                      fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping figures")
        return

    # --- Figure 1: Stacked bar by suite (overall) ---
    suites_order = [s for s in SUITES if any(r["suite"] == s for r in per_model_suite)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]})

    # Panel A: by suite
    ax = axes[0]
    suite_data = {}
    for suite in suites_order:
        items = [r for r in all_items if r["suite"] == suite]
        s = aggregate_shares(items)
        suite_data[suite] = s

    x = np.arange(len(suites_order))
    harmful = [suite_data[s]["harmful_pct"] for s in suites_order]
    helpful = [suite_data[s]["helpful_pct"] for s in suites_order]
    neutral = [suite_data[s]["neutral_pct"] for s in suites_order]

    bar_w = 0.55
    ax.bar(x, harmful, bar_w, label="Harmful", color="#d62728", alpha=0.85)
    ax.bar(x, neutral, bar_w, bottom=harmful, label="Neutral", color="#aec7e8", alpha=0.85)
    ax.bar(x, helpful, bar_w,
           bottom=[h + n for h, n in zip(harmful, neutral)],
           label="Helpful", color="#2ca02c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in suites_order], fontsize=10)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title("(a) Error shift by suite", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)

    # Panel B: by anchor proximity
    ax2 = axes[1]
    prox_labels = ["Anchor\ncloser", "Anchor\nfarther"]
    closer = [r for r in all_items if r["anchor_proximity"] == "anchor_closer"]
    farther = [r for r in all_items if r["anchor_proximity"] == "anchor_farther"]
    sc = aggregate_shares(closer)
    sf = aggregate_shares(farther)

    x2 = np.arange(2)
    h_vals = [sc["harmful_pct"], sf["harmful_pct"]]
    n_vals = [sc["neutral_pct"], sf["neutral_pct"]]
    hp_vals = [sc["helpful_pct"], sf["helpful_pct"]]

    ax2.bar(x2, h_vals, bar_w, label="Harmful", color="#d62728", alpha=0.85)
    ax2.bar(x2, n_vals, bar_w, bottom=h_vals, label="Neutral", color="#aec7e8", alpha=0.85)
    ax2.bar(x2, hp_vals, bar_w,
            bottom=[h + n for h, n in zip(h_vals, n_vals)],
            label="Helpful", color="#2ca02c", alpha=0.85)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(prox_labels, fontsize=10)
    ax2.set_ylabel("Share (%)", fontsize=11)
    ax2.set_title("(b) By anchor proximity to gold", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)

    for a in [ax, ax2]:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = fig_dir / f"gold_shift_decomposition_overview.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)

    # --- Figure 2: Per-model grouped dot plot (harmful share) ---
    models_seen = []
    for r in per_model_suite:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    fig2, ax3 = plt.subplots(figsize=(10, max(3, 0.35 * len(models_seen))))
    y_pos = np.arange(len(models_seen))
    irr_harm = []
    pls_harm = []
    for m in models_seen:
        model_items = [r for r in all_items if r["model"] == m]
        irr = [r for r in model_items if r["relevance"] == "irrelevant"]
        pls = [r for r in model_items if r["relevance"] == "plausible"]
        irr_harm.append(aggregate_shares(irr)["harmful_pct"])
        pls_harm.append(aggregate_shares(pls)["harmful_pct"])

    ax3.scatter(irr_harm, y_pos, marker="o", s=60, color="#1f77b4",
                label="Irrelevant", zorder=3)
    ax3.scatter(pls_harm, y_pos, marker="s", s=60, color="#d62728",
                label="Plausible", zorder=3)

    for i in range(len(models_seen)):
        ax3.plot([irr_harm[i], pls_harm[i]], [y_pos[i], y_pos[i]],
                 color="#999999", lw=1, zorder=2)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(models_seen, fontsize=9)
    ax3.set_xlabel("Harmful shift share (%)", fontsize=11)
    ax3.set_title("Harmful error share by model and relevance", fontsize=12,
                  fontweight="bold")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.invert_yaxis()

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = fig_dir / f"gold_shift_decomposition_per_model.{ext}"
        fig2.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig2)


def main():
    p = argparse.ArgumentParser(description="Gold-referenced error decomposition")
    p.add_argument("--results_dirs", nargs="+", type=Path,
                   default=[
                       Path("results/full_benchmark"),
                       Path("results/api_benchmark"),
                   ])
    p.add_argument("--out_dir", type=Path,
                   default=Path("results/revision/gold_shift_decomposition"))
    p.add_argument("--fig_dir", type=Path,
                   default=Path("figures/revision"))
    p.add_argument("--tolerance", type=float, default=0,
                   help="Tolerance for neutral classification (default: exact tie only)")
    args = p.parse_args()

    run_analysis(args.results_dirs, args.tolerance, args.out_dir, args.fig_dir)


if __name__ == "__main__":
    main()
