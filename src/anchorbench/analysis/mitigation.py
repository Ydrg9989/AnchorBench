#!/usr/bin/env python3
"""Aggregate mitigation headroom results from all models into combined table + figure.

Reads results/revision/mitigation_headroom/<suite>/<model>/<strategy>/results.jsonl
and regenerates the combined comparison table, LaTeX, interpretation, and figure.

Usage:
    python scripts/eval/aggregate_mitigation_headroom.py [--results_dir DIR] [--fig_dir DIR] [--out_dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from anchorbench.eval.constants import MODEL_SHORT
from anchorbench.eval.io import load_records
from anchorbench.eval.metrics import compute_unified_metrics

ROOT = Path(__file__).resolve().parents[3]
log = logging.getLogger(__name__)

METRIC_KEYS = [
    "uai_irr", "uai_plaus", "disc_delta", "mae_control",
    "acc10_control", "tar_irr", "tar_plaus", "parse_rate",
]

STRATEGY_ORDER = ["baseline", "ignore", "self_check", "cot"]


def discover_results(base_dir: Path) -> list[dict]:
    rows = []
    for suite_dir in sorted(base_dir.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite = suite_dir.name
        for model_dir in sorted(suite_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            slug = model_dir.name
            short = MODEL_SHORT.get(slug, slug)
            for strat_dir in sorted(model_dir.iterdir()):
                rpath = strat_dir / "results.jsonl"
                if not rpath.exists():
                    continue
                records = load_records(str(rpath))
                m = compute_unified_metrics(records)
                row = {"suite": suite, "model": short, "model_slug": slug,
                       "strategy": strat_dir.name}
                for k in METRIC_KEYS:
                    row[k] = m.get(k)
                rows.append(row)
    return rows


def _fmt(v, prec=3):
    if v is None:
        return "---"
    return f"{v:.{prec}f}"


def _fmt_delta(v, prec=3):
    if v is None:
        return "---"
    return f"{v:+.{prec}f}"


def write_outputs(rows: list[dict], out_dir: Path, fig_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "mitigation_headroom_comparison.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    # JSON
    json_path = out_dir / "mitigation_headroom_comparison.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {json_path}")

    # LaTeX
    latex_path = out_dir / "mitigation_headroom_table.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mitigation headroom probe: effect of prompt-based debiasing "
        r"strategies on anchoring susceptibility (External and RAG suites). "
        r"\textit{Ignore}: instruction to disregard extraneous numbers. "
        r"\textit{Self-check}: post-hoc verification. "
        r"\textit{CoT}: step-by-step reasoning.}",
        r"\label{tab:mitigation_headroom}",
        r"\small",
        r"\begin{tabular}{ll l rrrr}",
        r"\toprule",
        r"\textbf{Suite} & \textbf{Model} & \textbf{Strategy} "
        r"& \textbf{Disc}$_\Delta$ & \textbf{UAI}\textsubscript{pls} "
        r"& \textbf{MAE}\textsubscript{ctrl} & \textbf{Parse} \\",
        r"\midrule",
    ]

    suites = sorted(set(r["suite"] for r in rows))
    models = sorted(set(r["model"] for r in rows))

    prev_suite = ""
    for suite in suites:
        for model in models:
            for strat in STRATEGY_ORDER:
                r = [x for x in rows if x["suite"] == suite
                     and x["model"] == model and x["strategy"] == strat]
                if not r:
                    continue
                r = r[0]
                suite_label = suite if suite != prev_suite else ""
                prev_suite = suite
                lines.append(
                    f"{suite_label} & {model} & {strat} "
                    f"& {_fmt(r.get('disc_delta'),3)} & {_fmt(r.get('uai_plaus'),3)} "
                    f"& {_fmt(r.get('mae_control'),1)} & {_fmt(r.get('parse_rate'),3)} \\\\"
                )
        if suite != suites[-1]:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    latex_path.write_text("\n".join(lines))
    print(f"Wrote {latex_path}")

    # Interpretation
    interp_path = out_dir / "mitigation_headroom_interpretation.md"
    ilines = ["# Mitigation Headroom Probe — Interpretation\n\n## Summary\n"]

    strategies = [s for s in STRATEGY_ORDER if any(r["strategy"] == s for r in rows)]

    for suite in suites:
        ilines.append(f"\n### {suite.capitalize()}\n")
        for model in models:
            ilines.append(f"**{model}:**\n")
            base = [r for r in rows if r["suite"] == suite
                    and r["model"] == model and r["strategy"] == "baseline"]
            if not base:
                ilines.append("- No baseline data\n")
                continue
            b = base[0]
            ilines.append(f"- Baseline: Disc={_fmt(b.get('disc_delta'),3)}, "
                          f"MAE={_fmt(b.get('mae_control'),1)}, "
                          f"Parse={_fmt(b.get('parse_rate'),3)}\n")

            for strat in strategies:
                if strat == "baseline":
                    continue
                s = [r for r in rows if r["suite"] == suite
                     and r["model"] == model and r["strategy"] == strat]
                if not s:
                    continue
                s = s[0]
                disc_b = b.get("disc_delta") or 0
                disc_s = s.get("disc_delta") or 0
                mae_b = b.get("mae_control") or 0
                mae_s = s.get("mae_control") or 0
                disc_d = disc_s - disc_b
                mae_d = mae_s - mae_b

                effect = "reduces anchoring" if disc_d < -0.01 else \
                         "increases anchoring" if disc_d > 0.01 else "negligible change"
                acc_eff = "hurts accuracy" if mae_d > 0.5 else \
                          "improves accuracy" if mae_d < -0.5 else "neutral on accuracy"

                ilines.append(f"- {strat}: Disc={_fmt(s.get('disc_delta'),3)} "
                              f"(Δ={disc_d:+.3f}), "
                              f"MAE={_fmt(s.get('mae_control'),1)} "
                              f"(Δ={mae_d:+.1f}) → **{effect}**, {acc_eff}\n")

    ilines.append("\n## Key Takeaways\n")
    total_disc = {}
    for strat in strategies:
        if strat == "baseline":
            continue
        deltas = []
        for suite in suites:
            for model in models:
                base = [r for r in rows if r["suite"] == suite
                        and r["model"] == model and r["strategy"] == "baseline"]
                curr = [r for r in rows if r["suite"] == suite
                        and r["model"] == model and r["strategy"] == strat]
                if base and curr:
                    deltas.append((curr[0].get("disc_delta") or 0) -
                                  (base[0].get("disc_delta") or 0))
        if deltas:
            total_disc[strat] = float(np.mean(deltas))

    for strat, md in sorted(total_disc.items(), key=lambda x: x[1]):
        direction = "reduces" if md < 0 else "increases"
        ilines.append(f"- **{strat}**: mean Disc$_\\Delta$ change = {md:+.3f} "
                      f"({direction} anchoring on average)\n")

    interp_path.write_text("\n".join(ilines))
    print(f"Wrote {interp_path}")

    # Figure
    _make_figure(rows, fig_dir)


def _make_figure(rows: list[dict], fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available")
        return

    strategies = [s for s in STRATEGY_ORDER if any(r["strategy"] == s for r in rows)]
    models = sorted(set(r["model"] for r in rows))
    suites = sorted(set(r["suite"] for r in rows))

    n_groups = len(suites) * len(models)
    fig, ax = plt.subplots(figsize=(max(8, n_groups * 2.2), 4.5))

    colors = {
        "baseline": "#7f7f7f",
        "ignore": "#1f77b4",
        "self_check": "#2ca02c",
        "cot": "#ff7f0e",
    }
    bar_w = 0.8 / len(strategies)
    x_positions = []
    x_labels = []

    gi = 0
    for _si, suite in enumerate(suites):
        for _mi, model in enumerate(models):
            x_labels.append(f"{suite[:3].upper()}\n{model}")
            x_positions.append(gi)
            for sti, strat in enumerate(strategies):
                r = [x for x in rows if x["suite"] == suite
                     and x["model"] == model and x["strategy"] == strat]
                if not r:
                    continue
                v = r[0].get("disc_delta")
                if v is None:
                    continue
                x = gi + (sti - len(strategies) / 2 + 0.5) * bar_w
                ax.bar(x, v, bar_w * 0.9, color=colors.get(strat, "#333"),
                       label=strat if gi == 0 else "", alpha=0.85)
            gi += 1
        gi += 0.5  # gap between suites

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel(r"Disc$_\Delta$", fontsize=11)
    ax.set_title("Mitigation Headroom: Disc$_\\Delta$ by Strategy",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = fig_dir / f"mitigation_headroom_disc.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_dir",
        default="results/revision/mitigation_headroom",
        help="Directory containing suite/model/strategy/results.jsonl trees",
    )
    p.add_argument(
        "--fig_dir",
        default="figures/revision",
        help="Directory for mitigation headroom figures (PDF/PNG)",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Directory for CSV, JSON, LaTeX, interpretation (default: same as --results_dir)",
    )
    args = p.parse_args()
    base = ROOT / args.results_dir
    out_dir = ROOT / args.out_dir if args.out_dir is not None else base
    fig_dir = ROOT / args.fig_dir

    rows = discover_results(base)
    if not rows:
        print("ERROR: No mitigation headroom results found.")
        return

    write_outputs(rows, out_dir, fig_dir)


if __name__ == "__main__":
    main()
