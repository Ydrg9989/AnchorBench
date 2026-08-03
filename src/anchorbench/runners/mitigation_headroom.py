#!/usr/bin/env python3
"""Mitigation headroom probe: compare multiple prompt-based mitigation strategies.

Extends the existing ignore-anchor baseline with additional strategies:
  - ignore:     "Disregard any other numeric values..." (existing)
  - self_check: "After producing your estimate, briefly verify it wasn't
                 influenced by extraneous numbers, then give your final answer."
  - cot:        "Think step by step. Show your reasoning, then provide your
                 final answer on the last line."

Runs a targeted subset: External and RAG suites, 1-2 models.
Results saved to results/revision/mitigation_headroom/.

Usage (vLLM):
    bash scripts/run_with_env.sh python scripts/eval/run_mitigation_headroom.py \
        --model_id Qwen/Qwen2.5-7B-Instruct --backend vllm

    bash scripts/run_with_env.sh python scripts/eval/run_mitigation_headroom.py \
        --model_id meta-llama/Llama-3.1-8B-Instruct --backend vllm
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorbench.eval.evaluator import (
    CONDITIONS,
    build_record,
    parse_response,
    prepare_items,
    run_single_stage,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.constants import MODEL_SHORT, SUITE_DATASETS as _ALL
from anchorbench.eval.metrics import compute_unified_metrics

log = logging.getLogger(__name__)

SUITE_DATASETS = {k: v for k, v in _ALL.items() if k in ("external", "rag")}

MITIGATION_STRATEGIES = {
    "baseline": "",
    "ignore": (
        "\n\nImportant: Base your final numeric estimate solely on the "
        "evidence provided in the task data. Disregard any other numeric "
        "values from the surrounding context that are not part of the "
        "core evidence."
    ),
    "self_check": (
        "\n\nAfter producing your initial estimate, briefly check whether "
        "any extraneous numbers in the context may have biased your answer. "
        "If so, correct it. Then state your final numeric estimate."
    ),
    "cot": (
        "\n\nThink step by step. List the relevant evidence, compute "
        "your estimate from that evidence only, then provide your final "
        "numeric answer on the last line."
    ),
}

METRIC_KEYS = [
    "uai_irr", "uai_plaus", "disc_delta", "mae_control",
    "acc10_control", "tar_irr", "tar_plaus", "parse_rate",
]


def run_strategy(backend, suite: str, items: list[dict], out_dir: Path,
                 strategy_name: str, prompt_suffix: str,
                 max_tokens: int, batch_size: int,
                 force: bool = False) -> list[dict]:
    model_slug = backend.model_id.replace("/", "_")
    model_out = out_dir / suite / model_slug / strategy_name
    model_out.mkdir(parents=True, exist_ok=True)
    results_path = model_out / "results.jsonl"

    existing = model_out / "summary.json"
    if existing.exists() and not force:
        log.info("[%s/%s/%s] Already exists, loading. Use --force to re-run.", suite, model_slug, strategy_name)
        from anchorbench.eval.io import load_records
        return load_records(str(results_path))

    records = run_single_stage(
        backend, items, results_path,
        max_tokens=max_tokens, batch_size=batch_size,
        prompt_suffix=prompt_suffix,
    )
    write_and_summarize(
        records, model_out,
        label=f"{suite} ({strategy_name}) | {backend.model_id}",
    )
    return records


def compute_comparison_table(all_results: dict, out_dir: Path):
    """Build a comparison table across strategies."""
    rows = []
    for (suite, model_slug, strategy), records in sorted(all_results.items()):
        if not records:
            continue
        m = compute_unified_metrics(records)
        short = MODEL_SHORT.get(model_slug, model_slug)
        row = {"suite": suite, "model": short, "strategy": strategy}
        for k in METRIC_KEYS:
            row[k] = m.get(k)
        rows.append(row)

    csv_path = out_dir / "mitigation_headroom_comparison.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        log.info("Wrote %s (%d rows)", csv_path, len(rows))

    json_path = out_dir / "mitigation_headroom_comparison.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    log.info("Wrote %s", json_path)

    # LaTeX
    latex_path = out_dir / "mitigation_headroom_table.tex"
    _write_latex(rows, latex_path)

    # Interpretation note
    _write_interpretation(rows, out_dir / "mitigation_headroom_interpretation.md")

    return rows


def _write_latex(rows: list[dict], path: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mitigation headroom: effect of prompt-based debiasing strategies "
        r"on anchoring susceptibility. \textit{Baseline}: no mitigation. "
        r"\textit{Ignore}: instruction to disregard extraneous numbers. "
        r"\textit{Self-check}: post-hoc verification instruction. "
        r"\textit{CoT}: step-by-step reasoning instruction.}",
        r"\label{tab:mitigation_headroom}",
        r"\small",
        r"\begin{tabular}{ll l rrrr}",
        r"\toprule",
        r"\textbf{Suite} & \textbf{Model} & \textbf{Strategy} "
        r"& \textbf{Disc}$_\Delta$ & \textbf{UAI}\textsubscript{pls} "
        r"& \textbf{MAE}\textsubscript{ctrl} & \textbf{Parse} \\",
        r"\midrule",
    ]

    prev_suite = ""
    for row in rows:
        suite = row["suite"] if row["suite"] != prev_suite else ""
        prev_suite = row["suite"]
        model = row.get("model", "")
        strat = row.get("strategy", "")
        disc = _fmt(row.get("disc_delta"), 3)
        uai_p = _fmt(row.get("uai_plaus"), 3)
        mae = _fmt(row.get("mae_control"), 1)
        pr = _fmt(row.get("parse_rate"), 3)
        lines.append(f"{suite} & {model} & {strat} & {disc} & {uai_p} & {mae} & {pr} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines))
    log.info("Wrote %s", path)


def _fmt(v, prec=3):
    if v is None:
        return "---"
    return f"{v:.{prec}f}"


def _write_interpretation(rows: list[dict], path: Path):
    lines = ["# Mitigation Headroom Probe — Interpretation\n"]
    lines.append("## Summary\n")

    strategies = sorted(set(r["strategy"] for r in rows))
    suites = sorted(set(r["suite"] for r in rows))
    models = sorted(set(r["model"] for r in rows))

    for suite in suites:
        lines.append(f"\n### {suite.capitalize()}\n")
        for model in models:
            lines.append(f"**{model}:**\n")
            base = [r for r in rows if r["suite"] == suite
                    and r["model"] == model and r["strategy"] == "baseline"]
            if not base:
                lines.append("- No baseline data\n")
                continue
            b = base[0]
            lines.append(f"- Baseline: Disc={_fmt(b.get('disc_delta'),3)}, "
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
                disc_change = disc_s - disc_b
                mae_change = mae_s - mae_b

                effect = "reduces anchoring" if disc_change < -0.01 else \
                         "increases anchoring" if disc_change > 0.01 else "negligible change"
                acc_effect = "hurts accuracy" if mae_change > 0.5 else \
                             "improves accuracy" if mae_change < -0.5 else "neutral on accuracy"

                lines.append(f"- {strat}: Disc={_fmt(s.get('disc_delta'),3)} "
                             f"(Δ={disc_change:+.3f}), "
                             f"MAE={_fmt(s.get('mae_control'),1)} "
                             f"(Δ={mae_change:+.1f}) → **{effect}**, {acc_effect}\n")

    lines.append("\n## Key Takeaways\n")
    lines.append("*(Auto-generated; verify against actual values above.)*\n")

    total_disc_deltas = {}
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
                    b_disc = base[0].get("disc_delta") or 0
                    c_disc = curr[0].get("disc_delta") or 0
                    deltas.append(c_disc - b_disc)
        if deltas:
            total_disc_deltas[strat] = float(np.mean(deltas))

    for strat, mean_d in sorted(total_disc_deltas.items(), key=lambda x: x[1]):
        direction = "reduces" if mean_d < 0 else "increases"
        lines.append(f"- **{strat}**: mean Disc delta change = {mean_d:+.3f} "
                     f"({direction} anchoring on average)\n")

    path.write_text("\n".join(lines))
    log.info("Wrote %s", path)


def make_figure(rows: list[dict], fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping figure")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    strategies = ["baseline", "ignore", "self_check", "cot"]
    strat_present = [s for s in strategies if any(r["strategy"] == s for r in rows)]
    models = sorted(set(r["model"] for r in rows))
    suites = sorted(set(r["suite"] for r in rows))

    n_groups = len(suites) * len(models)
    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.8), 4.5))

    colors = {
        "baseline": "#7f7f7f",
        "ignore": "#1f77b4",
        "self_check": "#2ca02c",
        "cot": "#ff7f0e",
    }
    bar_w = 0.8 / len(strat_present)

    group_labels = []
    for gi, (suite, model) in enumerate([(s, m) for s in suites for m in models]):
        group_labels.append(f"{suite[:3].upper()}\n{model}")
        for si, strat in enumerate(strat_present):
            r = [x for x in rows if x["suite"] == suite
                 and x["model"] == model and x["strategy"] == strat]
            if not r:
                continue
            v = r[0].get("disc_delta")
            if v is None:
                continue
            x = gi + (si - len(strat_present) / 2 + 0.5) * bar_w
            ax.bar(x, v, bar_w * 0.9, color=colors.get(strat, "#333"),
                   label=strat if gi == 0 else "", alpha=0.85)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_labels, fontsize=9)
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
        log.info("Wrote %s", p)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Mitigation headroom probe")
    p.add_argument("--model_id", required=True)
    p.add_argument("--suites", nargs="+", default=list(SUITE_DATASETS.keys()),
                   choices=list(SUITE_DATASETS.keys()))
    p.add_argument("--strategies", nargs="+",
                   default=list(MITIGATION_STRATEGIES.keys()),
                   choices=list(MITIGATION_STRATEGIES.keys()))
    p.add_argument("--out_dir", type=Path,
                   default=Path("results/revision/mitigation_headroom"))
    p.add_argument("--fig_dir", type=Path,
                   default=Path("figures/revision"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if summary.json already exists")
    args = p.parse_args()

    from anchorbench.eval.backends import HFBackend, VLLMBackend

    log.info("Loading model %s (%s)...", args.model_id, args.backend)
    if args.backend == "vllm":
        backend = VLLMBackend(
            args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype="bfloat16", trust_remote_code=True,
        )
    else:
        backend = HFBackend(args.model_id, device="auto", dtype="bfloat16")

    all_results = {}

    for suite in args.suites:
        ds = Path(SUITE_DATASETS[suite])
        pv_file = ds / "promptviews_core.jsonl"
        if not pv_file.exists():
            pv_file = ds / "promptviews.jsonl"
        spec_file = ds / "itemspecs.jsonl"

        views = load_promptviews(pv_file)
        specs = load_itemspecs(spec_file)
        items = prepare_items(views, specs, args.max_items, args.seed)
        log.info("[%s] %d items loaded", suite, len(items))

        model_slug = backend.model_id.replace("/", "_")

        for strategy_name in args.strategies:
            suffix = MITIGATION_STRATEGIES[strategy_name]
            records = run_strategy(
                backend, suite, items, args.out_dir,
                strategy_name, suffix,
                args.max_tokens, args.batch_size,
                force=args.force,
            )
            all_results[(suite, model_slug, strategy_name)] = records

    rows = compute_comparison_table(all_results, args.out_dir)
    make_figure(rows, args.fig_dir)

    # README
    readme = args.out_dir / "README.md"
    readme.write_text(
        "# Mitigation Headroom Probe\n\n"
        "Compares prompt-based debiasing strategies for anchoring mitigation.\n\n"
        "## Strategies\n"
        "- **baseline**: no prompt modification\n"
        "- **ignore**: \"Disregard extraneous numeric values\"\n"
        "- **self_check**: \"Verify your answer wasn't biased by irrelevant numbers\"\n"
        "- **cot**: \"Think step by step, list evidence, compute from evidence only\"\n\n"
        "## Reproduction\n"
        "```bash\n"
        "bash scripts/run_with_env.sh python scripts/eval/run_mitigation_headroom.py \\\n"
        "    --model_id Qwen/Qwen2.5-7B-Instruct --backend vllm\n"
        "bash scripts/run_with_env.sh python scripts/eval/run_mitigation_headroom.py \\\n"
        "    --model_id meta-llama/Llama-3.1-8B-Instruct --backend vllm\n"
        "```\n"
    )
    log.info("Wrote %s", readme)


if __name__ == "__main__":
    main()
