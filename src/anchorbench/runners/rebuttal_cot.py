#!/usr/bin/env python3
"""CoT (reasoning-allowed) extension runner for the COLM 2026 rebuttal.

Addresses reviewer Cluster F (REVIEWER #1): does anchoring persist when the model
is allowed to reason before answering, rather than being forced to output a
single integer?

Extends the existing ``mitigation_headroom.py`` runner from
{External, RAG} x {Qwen-7B, Llama-8B} to {External, RAG, History} x the
locked five-model panel, using *the same* CoT prompt suffix so the new
results are directly comparable to the existing two-model CoT data in
``results/revision/mitigation_headroom/``.

The CoT prompt suffix matches
``mitigation_headroom.MITIGATION_STRATEGIES["cot"]`` verbatim::

    "\\n\\nThink step by step. List the relevant evidence, compute "
    "your estimate from that evidence only, then provide your final "
    "numeric answer on the last line."

For History the same suffix is appended to BOTH stages of the two-stage
protocol (control single-stage, anchored stage 1 elicitation, anchored
stage 2 final estimate).

Outputs are written to
``results/rebuttal/cot_extended/<suite>/<model_slug>/{baseline,cot}/``
so the existing analysis script (``anchorbench.analysis.cot_reasoning``)
can pick them up without modification.

Usage (open-weight, vLLM)::

    bash scripts/run_with_env.sh \\
      PYTHONPATH=src python -m anchorbench.runners.rebuttal_cot \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --backend vllm --suites external rag history

Usage (also baseline so you have a paired comparison)::

    --strategies baseline cot
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from anchorbench.eval.constants import SUITE_DATASETS as _ALL_SUITE_DATASETS
from anchorbench.eval.evaluator import (
    CONDITIONS,
    prepare_items,
    run_history_two_stage,
    run_single_stage,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.metrics import compute_unified_metrics

log = logging.getLogger(__name__)

REBUTTAL_STRATEGIES: dict[str, str] = {
    "baseline": "",
    "cot": (
        "\n\nThink step by step. List the relevant evidence, compute "
        "your estimate from that evidence only, then provide your final "
        "numeric answer on the last line."
    ),
    # P4 task-specification ablation (closes REVIEWER-2).
    "rule": (
        "\n\nYour estimate should be the unweighted arithmetic mean of "
        "the visible ratings, rounded to the nearest integer."
    ),
    "judgment": (
        "\n\nYour estimate should be a weighted average of the available "
        "evidence, weighting each piece according to which sources you "
        "find more or less credible."
    ),
}

ALLOWED_SUITES = ("external", "history", "icl", "rag", "tool")

HISTORY_CONDITIONS_TWOSTAGE = [
    "control_twostage",
    "irrelevant_low",
    "irrelevant_high",
    "plausible_low",
    "plausible_high",
]


def _suite_files(suite: str) -> tuple[Path, Path]:
    ds = ROOT / _ALL_SUITE_DATASETS[suite]
    # History needs the two-stage control_twostage condition which only ships
    # in promptviews.jsonl (not promptviews_core.jsonl); all other suites use
    # the core file.
    if suite == "history":
        pv = ds / "promptviews.jsonl"
        if not pv.exists():
            pv = ds / "promptviews_core.jsonl"
    else:
        pv = ds / "promptviews_core.jsonl"
        if not pv.exists():
            pv = ds / "promptviews.jsonl"
    return pv, ds / "itemspecs.jsonl"


def _run_one(
    backend, suite: str, items: list[dict], strategy_name: str,
    suffix: str, out_dir: Path,
    max_tokens: int, batch_size: int,
    history_baseline: str = "control_twostage",
) -> list[dict]:
    model_slug = backend.model_id.replace("/", "_")
    cell_dir = out_dir / suite / model_slug / strategy_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    results_path = cell_dir / "results.jsonl"

    if (cell_dir / "summary.json").exists():
        log.info("[%s/%s/%s] already exists, loading.",
                 suite, model_slug, strategy_name)
        from anchorbench.eval.io import load_records
        return load_records(str(results_path))

    if suite == "history":
        records = run_history_two_stage(
            backend, items, results_path,
            conditions=HISTORY_CONDITIONS_TWOSTAGE,
            max_tokens=max_tokens,
            prompt_suffix=suffix,
        )
        write_and_summarize(
            records, cell_dir,
            label=f"history ({strategy_name}) | {backend.model_id}",
            baseline_condition=history_baseline,
        )
    else:
        records = run_single_stage(
            backend, items, results_path,
            max_tokens=max_tokens,
            batch_size=batch_size,
            prompt_suffix=suffix,
        )
        write_and_summarize(
            records, cell_dir,
            label=f"{suite} ({strategy_name}) | {backend.model_id}",
        )
    return records


def _comparison_table(all_results: dict, out_dir: Path) -> list[dict]:
    """Build comparison table across (suite, model, strategy) cells."""
    import csv

    from anchorbench.eval.constants import MODEL_SHORT

    metric_keys = (
        "uai_irr", "uai_plaus", "disc_delta",
        "mae_control", "acc10_control",
        "tar_irr", "tar_plaus", "parse_rate",
    )

    rows = []
    for (suite, model_slug, strategy), records in sorted(all_results.items()):
        if not records:
            continue
        baseline = (
            "control_twostage" if suite == "history" else "control"
        )
        m = compute_unified_metrics(records, baseline_condition=baseline)
        row = {
            "suite": suite,
            "model": MODEL_SHORT.get(model_slug, model_slug),
            "model_slug": model_slug,
            "strategy": strategy,
        }
        for k in metric_keys:
            row[k] = m.get(k)
        rows.append(row)

    csv_path = out_dir / "rebuttal_cot_comparison.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log.info("Wrote %s (%d rows)", csv_path, len(rows))

    json_path = out_dir / "rebuttal_cot_comparison.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    log.info("Wrote %s", json_path)
    return rows


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Rebuttal CoT runner (B1)")
    p.add_argument("--model_id", required=True)
    p.add_argument(
        "--suites", nargs="+", default=["external", "rag", "history"],
        choices=list(ALLOWED_SUITES),
    )
    p.add_argument(
        "--strategies", nargs="+", default=["baseline", "cot"],
        choices=list(REBUTTAL_STRATEGIES.keys()),
    )
    p.add_argument("--out_dir", type=Path,
                   default=Path("results/rebuttal/cot_extended"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    args = p.parse_args(argv)

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

    all_results: dict[tuple[str, str, str], list[dict]] = {}
    for suite in args.suites:
        pv_file, spec_file = _suite_files(suite)
        if not pv_file.exists():
            log.error("Missing dataset for %s: %s", suite, pv_file)
            continue

        views = load_promptviews(pv_file)
        specs = load_itemspecs(spec_file)
        conds = (
            HISTORY_CONDITIONS_TWOSTAGE if suite == "history" else CONDITIONS
        )
        items = prepare_items(views, specs, args.max_items, args.seed,
                              conditions=conds)
        log.info("[%s] %d items loaded", suite, len(items))

        model_slug = backend.model_id.replace("/", "_")
        for strat in args.strategies:
            suffix = REBUTTAL_STRATEGIES[strat]
            recs = _run_one(
                backend, suite, items, strat, suffix, args.out_dir,
                args.max_tokens, args.batch_size,
            )
            all_results[(suite, model_slug, strat)] = recs

    rows = _comparison_table(all_results, args.out_dir)
    log.info("Done. %d cells in comparison table.", len(rows))


if __name__ == "__main__":
    main()
