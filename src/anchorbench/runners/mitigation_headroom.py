#!/usr/bin/env python3
"""Prompt-suffix probes: mitigation headroom, the CoT extension and the
task-specification ablation, one runner.

Every probe appends one instruction from ``runner_utils.PROMPT_SUFFIXES`` to
every prompt of a suite and scores the run like the benchmark; the probes
differ only in which suites, strategies and output directory they use:

    # Appendix Table 19, mitigation headroom (External, RAG; Qwen-7B, Llama-8B)
    python -m anchorbench.runners.mitigation_headroom --model_id Qwen/Qwen2.5-7B-Instruct \\
        --suites external rag --strategies baseline ignore self_check cot

    # tab:cot_extended: reasoning allowed, five models, History included
    python -m anchorbench.runners.mitigation_headroom --model_id Qwen/Qwen2.5-7B-Instruct \\
        --suites external rag history --strategies baseline cot \\
        --out_dir results/rebuttal/cot_extended --max_tokens 768

    # tab:task_spec: rule vs judgment task specification
    python -m anchorbench.runners.mitigation_headroom --model_id Qwen/Qwen2.5-7B-Instruct \\
        --suites external --strategies baseline rule judgment --out_dir results/rebuttal/task_spec

Each (suite, model, strategy) cell writes results.jsonl and summary.json under
<out_dir>/<suite>/<slug>/<strategy>/; History runs the two-stage protocol with
the suffix on both stages and is scored against control_twostage. The
comparison table, figure and interpretation come from
``python -m anchorbench.analysis.mitigation`` over the finished tree.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anchorbench.eval.constants import SUITE_DATASETS
from anchorbench.eval.evaluator import (
    prepare_items,
    run_history_two_stage,
    run_single_stage,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews, load_records, suite_files
from anchorbench.eval.runner_utils import PROMPT_SUFFIXES, build_backend
from anchorbench.paths import RESULTS_DIR, ROOT

log = logging.getLogger(__name__)

HISTORY_BASELINE = "control_twostage"
HISTORY_CONDITIONS = [HISTORY_BASELINE, "irrelevant_low", "irrelevant_high",
                      "plausible_low", "plausible_high"]


def run_strategy(backend, suite: str, items: list[dict], out_dir: Path,
                 strategy: str, max_tokens: int, batch_size: int,
                 force: bool = False) -> list[dict]:
    """One (suite, model, strategy) cell; a finished cell is loaded, not re-run."""
    model_slug = backend.model_id.replace("/", "_")
    cell_dir = out_dir / suite / model_slug / strategy
    cell_dir.mkdir(parents=True, exist_ok=True)
    results_path = cell_dir / "results.jsonl"
    if (cell_dir / "summary.json").exists() and not force:
        log.info("[%s/%s/%s] already exists, loading (--force re-runs).", suite, model_slug, strategy)
        return load_records(results_path)

    suffix = PROMPT_SUFFIXES[strategy]
    label = f"{suite} ({strategy}) | {backend.model_id}"
    if suite == "history":
        records = run_history_two_stage(
            backend, items, results_path, conditions=HISTORY_CONDITIONS,
            max_tokens=max_tokens, prompt_suffix=suffix,
        )
        write_and_summarize(records, cell_dir, label=label, baseline_condition=HISTORY_BASELINE)
    else:
        records = run_single_stage(
            backend, items, results_path,
            max_tokens=max_tokens, batch_size=batch_size, prompt_suffix=suffix,
        )
        write_and_summarize(records, cell_dir, label=label)
    return records


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Prompt-suffix probes (mitigation, CoT, task spec)")
    p.add_argument("--model_id", required=True)
    p.add_argument("--suites", nargs="+", default=["external", "rag"],
                   choices=sorted(SUITE_DATASETS))
    p.add_argument("--strategies", nargs="+", default=list(PROMPT_SUFFIXES),
                   choices=list(PROMPT_SUFFIXES))
    p.add_argument("--out_dir", type=Path, default=RESULTS_DIR / "revision/mitigation_headroom")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--force", action="store_true",
                   help="Re-run cells whose summary.json already exists")
    args = p.parse_args(argv)

    backend = build_backend(
        args.backend, args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    for suite in args.suites:
        pv_file, spec_file = suite_files(ROOT / SUITE_DATASETS[suite], prefer_full=suite == "history")
        conditions = HISTORY_CONDITIONS if suite == "history" else None
        items = prepare_items(load_promptviews(pv_file), load_itemspecs(spec_file),
                              args.max_items, args.seed, conditions=conditions)
        log.info("[%s] %d items loaded", suite, len(items))
        for strategy in args.strategies:
            run_strategy(backend, suite, items, args.out_dir, strategy,
                         args.max_tokens, args.batch_size, force=args.force)
    log.info("Done. Table, figure and interpretation: python -m anchorbench.analysis.mitigation "
             "--results_dir %s", args.out_dir)


if __name__ == "__main__":
    main()
