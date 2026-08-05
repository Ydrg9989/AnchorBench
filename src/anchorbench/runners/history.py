#!/usr/bin/env python3
"""Run inference on History dataset (two-stage) and evaluate.

Default: five conditions with ``control_twostage`` as the baseline (matched two-turn
format vs plausible/irrelevant). Use ``--baseline_condition control`` for the
legacy single-stage control only.

Usage:
    PYTHONPATH=src python scripts/eval/run_history.py \
        --promptviews datasets/anchorbench_history_core/promptviews.jsonl \
        --itemspecs datasets/anchorbench_history_core/itemspecs.jsonl \
        --model_id Qwen/Qwen2.5-7B-Instruct \
        --out_dir results/history_core
"""

from __future__ import annotations

import argparse
import logging
import sys

from anchorbench.eval.evaluator import (
    CONDITIONS as EVAL_CONDITIONS,
)
from anchorbench.eval.evaluator import (
    prepare_items,
    run_history_two_stage,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.runner_utils import (
    add_common_args,
    build_suffix,
    make_backend,
    make_fallback,
    model_output_dir,
)

# The matched-control analysis compares the two baselines against each other,
# so a run has to carry BOTH: `control` is single-stage (the protocol used in
# the main benchmark) and `control_twostage` is the format-matched two-turn
# version. paper.tables_appendix.build_history_matched computes metrics twice
# from one results.jsonl, once per baseline, and skips any model missing
# either. Omitting `control` here made tab:history_matched unbuildable.
HISTORY_CONDITIONS_TWOSTAGE_BASELINE = [
    "control",
    "control_twostage",
    "irrelevant_low",
    "irrelevant_high",
    "plausible_low",
    "plausible_high",
]

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Run History inference (two-stage) and evaluation",
    )
    add_common_args(p)
    p.add_argument(
        "--baseline_condition",
        choices=["control", "control_twostage"],
        default="control",
        help="Reference condition for MAE/UAI (default: control).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing results.jsonl and skip (item_id, condition) pairs already present.",
    )
    args = p.parse_args()

    if args.baseline_condition == "control_twostage":
        hist_conditions = HISTORY_CONDITIONS_TWOSTAGE_BASELINE
    else:
        hist_conditions = list(EVAL_CONDITIONS)

    model_out = model_output_dir(args)

    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items(
        views, specs, args.max_items, args.seed, conditions=hist_conditions,
    )
    log.info(
        "Loaded %d items × %d conditions (baseline=%s)",
        len(items), len(hist_conditions), args.baseline_condition,
    )

    if not items:
        log.error(
            "No History items found (need all conditions in %s)",
            hist_conditions,
        )
        sys.exit(1)

    fallback = make_fallback(args)
    suffix = build_suffix(args)
    backend = make_backend(args)

    results_path = model_out / "results.jsonl"
    records = run_history_two_stage(
        backend, items, results_path,
        conditions=hist_conditions,
        max_tokens=args.max_tokens,
        prompt_suffix=suffix,
        use_llm_fallback=args.llm_fallback,
        fallback_extractor=fallback,
        resume=args.resume,
    )

    write_and_summarize(
        records,
        model_out,
        label=f"History | {args.model_id}",
        baseline_condition=args.baseline_condition,
    )


if __name__ == "__main__":
    main()
