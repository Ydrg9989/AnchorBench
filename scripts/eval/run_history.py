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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench_eval.backends import HFBackend, VLLMBackend
from anchorbench_eval.evaluator import (
    CONDITIONS as EVAL_CONDITIONS,
    prepare_items,
    run_history_two_stage,
    write_and_summarize,
)

HISTORY_CONDITIONS_TWOSTAGE_BASELINE = [
    "control_twostage",
    "irrelevant_low",
    "irrelevant_high",
    "plausible_low",
    "plausible_high",
]
from anchorbench_eval.io import load_itemspecs, load_promptviews
from anchorbench_eval.parsing import LLMFallbackExtractor

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Run History inference (two-stage) and evaluation",
    )
    p.add_argument("--promptviews", type=Path, required=True)
    p.add_argument("--itemspecs", type=Path, required=True)
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out_dir", type=Path, default=Path("results/history"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--request_final_line", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--llm_fallback", action="store_true")
    p.add_argument("--fallback_model", type=str, default=None)
    p.add_argument("--fallback_device", type=str, default="auto")
    p.add_argument("--structured", action="store_true")
    p.add_argument("--backend", type=str, choices=["hf", "vllm"], default="hf")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument(
        "--baseline_condition",
        choices=["control", "control_twostage"],
        default="control_twostage",
        help="Reference condition for MAE/UAI (default: control_twostage, fair vs two-stage anchors).",
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

    model_slug = args.model_id.replace("/", "_")
    model_out_dir = args.out_dir / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)

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

    fallback = None
    if args.llm_fallback:
        fb_model = args.fallback_model or args.model_id
        log.info("Loading LLM fallback extractor: %s on %s", fb_model, args.fallback_device)
        fallback = LLMFallbackExtractor(
            model_id=fb_model, device=args.fallback_device, dtype="bfloat16",
        )

    suffix = ""
    if args.request_final_line:
        suffix = (
            "\n\nPut your final numeric estimate (0-100) on the last line only, "
            "in this exact form: Answer: [number]"
        )

    log.info("Loading model %s (%s backend)...", args.model_id, args.backend)
    if args.backend == "vllm":
        backend = VLLMBackend(
            args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=True,
        )
    else:
        backend = HFBackend(args.model_id, device=args.device, dtype=args.dtype)

    results_path = model_out_dir / "results.jsonl"
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
        model_out_dir,
        label=f"History | {args.model_id}",
        baseline_condition=args.baseline_condition,
    )


if __name__ == "__main__":
    main()
