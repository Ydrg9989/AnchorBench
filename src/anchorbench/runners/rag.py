#!/usr/bin/env python3
"""Run inference on RAG dataset (5 conditions, batched) and evaluate.

Usage:
    PYTHONPATH=src python scripts/eval/run_rag.py \
        --promptviews datasets/anchorbench_rag_core/promptviews.jsonl \
        --itemspecs datasets/anchorbench_rag_core/itemspecs.jsonl \
        --model_id Qwen/Qwen2.5-7B-Instruct \
        --out_dir results/rag_core \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import logging
import sys

from anchorbench.eval.evaluator import (
    prepare_items,
    run_single_stage,
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

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run RAG inference (batched) and evaluation")
    add_common_args(p)
    args = p.parse_args()

    model_out = model_output_dir(args)

    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items(views, specs, args.max_items, args.seed)
    log.info("Loaded %d items (%d prompts)", len(items), len(items) * 5)

    if not items:
        log.error("No RAG items found (need 5 conditions per item)")
        sys.exit(1)

    fallback = make_fallback(args)
    suffix = build_suffix(args)
    backend = make_backend(args)

    results_path = model_out / "results.jsonl"
    records = run_single_stage(
        backend, items, results_path,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        prompt_suffix=suffix,
        use_llm_fallback=args.llm_fallback,
        fallback_extractor=fallback,
    )

    write_and_summarize(records, model_out, label=f"RAG | {args.model_id}")


if __name__ == "__main__":
    main()
