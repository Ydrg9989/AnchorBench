#!/usr/bin/env python3
"""Run inference on ICL-style datasets (5 core conditions, batched) and evaluate.

Works for the metadata-header ICL suite (``icl``) and for ``icl_dist`` (demo-label
bands × framing); both ship the same five condition names.

Usage:
    PYTHONPATH=src python scripts/eval/run_icl.py \
        --promptviews datasets/anchorbench_icl_core/promptviews.jsonl \
        --itemspecs datasets/anchorbench_icl_core/itemspecs.jsonl \
        --model_id Qwen/Qwen2.5-7B-Instruct \
        --out_dir results/icl_core \
        --batch_size 32

    PYTHONPATH=src python scripts/eval/run_icl.py \
        --promptviews datasets/anchorbench_icl_dist_smoke/promptviews.jsonl \
        --itemspecs datasets/anchorbench_icl_dist_smoke/itemspecs.jsonl \
        --out_dir results/icl_dist_smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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

    p = argparse.ArgumentParser(description="Run ICL inference (batched) and evaluation")
    add_common_args(p)
    args = p.parse_args()

    model_out = model_output_dir(args)

    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items(views, specs, args.max_items, args.seed)
    log.info("Loaded %d items (%d prompts)", len(items), len(items) * 5)

    if not items:
        log.error("No ICL items found (need 5 conditions per item)")
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

    write_and_summarize(records, model_out, label=f"ICL | {args.model_id}")


if __name__ == "__main__":
    main()
