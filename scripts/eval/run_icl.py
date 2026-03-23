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

from anchorbench_eval.backends import HFBackend, VLLMBackend
from anchorbench_eval.evaluator import (
    prepare_items,
    run_single_stage,
    write_and_summarize,
)
from anchorbench_eval.io import load_itemspecs, load_promptviews
from anchorbench_eval.parsing import LLMFallbackExtractor, XML_TAG_INSTRUCTION

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run ICL inference (batched) and evaluation")
    p.add_argument("--promptviews", type=Path, required=True)
    p.add_argument("--itemspecs", type=Path, required=True)
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out_dir", type=Path, default=Path("results/icl"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--request_xml_answer", action="store_true",
                   help="Append XML tag instruction: wrap answer in <answer>N</answer>")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--llm_fallback", action="store_true")
    p.add_argument("--fallback_model", type=str, default=None)
    p.add_argument("--fallback_device", type=str, default="cuda:0")
    p.add_argument("--structured", action="store_true")
    p.add_argument("--backend", type=str, choices=["hf", "vllm"], default="hf",
                   help="Inference backend: hf or vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    args = p.parse_args()

    model_slug = args.model_id.replace("/", "_")
    model_out_dir = args.out_dir / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)

    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items(views, specs, args.max_items, args.seed)
    log.info("Loaded %d items (%d prompts)", len(items), len(items) * 5)

    if not items:
        log.error("No ICL items found (need 5 conditions per item)")
        sys.exit(1)

    fallback = None
    if args.llm_fallback:
        fb_model = args.fallback_model or args.model_id
        log.info("Loading LLM fallback extractor: %s on %s", fb_model, args.fallback_device)
        fallback = LLMFallbackExtractor(
            model_id=fb_model, device=args.fallback_device, dtype="bfloat16",
        )

    suffix = ""
    if args.request_xml_answer:
        suffix += XML_TAG_INSTRUCTION

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
        backend = HFBackend(
            args.model_id, device=args.device,
            device_map=args.device_map, dtype=args.dtype,
        )

    results_path = model_out_dir / "results.jsonl"
    records = run_single_stage(
        backend, items, results_path,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        prompt_suffix=suffix,
        use_llm_fallback=args.llm_fallback,
        fallback_extractor=fallback,
    )

    write_and_summarize(records, model_out_dir, label=f"ICL | {args.model_id}")


if __name__ == "__main__":
    main()
