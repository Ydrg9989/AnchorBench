#!/usr/bin/env python3
"""Run inference on Tool dataset (5 conditions, chat-template) and evaluate.

Usage:
    PYTHONPATH=src python scripts/eval/run_tool.py \
        --promptviews datasets/anchorbench_tool_core/promptviews.jsonl \
        --itemspecs datasets/anchorbench_tool_core/itemspecs.jsonl \
        --model_id Qwen/Qwen2.5-7B-Instruct \
        --out_dir results/tool_core \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench_eval.backends import HFBackend, VLLMBackend
from anchorbench_eval.evaluator import (
    CONDITIONS,
    prepare_items,
    run_single_stage,
    write_and_summarize,
)
from anchorbench_eval.io import load_itemspecs, load_promptviews
from anchorbench_eval.parsing import LLMFallbackExtractor

from anchorbench_v1.schema import ItemSpec
from anchorbench_v1.suites.tool import TOOL_SCHEMAS, get_tool_messages

log = logging.getLogger(__name__)

# Models whose chat templates break on tool-call message shapes (None content, etc.).
# Use dataset plaintext (same anchoring content as native tool path).
def _tool_use_plaintext_prompts(model_id: str) -> bool:
    if os.environ.get("ANCHORBENCH_TOOL_PLAINTEXT"):
        return True
    m = model_id.lower()
    return "gemma" in m or "olmo" in m


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run Tool inference (batched) and evaluation")
    p.add_argument("--promptviews", type=Path, required=True)
    p.add_argument("--itemspecs", type=Path, required=True)
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out_dir", type=Path, default=Path("results/tool"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--llm_fallback", action="store_true")
    p.add_argument("--fallback_model", type=str, default=None)
    p.add_argument("--fallback_device", type=str, default="cuda:0")
    p.add_argument("--no_chat_template", action="store_true")
    p.add_argument("--structured", action="store_true")
    p.add_argument("--backend", type=str, choices=["hf", "vllm"], default="hf")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument(
        "--tool_plaintext",
        action="store_true",
        help="Use plaintext tool prompts (tool outputs in text); required for Gemma/OLMo templates",
    )
    args = p.parse_args()

    model_slug = args.model_id.replace("/", "_")
    model_out_dir = args.out_dir / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)

    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items(views, specs, args.max_items, args.seed)
    log.info("Loaded %d items (%d prompts)", len(items), len(items) * 5)

    if not items:
        log.error("No Tool items found (need 5 conditions per item)")
        sys.exit(1)

    fallback = None
    if args.llm_fallback:
        fb_model = args.fallback_model or args.model_id
        log.info("Loading LLM fallback extractor: %s on %s", fb_model, args.fallback_device)
        fallback = LLMFallbackExtractor(
            model_id=fb_model, device=args.fallback_device, dtype="bfloat16",
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
        backend = HFBackend(
            args.model_id, device=args.device,
            device_map=args.device_map, dtype=args.dtype,
        )

    results_path = model_out_dir / "results.jsonl"

    if args.no_chat_template:
        records = run_single_stage(
            backend, items, results_path,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            use_llm_fallback=args.llm_fallback,
            fallback_extractor=fallback,
        )
    else:
        task_list = []
        for item_idx, item in enumerate(items):
            for cond in CONDITIONS:
                pv = item[cond]
                task_list.append((item_idx, cond, pv))

        messages_list = []
        for item_idx, cond, pv in task_list:
            item = items[item_idx]
            spec = ItemSpec.from_dict(item["spec"])
            msgs, _ = get_tool_messages(spec, cond)
            messages_list.append(msgs)

        use_plain = args.tool_plaintext or _tool_use_plaintext_prompts(args.model_id)
        if use_plain:
            prompts = [pv.get("prompt_text", "") for _, _, pv in task_list]
            log.info(
                "Tool plaintext mode (%d prompts, batch_size=%d) — Gemma/OLMo-safe",
                len(prompts), args.batch_size,
            )
            raw_outputs = backend.generate_batch(
                prompts,
                max_tokens=args.max_tokens,
                temperature=0.0,
                batch_size=args.batch_size,
            )
        else:
            log.info(
                "Running batched tool inference: %d prompts, batch_size=%d",
                len(messages_list), args.batch_size,
            )
            raw_outputs = backend.generate_batch_tool(
                messages_list, tools=TOOL_SCHEMAS,
                max_tokens=args.max_tokens, temperature=0.0,
                batch_size=args.batch_size,
            )

        import json
        from anchorbench_eval.evaluator import build_record, parse_response

        records = []
        with open(results_path, "w", encoding="utf-8") as fh:
            for (item_idx, cond, pv), raw in zip(task_list, raw_outputs):
                item = items[item_idx]
                prompt_text = pv.get("prompt_text", "")
                answer, parsed_ok, strategy = parse_response(
                    raw, prompt_text,
                    use_llm_fallback=args.llm_fallback,
                    fallback_extractor=fallback,
                )
                rec = build_record(
                    backend.model_id, item, cond, pv,
                    answer, parsed_ok, strategy, raw,
                )
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    write_and_summarize(records, model_out_dir, label=f"Tool | {args.model_id}")


if __name__ == "__main__":
    main()
