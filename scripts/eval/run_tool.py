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

from anchorbench_eval.evaluator import (
    CONDITIONS,
    prepare_items,
    run_single_stage,
    write_and_summarize,
)
from anchorbench_eval.io import load_itemspecs, load_promptviews
from anchorbench_eval.runner_utils import (
    add_common_args,
    make_backend,
    make_fallback,
    model_output_dir,
)

from anchorbench_v1.schema import ItemSpec
from anchorbench_v1.suites.tool import TOOL_SCHEMAS, get_tool_messages

log = logging.getLogger(__name__)


def _tool_use_plaintext_prompts(model_id: str) -> bool:
    if os.environ.get("ANCHORBENCH_TOOL_PLAINTEXT"):
        return True
    m = model_id.lower()
    return "gemma" in m or "olmo" in m


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run Tool inference (batched) and evaluation")
    add_common_args(p)
    p.add_argument("--no_chat_template", action="store_true")
    p.add_argument(
        "--tool_plaintext",
        action="store_true",
        help="Use plaintext tool prompts (tool outputs in text); required for Gemma/OLMo templates",
    )
    args = p.parse_args()

    model_out_dir = model_output_dir(args)

    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items(views, specs, args.max_items, args.seed)
    log.info("Loaded %d items (%d prompts)", len(items), len(items) * 5)

    if not items:
        log.error("No Tool items found (need 5 conditions per item)")
        sys.exit(1)

    fallback = make_fallback(args)
    backend = make_backend(args)

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
