#!/usr/bin/env python3
"""Run inference on the Tool suite (5 conditions) and evaluate.

The anchor arrives as a tool-call response. Models whose chat template has a
tool role receive it as native tool messages; Gemma, OLMo and hosted API
models receive the plaintext rendering (see :func:`use_plaintext`).

Usage:
    python -m anchorbench.runners.tool \
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

from anchorbench.data.schema import ItemSpec
from anchorbench.data.suites.tool import TOOL_SCHEMAS, get_tool_messages
from anchorbench.eval.backends import Backend
from anchorbench.eval.evaluator import (
    prepare_items,
    run_single_stage,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.runner_utils import (
    add_common_args,
    make_backend,
    make_fallback,
    model_output_dir,
)

log = logging.getLogger(__name__)


def _tool_use_plaintext_prompts(model_id: str) -> bool:
    if os.environ.get("ANCHORBENCH_TOOL_PLAINTEXT"):
        return True
    m = model_id.lower()
    return "gemma" in m or "olmo" in m


def use_plaintext(args: argparse.Namespace, backend: Backend) -> bool:
    """Whether tool outputs are rendered as plaintext instead of tool messages.

    Plaintext is used when asked (``--tool_plaintext``), for model families
    whose chat template has no tool role (Gemma, OLMo), and for backends that
    cannot send tool messages at all (the hosted API). Everything else gets
    the native tool-call rendering; Appendix Table 15 compares the two
    within-model.
    """
    return (
        bool(getattr(args, "tool_plaintext", False))
        or _tool_use_plaintext_prompts(args.model_id)
        or not getattr(backend, "supports_tool_messages", True)
    )


def tool_messages(item: dict, cond: str, pv: dict) -> list[dict]:
    """Chat messages carrying the tool call and its (anchor-bearing) response."""
    return get_tool_messages(ItemSpec.from_dict(item["spec"]), cond)[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run Tool inference (batched) and evaluation")
    add_common_args(p)
    p.add_argument("--no_chat_template", action="store_true",
                   help="Send the plaintext promptview even to models with a tool template")
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

    if args.no_chat_template or use_plaintext(args, backend):
        log.info("Tool plaintext mode (%d prompts, batch_size=%d)", len(items) * 5, args.batch_size)
        records = run_single_stage(
            backend, items, results_path,
            max_tokens=args.max_tokens, batch_size=args.batch_size,
            use_llm_fallback=args.llm_fallback, fallback_extractor=fallback,
        )
    else:
        log.info("Tool-message mode (%d prompts, batch_size=%d)", len(items) * 5, args.batch_size)
        records = run_single_stage(
            backend, items, results_path,
            max_tokens=args.max_tokens, batch_size=args.batch_size,
            use_llm_fallback=args.llm_fallback, fallback_extractor=fallback,
            messages_fn=tool_messages, tools=TOOL_SCHEMAS,
        )

    write_and_summarize(records, model_out_dir, label=f"Tool | {args.model_id}")


if __name__ == "__main__":
    main()
