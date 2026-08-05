#!/usr/bin/env python3
"""Run the ICL numeric (distribution-matching) positive control on API models.

Reuses the same async API infrastructure as run_api_benchmark.py, pointing
at the anchorbench_icl_dist_core dataset.

Usage:
    python -m anchorbench.runners.icl_dist_api \
        --model_id openai/gpt-5.4-mini \
        --out_dir results/icl_numeric_api
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from anchorbench.eval.constants import VARIANT_DATASETS
from anchorbench.eval.evaluator import (
    CONDITIONS,
    build_record,
    parse_response,
    prepare_items,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.inference.async_api import AsyncOpenRouterClient

log = logging.getLogger(__name__)

ICL_DIST_DATASET = VARIANT_DATASETS.get("icl_dist", "datasets/anchorbench_icl_dist_core")


async def run_icl_dist_api(
    client: AsyncOpenRouterClient,
    model_id: str,
    items: list[dict],
    out_dir: Path,
    max_tokens: int = 512,
    max_concurrent: int = 50,
) -> list[dict]:
    """Run ICL-dist suite via API."""
    model_slug = model_id.replace("/", "_")
    model_out = out_dir / model_slug
    model_out.mkdir(parents=True, exist_ok=True)
    results_path = model_out / "results.jsonl"

    task_list = []
    for idx, item in enumerate(items):
        for cond in CONDITIONS:
            pv = item[cond]
            task_list.append((idx, cond, pv))

    prompts = [pv.get("prompt_text", "") for _, _, pv in task_list]

    log.info("Running %d ICL-dist prompts for %s...", len(prompts), model_id)
    t0 = time.time()
    results = await client.query_batch(
        model_id, prompts, max_tokens=max_tokens, temperature=0.0,
    )
    elapsed = time.time() - t0
    log.info("Done in %.1fs (%.1f prompts/sec)", elapsed, len(prompts) / elapsed)

    total_in = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    total_out = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)
    log.info("Tokens: input=%d, output=%d", total_in, total_out)

    records = []
    with open(results_path, "w") as fh:
        for (idx, cond, pv), api_res in zip(task_list, results):
            item = items[idx]
            raw = api_res.get("raw_text", "")
            prompt_text = pv.get("prompt_text", "")
            answer, ok, strategy = parse_response(raw, prompt_text)
            rec = build_record(model_id, item, cond, pv, answer, ok, strategy, raw)
            rec["api_usage"] = api_res.get("usage", {})
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_and_summarize(records, model_out, label=f"ICL-dist | {model_id}")
    return records


async def main_async(args):
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error("No API key. Set OPENROUTER_API_KEY or --api_key")
        sys.exit(1)

    client = AsyncOpenRouterClient(api_key=api_key, max_concurrent=args.max_concurrent)

    pv_file = Path(ICL_DIST_DATASET) / "promptviews_core.jsonl"
    spec_file = Path(ICL_DIST_DATASET) / "itemspecs.jsonl"

    views = load_promptviews(pv_file)
    specs = load_itemspecs(spec_file)
    items = prepare_items(views, specs, args.max_items, args.seed)
    log.info("Loaded %d ICL-dist items (%d prompts)", len(items), len(items) * 5)

    for model_id in args.model_ids:
        model_slug = model_id.replace("/", "_")
        existing = Path(args.out_dir) / model_slug / "summary.json"
        if existing.exists() and not args.force:
            log.info("[%s] Already done, skipping.", model_id)
            continue
        await run_icl_dist_api(client, model_id, items, Path(args.out_dir),
                               max_tokens=args.max_tokens,
                               max_concurrent=args.max_concurrent)

    await client.close()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Run ICL-dist on API models")
    p.add_argument("--model_ids", nargs="+", default=[
        "openai/gpt-5.4-mini",
        "anthropic/claude-haiku-4.5",
        "google/gemini-2.5-flash",
        "x-ai/grok-3-mini-beta",
    ])
    p.add_argument("--out_dir", type=str, default="results/icl_numeric_api")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--max_concurrent", type=int, default=50)
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
