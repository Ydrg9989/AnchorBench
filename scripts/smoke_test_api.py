#!/usr/bin/env python3
"""AnchorBench Smoke Test — API Models (OpenRouter).

Runs 10% of each suite (36 items) on API models via OpenRouter.
Supports GPT-4o-mini, Gemini 2.5 Pro, and any OpenRouter model.

Usage:
    PYTHONPATH=src python scripts/smoke_test_api.py

    # Single model, single suite:
    PYTHONPATH=src python scripts/smoke_test_api.py \
        --model openai/gpt-4o-mini --suites external

Environment:
    OPENROUTER_API_KEY — required for OpenRouter API access
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anchorbench_eval.evaluator import (
    CONDITIONS,
    build_record,
    parse_response,
    prepare_items,
    write_and_summarize,
)
from anchorbench_eval.io import load_itemspecs, load_promptviews
from anchorbench_eval.parsing import LLMFallbackExtractor
from mitigation_eval.async_api import AsyncOpenRouterClient

log = logging.getLogger(__name__)

API_MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview-03-25",
}

ALL_SUITES = ["external", "icl", "rag", "tool", "history"]


async def run_api_suite(
    client: AsyncOpenRouterClient,
    model_id: str,
    suite: str,
    items: list[dict],
    out_path: Path,
    max_tokens: int = 512,
) -> list[dict]:
    """Run all items x conditions for one suite via API."""
    conds = CONDITIONS

    task_list = []
    for item_idx, item in enumerate(items):
        for cond in conds:
            pv = item[cond]
            task_list.append((item_idx, cond, pv))

    prompts = [pv["prompt_text"] for _, _, pv in task_list]

    log.info(
        "  Querying API: %d prompts for %s/%s (max_tokens=%d)",
        len(prompts), suite, model_id, max_tokens,
    )

    results = await client.query_batch(
        model_id, prompts,
        max_tokens=max_tokens, temperature=0.0,
    )

    records = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for (item_idx, cond, pv), api_result in zip(task_list, results):
            item = items[item_idx]
            raw = api_result.get("raw_text", "")
            prompt_text = pv.get("prompt_text", "")

            answer, parsed_ok, strategy = parse_response(
                raw, prompt_text,
            )

            rec = build_record(
                model_id, item, cond, pv,
                answer, parsed_ok, strategy, raw,
            )
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    return records


async def run_api_history_suite(
    client: AsyncOpenRouterClient,
    model_id: str,
    items: list[dict],
    out_path: Path,
    max_tokens: int = 512,
) -> list[dict]:
    """Run History suite with two-stage protocol via API.

    Control conditions are single-stage; anchored conditions use
    Stage1 → parse → Stage2 (multi-turn chat). API calls are
    sequential per item since Stage2 depends on Stage1.
    """
    conds = CONDITIONS
    records = []
    total = len(items) * len(conds)
    done = 0

    with open(out_path, "w", encoding="utf-8") as fh:
        for item in items:
            for cond in conds:
                pv = item[cond]
                comp = pv.get("prompt_components", {})

                if cond == "control":
                    api_results = await client.query_batch(
                        model_id, [pv["prompt_text"]],
                        max_tokens=max_tokens, temperature=0.0,
                    )
                    raw = api_results[0].get("raw_text", "")
                    answer, parsed_ok, strategy = parse_response(
                        raw, pv["prompt_text"],
                    )
                    rec = build_record(
                        model_id, item, cond, pv,
                        answer, parsed_ok, strategy, raw,
                        anchor_value=None,
                        stage1_answer=None,
                        stage1_raw_text=None,
                    )
                else:
                    stage1_msg = comp.get("stage1_user_message", "")
                    stage2_msg = comp.get("stage2_user_message", "")

                    if not stage1_msg or not stage2_msg:
                        rec = build_record(
                            model_id, item, cond, pv,
                            None, False, "failed",
                            "MISSING_STAGE_COMPONENTS",
                            anchor_value=None,
                            stage1_answer=None,
                            stage1_raw_text=None,
                        )
                        records.append(rec)
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        done += 1
                        continue

                    s1_results = await client.query_batch(
                        model_id, [stage1_msg],
                        max_tokens=max_tokens, temperature=0.0,
                    )
                    stage1_raw = s1_results[0].get("raw_text", "")

                    stage1_answer, _, _ = parse_response(
                        stage1_raw, stage1_msg,
                    )

                    s2_results = await client.query_batch(
                        model_id, [stage2_msg],
                        max_tokens=max_tokens, temperature=0.0,
                        system_prompt=(
                            f"Previous conversation:\n"
                            f"User: {stage1_msg}\n"
                            f"Assistant: {stage1_raw}\n"
                        ),
                    )
                    stage2_raw = s2_results[0].get("raw_text", "")

                    answer, parsed_ok, strategy = parse_response(
                        stage2_raw, stage2_msg,
                    )

                    rec = build_record(
                        model_id, item, cond, pv,
                        answer, parsed_ok, strategy, stage2_raw,
                        anchor_value=stage1_answer,
                        stage1_answer=stage1_answer,
                        stage1_raw_text=stage1_raw,
                    )

                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                if done % 50 == 0:
                    log.info("  API inference %d/%d", done, total)

    return records


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    client = AsyncOpenRouterClient(
        api_key=api_key, max_concurrent=args.max_concurrent,
    )

    for model_name in args.models:
        model_id = API_MODELS.get(model_name, model_name)
        model_slug = model_id.replace("/", "_")
        log.info("=" * 60)
        log.info("Model: %s (%s)", model_name, model_id)

        for suite in args.suites:
            data_dir = Path(f"datasets/anchorbench_{suite}_core")
            if not data_dir.exists():
                log.warning("  Skipping %s: %s not found", suite, data_dir)
                continue

            out_dir = Path(args.results_root) / suite / model_slug
            out_dir.mkdir(parents=True, exist_ok=True)

            views = load_promptviews(data_dir / "promptviews.jsonl")
            specs = load_itemspecs(data_dir / "itemspecs.jsonl")
            items = prepare_items(views, specs, args.max_items, args.seed)

            if not items:
                log.warning("  No items for %s", suite)
                continue

            results_path = out_dir / "results.jsonl"
            log.info("  Suite: %s | %d items → %s", suite, len(items), results_path)

            t0 = time.time()
            if suite == "history":
                records = await run_api_history_suite(
                    client, model_id, items, results_path,
                    max_tokens=args.max_tokens,
                )
            else:
                records = await run_api_suite(
                    client, model_id, suite, items, results_path,
                    max_tokens=args.max_tokens,
                )
            elapsed = time.time() - t0

            metrics = write_and_summarize(
                records, out_dir,
                label=f"{suite} | {model_name}",
            )
            log.info(
                "  Done in %.1fs — parse_rate=%.1f%%, UAI_irr=%s, UAI_plaus=%s",
                elapsed,
                (metrics.get("parse_rate", 0) or 0) * 100,
                metrics.get("uai_irr"),
                metrics.get("uai_plaus"),
            )

    await client.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="AnchorBench smoke test — API models")
    p.add_argument(
        "--models", nargs="+",
        default=list(API_MODELS.keys()),
        help=f"Model names or OpenRouter IDs (default: {list(API_MODELS.keys())})",
    )
    p.add_argument(
        "--suites", nargs="+",
        default=ALL_SUITES,
        help=f"Suites to test (default: {ALL_SUITES})",
    )
    p.add_argument("--max_items", type=int, default=36, help="Items per suite (36 = 10%%)")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_concurrent", type=int, default=20)
    p.add_argument(
        "--results_root", type=str,
        default="results/smoke_test",
    )
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
