#!/usr/bin/env python3
"""Run AnchorBench evaluation via OpenRouter API for all suites.

Supports all 5 suites (external, history, icl, rag, tool) with async
concurrent API calls for maximum throughput and cost efficiency.

Usage:
    PYTHONPATH=src python scripts/eval/run_api_benchmark.py \
        --model_id openai/gpt-5.4-mini \
        --suites external history icl rag tool \
        --out_dir results/api_benchmark \
        --max_concurrent 50 \
        --max_items 10

    # Smoke test (10 items/suite, all 4 models):
    PYTHONPATH=src python scripts/eval/run_api_benchmark.py \
        --model_id openai/gpt-5.4-mini \
        --suites external icl rag tool \
        --out_dir results/api_smoke \
        --max_items 10
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench.eval.evaluator import (
    CONDITIONS,
    build_record,
    parse_response,
    prepare_items,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.constants import SUITE_DATASETS as _CORE, VARIANT_DATASETS
from anchorbench.inference.async_api import AsyncOpenRouterClient

log = logging.getLogger(__name__)

SUITE_DATASETS = {**_CORE, **VARIANT_DATASETS}


async def run_suite_api(
    client: AsyncOpenRouterClient,
    model_id: str,
    suite: str,
    items: list[dict],
    out_dir: Path,
    max_tokens: int = 512,
    max_concurrent: int = 50,
    prompt_suffix: str = "",
    conditions: list[str] | None = None,
) -> list[dict]:
    """Run a single suite via API with high concurrency.

    ``prompt_suffix`` is appended to every prompt before sending (used for
    rebuttal experiments such as the CoT prompt). Defaults to '' for
    backwards-compatible behavior.

    ``conditions`` overrides the default 5 conditions (used by ablation runs
    that only need a subset).
    """
    model_slug = model_id.replace("/", "_")
    model_out_dir = out_dir / suite / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_out_dir / "results.jsonl"

    use_conds = list(conditions) if conditions else CONDITIONS

    task_list = []
    for item_idx, item in enumerate(items):
        for cond in use_conds:
            pv = item.get(cond)
            if pv is None:
                continue
            task_list.append((item_idx, cond, pv))

    prompts = [pv.get("prompt_text", "") + prompt_suffix for _, _, pv in task_list]

    log.info(
        "[%s] Running %d prompts via API (max_concurrent=%d)...",
        suite, len(prompts), max_concurrent,
    )
    t0 = time.time()

    results = await client.query_batch(
        model_id, prompts,
        max_tokens=max_tokens, temperature=0.0,
    )

    elapsed = time.time() - t0
    log.info(
        "[%s] API calls completed in %.1fs (%.1f prompts/sec)",
        suite, elapsed, len(prompts) / elapsed,
    )

    total_input_tok = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    total_output_tok = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)
    log.info(
        "[%s] Token usage: input=%d, output=%d",
        suite, total_input_tok, total_output_tok,
    )

    records = []
    with open(results_path, "w", encoding="utf-8") as fh:
        for (item_idx, cond, pv), api_result in zip(task_list, results):
            item = items[item_idx]
            raw = api_result.get("raw_text", "")
            prompt_text = pv.get("prompt_text", "")

            answer, parsed_ok, strategy = parse_response(raw, prompt_text)

            rec = build_record(
                model_id, item, cond, pv,
                answer, parsed_ok, strategy, raw,
            )
            rec["api_usage"] = api_result.get("usage", {})
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    write_and_summarize(
        records, model_out_dir,
        label=f"{suite.capitalize()} | {model_id}",
    )
    return records


async def run_history_suite_api(
    client: AsyncOpenRouterClient,
    model_id: str,
    items: list[dict],
    out_dir: Path,
    max_tokens: int = 512,
    max_concurrent: int = 50,
    baseline_condition: str = "control_twostage",
    prompt_suffix: str = "",
) -> list[dict]:
    """Run History suite with two-stage protocol via API.

    Stage 1: send partial-evidence prompt, get model's initial estimate.
    Stage 2: send full-evidence prompt with Stage 1 answer in 'history'.
    Control: single-stage (no history) or two-stage (control_twostage).
    """
    model_slug = model_id.replace("/", "_")
    model_out_dir = out_dir / "history" / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_out_dir / "results.jsonl"

    control_tasks = []
    anchored_tasks = []

    conditions = [
        baseline_condition,
        "irrelevant_low",
        "irrelevant_high",
        "plausible_low",
        "plausible_high",
    ]

    for item_idx, item in enumerate(items):
        for cond in conditions:
            pv = item[cond]
            if cond == baseline_condition:
                control_tasks.append((item_idx, cond, pv))
            else:
                anchored_tasks.append((item_idx, cond, pv))

    log.info("[history] Running %d control prompts (single-stage)...", len(control_tasks))
    t0 = time.time()

    control_prompts = [pv.get("prompt_text", "") + prompt_suffix for _, _, pv in control_tasks]
    control_results = await client.query_batch(
        model_id, control_prompts,
        max_tokens=max_tokens, temperature=0.0,
    )

    log.info(
        "[history] Running %d anchored prompts (two-stage)...",
        len(anchored_tasks),
    )

    stage1_prompts = []
    for item_idx, cond, pv in anchored_tasks:
        comps = pv.get("prompt_components", {})
        stage1_msg = comps.get("stage1_user_message", "")
        if not stage1_msg:
            stage1_msg = pv.get("prompt_text", "")
        stage1_prompts.append(stage1_msg + prompt_suffix)

    stage1_results = await client.query_batch(
        model_id, stage1_prompts,
        max_tokens=max_tokens, temperature=0.0,
    )

    stage2_prompts = []
    for (item_idx, cond, pv), s1_result in zip(anchored_tasks, stage1_results):
        comps = pv.get("prompt_components", {})
        stage2_msg = comps.get("stage2_user_message", "")
        s1_answer = s1_result.get("raw_text", "")

        if stage2_msg:
            conversation = (
                f"Previous conversation:\n"
                f"User: {stage1_prompts[anchored_tasks.index((item_idx, cond, pv))]}\n"
                f"Assistant: {s1_answer}\n\n"
                f"User: {stage2_msg}{prompt_suffix}"
            )
        else:
            conversation = pv.get("prompt_text", "") + prompt_suffix

        stage2_prompts.append(conversation)

    stage2_results = await client.query_batch(
        model_id, stage2_prompts,
        max_tokens=max_tokens, temperature=0.0,
    )

    elapsed = time.time() - t0
    log.info("[history] All stages completed in %.1fs", elapsed)

    records = []
    with open(results_path, "w", encoding="utf-8") as fh:
        for (item_idx, cond, pv), api_result in zip(control_tasks, control_results):
            item = items[item_idx]
            raw = api_result.get("raw_text", "")
            prompt_text = pv.get("prompt_text", "")
            answer, parsed_ok, strategy = parse_response(raw, prompt_text)
            rec = build_record(model_id, item, cond, pv, answer, parsed_ok, strategy, raw)
            rec["api_usage"] = api_result.get("usage", {})
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

        for (item_idx, cond, pv), s1_res, s2_res in zip(
            anchored_tasks, stage1_results, stage2_results
        ):
            item = items[item_idx]
            raw = s2_res.get("raw_text", "")
            prompt_text = pv.get("prompt_text", "")
            answer, parsed_ok, strategy = parse_response(raw, prompt_text)
            rec = build_record(model_id, item, cond, pv, answer, parsed_ok, strategy, raw)
            rec["stage1_raw_text"] = s1_res.get("raw_text", "")
            stage1_answer, _, _ = parse_response(rec["stage1_raw_text"], "")
            use_as_anchor = cond.startswith(("plausible_", "irrelevant_"))
            rec["anchor_value"] = stage1_answer if use_as_anchor else None
            rec["stage1_answer"] = stage1_answer
            s1_usage = s1_res.get("usage", {})
            s2_usage = s2_res.get("usage", {})
            rec["api_usage"] = {
                "prompt_tokens": s1_usage.get("prompt_tokens", 0) + s2_usage.get("prompt_tokens", 0),
                "completion_tokens": s1_usage.get("completion_tokens", 0) + s2_usage.get("completion_tokens", 0),
            }
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    write_and_summarize(records, model_out_dir, label=f"History | {model_id}", baseline_condition=baseline_condition)
    return records


async def main_async(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error("No API key. Set OPENROUTER_API_KEY or use --api_key")
        sys.exit(1)

    client = AsyncOpenRouterClient(
        api_key=api_key,
        max_concurrent=args.max_concurrent,
    )

    total_records = []
    total_input_tok = 0
    total_output_tok = 0
    t_global = time.time()

    model_slug = args.model_id.replace("/", "_")

    for suite in args.suites:
        dataset_dir = SUITE_DATASETS.get(suite)
        if not dataset_dir:
            log.warning("Unknown suite: %s (skipping)", suite)
            continue

        existing = Path(args.out_dir) / suite / model_slug / "summary.json"
        if existing.exists() and not args.force:
            log.info("[%s] Already complete (%s), skipping. Use --force to re-run.", suite, existing)
            continue

        pv_file = Path(dataset_dir) / "promptviews.jsonl"
        if not pv_file.exists():
            pv_file = Path(dataset_dir) / "promptviews_core.jsonl"
        spec_file = Path(dataset_dir) / "itemspecs.jsonl"

        if not pv_file.exists():
            log.error("Missing %s", pv_file)
            continue

        views = load_promptviews(pv_file)
        specs = load_itemspecs(spec_file)
        if suite == "history":
            hist_conditions = [
                args.history_baseline_condition,
                "irrelevant_low",
                "irrelevant_high",
                "plausible_low",
                "plausible_high",
            ]
            items = prepare_items(views, specs, args.max_items, args.seed, conditions=hist_conditions)
        else:
            items = prepare_items(views, specs, args.max_items, args.seed)
            
        log.info("[%s] Loaded %d items (%d prompts)", suite, len(items), len(items) * 5)

        if suite == "history":
            records = await run_history_suite_api(
                client, args.model_id, items, Path(args.out_dir),
                max_tokens=args.max_tokens,
                max_concurrent=args.max_concurrent,
                baseline_condition=args.history_baseline_condition,
                prompt_suffix=getattr(args, "prompt_suffix", ""),
            )
        else:
            records = await run_suite_api(
                client, args.model_id, suite, items, Path(args.out_dir),
                max_tokens=args.max_tokens,
                max_concurrent=args.max_concurrent,
                prompt_suffix=getattr(args, "prompt_suffix", ""),
            )

        total_records.extend(records)
        suite_input = sum(r.get("api_usage", {}).get("prompt_tokens", 0) for r in records)
        suite_output = sum(r.get("api_usage", {}).get("completion_tokens", 0) for r in records)
        total_input_tok += suite_input
        total_output_tok += suite_output

    await client.close()

    elapsed = time.time() - t_global
    log.info("=" * 60)
    log.info("COMPLETE: %d total records across %d suites in %.1fs",
             len(total_records), len(args.suites), elapsed)
    log.info("Total tokens: input=%d, output=%d", total_input_tok, total_output_tok)
    log.info("=" * 60)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Run AnchorBench via OpenRouter API")
    p.add_argument("--model_id", type=str, required=True,
                   help="OpenRouter model ID (e.g., openai/gpt-5.4-mini)")
    p.add_argument("--suites", nargs="+",
                   default=["external", "icl", "rag", "tool", "history"],
                   choices=["external", "history", "icl", "icl_dist", "rag", "tool"],
                   help="Suites to run")
    p.add_argument("--out_dir", type=str, default="results/api_benchmark")
    p.add_argument("--max_items", type=int, default=None,
                   help="Max items per suite (None = all 360)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--max_concurrent", type=int, default=50,
                   help="Max concurrent API requests")
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if results already exist")
    p.add_argument("--history_baseline_condition", type=str,
                   choices=["control", "control_twostage"],
                   default="control_twostage",
                   help="Baseline condition for History suite (default: control_twostage)")
    p.add_argument("--prompt_suffix", type=str, default="",
                   help="Optional text to append to every prompt (e.g., CoT instruction)")
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
