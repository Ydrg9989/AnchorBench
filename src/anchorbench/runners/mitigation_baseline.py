#!/usr/bin/env python3
"""Cross-suite mitigation baseline: "ignore-anchor reminder" prompt suffix.

Appends a short instruction telling the model to base its answer only on
task evidence.  Runs all five suites for a single model and compares with
the original (no-mitigation) results.

Usage (open-weight, vLLM):
    bash scripts/run_with_env.sh python scripts/eval/run_mitigation_baseline.py \
        --model_id Qwen/Qwen2.5-7B-Instruct --backend vllm \
        --out_dir results/mitigation_ignore_anchor

Usage (API):
    PYTHONPATH=src python scripts/eval/run_mitigation_baseline.py \
        --model_id openai/gpt-5.4-mini --backend api \
        --out_dir results/mitigation_ignore_anchor
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

from anchorbench_eval.evaluator import (
    CONDITIONS,
    build_record,
    parse_response,
    prepare_items,
    run_single_stage,
    write_and_summarize,
)
from anchorbench_eval.constants import SUITE_DATASETS
from anchorbench_eval.io import load_itemspecs, load_promptviews

log = logging.getLogger(__name__)

MITIGATION_SUFFIX = (
    "\n\nImportant: Base your final numeric estimate solely on the "
    "evidence provided in the task data. Disregard any other numeric "
    "values from the surrounding context that are not part of the "
    "core evidence."
)


def run_suite_local(backend, suite: str, items: list[dict], out_dir: Path,
                    max_tokens: int, batch_size: int) -> list[dict]:
    """Run a non-history suite locally with mitigation suffix."""
    model_slug = backend.model_id.replace("/", "_")
    model_out = out_dir / suite / model_slug
    model_out.mkdir(parents=True, exist_ok=True)
    results_path = model_out / "results.jsonl"

    records = run_single_stage(
        backend, items, results_path,
        max_tokens=max_tokens, batch_size=batch_size,
        prompt_suffix=MITIGATION_SUFFIX,
    )
    write_and_summarize(records, model_out, label=f"{suite} (mitigated) | {backend.model_id}")
    return records


def run_history_local(backend, items: list[dict], out_dir: Path,
                      max_tokens: int) -> list[dict]:
    """Run history suite locally with mitigation suffix."""
    from anchorbench_eval.evaluator import run_history_two_stage
    model_slug = backend.model_id.replace("/", "_")
    model_out = out_dir / "history" / model_slug
    model_out.mkdir(parents=True, exist_ok=True)
    results_path = model_out / "results.jsonl"

    records = run_history_two_stage(
        backend, items, results_path,
        max_tokens=max_tokens,
        prompt_suffix=MITIGATION_SUFFIX,
    )
    write_and_summarize(records, model_out, label=f"history (mitigated) | {backend.model_id}")
    return records


async def run_suite_api(client, model_id: str, suite: str,
                        items: list[dict], out_dir: Path,
                        max_tokens: int, max_concurrent: int) -> list[dict]:
    """Run a non-history suite via API with mitigation suffix."""
    model_slug = model_id.replace("/", "_")
    model_out = out_dir / suite / model_slug
    model_out.mkdir(parents=True, exist_ok=True)
    results_path = model_out / "results.jsonl"

    task_list = []
    for idx, item in enumerate(items):
        for cond in CONDITIONS:
            pv = item[cond]
            task_list.append((idx, cond, pv))

    prompts = [pv.get("prompt_text", "") + MITIGATION_SUFFIX
               for _, _, pv in task_list]

    log.info("[%s] Running %d mitigated prompts via API...", suite, len(prompts))
    t0 = time.time()
    results = await client.query_batch(
        model_id, prompts, max_tokens=max_tokens, temperature=0.0,
    )
    log.info("[%s] Done in %.1fs", suite, time.time() - t0)

    records = []
    with open(results_path, "w") as fh:
        for (idx, cond, pv), api_res in zip(task_list, results):
            item = items[idx]
            raw = api_res.get("raw_text", "")
            prompt_text = pv.get("prompt_text", "")
            answer, ok, strategy = parse_response(raw, prompt_text)
            rec = build_record(model_id, item, cond, pv, answer, ok, strategy, raw)
            rec["api_usage"] = api_res.get("usage", {})
            rec["mitigation"] = "ignore_anchor"
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_and_summarize(records, model_out,
                        label=f"{suite} (mitigated) | {model_id}")
    return records


async def run_history_api(client, model_id: str, items: list[dict],
                          out_dir: Path, max_tokens: int,
                          max_concurrent: int) -> list[dict]:
    """History suite via API with mitigation suffix on stage-2."""
    model_slug = model_id.replace("/", "_")
    model_out = out_dir / "history" / model_slug
    model_out.mkdir(parents=True, exist_ok=True)
    results_path = model_out / "results.jsonl"

    control_tasks, anchored_tasks = [], []
    for idx, item in enumerate(items):
        for cond in CONDITIONS:
            pv = item[cond]
            if cond == "control":
                control_tasks.append((idx, cond, pv))
            else:
                anchored_tasks.append((idx, cond, pv))

    ctrl_prompts = [pv.get("prompt_text", "") + MITIGATION_SUFFIX
                    for _, _, pv in control_tasks]
    log.info("[history] Running %d control prompts (mitigated)...", len(ctrl_prompts))
    ctrl_results = await client.query_batch(
        model_id, ctrl_prompts, max_tokens=max_tokens, temperature=0.0,
    )

    stage1_prompts = []
    for idx, cond, pv in anchored_tasks:
        comps = pv.get("prompt_components", {})
        stage1_prompts.append(comps.get("stage1_user_message", pv.get("prompt_text", "")))

    log.info("[history] Running %d stage-1 prompts...", len(stage1_prompts))
    s1_results = await client.query_batch(
        model_id, stage1_prompts, max_tokens=max_tokens, temperature=0.0,
    )

    stage2_prompts = []
    for (idx, cond, pv), s1_res in zip(anchored_tasks, s1_results):
        comps = pv.get("prompt_components", {})
        stage2_msg = comps.get("stage2_user_message", "")
        s1_text = s1_res.get("raw_text", "")
        if stage2_msg:
            conv = (f"Previous conversation:\nUser: {stage1_prompts[anchored_tasks.index((idx, cond, pv))]}\n"
                    f"Assistant: {s1_text}\n\nUser: {stage2_msg}")
        else:
            conv = pv.get("prompt_text", "")
        stage2_prompts.append(conv + MITIGATION_SUFFIX)

    log.info("[history] Running %d stage-2 prompts (mitigated)...", len(stage2_prompts))
    s2_results = await client.query_batch(
        model_id, stage2_prompts, max_tokens=max_tokens, temperature=0.0,
    )

    records = []
    with open(results_path, "w") as fh:
        for (idx, cond, pv), api_res in zip(control_tasks, ctrl_results):
            item = items[idx]
            raw = api_res.get("raw_text", "")
            answer, ok, strategy = parse_response(raw, pv.get("prompt_text", ""))
            rec = build_record(model_id, item, cond, pv, answer, ok, strategy, raw)
            rec["api_usage"] = api_res.get("usage", {})
            rec["mitigation"] = "ignore_anchor"
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for (idx, cond, pv), s1_res, s2_res in zip(anchored_tasks, s1_results, s2_results):
            item = items[idx]
            raw = s2_res.get("raw_text", "")
            answer, ok, strategy = parse_response(raw, pv.get("prompt_text", ""))
            rec = build_record(model_id, item, cond, pv, answer, ok, strategy, raw)
            rec["stage1_raw_text"] = s1_res.get("raw_text", "")
            rec["mitigation"] = "ignore_anchor"
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_and_summarize(records, model_out, label=f"history (mitigated) | {model_id}")
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Run mitigation baseline across suites")
    p.add_argument("--model_id", required=True)
    p.add_argument("--suites", nargs="+", default=list(SUITE_DATASETS.keys()),
                   choices=list(SUITE_DATASETS.keys()))
    p.add_argument("--out_dir", type=Path, default=Path("results/mitigation_ignore_anchor"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--backend", choices=["hf", "vllm", "api"], default="vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_concurrent", type=int, default=50)
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.backend == "api":
        asyncio.run(_run_api(args))
    else:
        _run_local(args)


def _run_local(args):
    from anchorbench_eval.backends import HFBackend, VLLMBackend

    log.info("Loading model %s (%s)...", args.model_id, args.backend)
    if args.backend == "vllm":
        backend = VLLMBackend(
            args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype="bfloat16", trust_remote_code=True,
        )
    else:
        backend = HFBackend(args.model_id, device="auto", dtype="bfloat16")

    for suite in args.suites:
        ds = Path(SUITE_DATASETS[suite])
        pv_file = ds / "promptviews_core.jsonl"
        if not pv_file.exists():
            pv_file = ds / "promptviews.jsonl"
        spec_file = ds / "itemspecs.jsonl"

        views = load_promptviews(pv_file)
        specs = load_itemspecs(spec_file)
        items = prepare_items(views, specs, args.max_items, args.seed)
        log.info("[%s] %d items loaded", suite, len(items))

        model_slug = backend.model_id.replace("/", "_")
        existing = args.out_dir / suite / model_slug / "summary.json"
        if existing.exists() and not args.force:
            log.info("[%s] Already done, skipping (use --force).", suite)
            continue

        if suite == "history":
            run_history_local(backend, items, args.out_dir, args.max_tokens)
        else:
            run_suite_local(backend, suite, items, args.out_dir,
                            args.max_tokens, args.batch_size)


async def _run_api(args):
    from mitigation_eval.async_api import AsyncOpenRouterClient

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error("No API key. Set OPENROUTER_API_KEY or --api_key")
        sys.exit(1)

    client = AsyncOpenRouterClient(api_key=api_key, max_concurrent=args.max_concurrent)

    for suite in args.suites:
        ds = Path(SUITE_DATASETS[suite])
        pv_file = ds / "promptviews_core.jsonl"
        if not pv_file.exists():
            pv_file = ds / "promptviews.jsonl"
        spec_file = ds / "itemspecs.jsonl"

        views = load_promptviews(pv_file)
        specs = load_itemspecs(spec_file)
        items = prepare_items(views, specs, args.max_items, args.seed)
        log.info("[%s] %d items loaded", suite, len(items))

        model_slug = args.model_id.replace("/", "_")
        existing = args.out_dir / suite / model_slug / "summary.json"
        if existing.exists() and not args.force:
            log.info("[%s] Already done, skipping.", suite)
            continue

        if suite == "history":
            await run_history_api(client, args.model_id, items, args.out_dir,
                                  args.max_tokens, args.max_concurrent)
        else:
            await run_suite_api(client, args.model_id, suite, items, args.out_dir,
                                args.max_tokens, args.max_concurrent)

    await client.close()


if __name__ == "__main__":
    main()
