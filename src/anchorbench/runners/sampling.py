#!/usr/bin/env python3
"""Sampling robustness check: greedy vs temperature=0.7 top_p=0.9.

Runs a subset of suites and models under sampling to test whether
anchoring conclusions are stable across decoding strategies.

Usage (open-weight, vLLM):
    bash scripts/run_with_env.sh python -m anchorbench.runners.sampling \
        --model_id Qwen/Qwen2.5-7B-Instruct --backend vllm \
        --out_dir results/decoding_sampling_robustness

Usage (API):
    python -m anchorbench.runners.sampling \
        --model_id openai/gpt-5.4-mini --backend api \
        --out_dir results/decoding_sampling_robustness
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

from anchorbench.eval.constants import SUITE_DATASETS as _ALL
from anchorbench.eval.constants import VARIANT_DATASETS
from anchorbench.eval.evaluator import (
    CONDITIONS,
    build_record,
    parse_response,
    prepare_items,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews

log = logging.getLogger(__name__)

SUITE_DATASETS = {
    k: v for k, v in {**_ALL, **VARIANT_DATASETS}.items()
    if k in ("external", "rag", "icl_dist")
}

SAMPLING_TEMP = 0.7
SAMPLING_TOP_P = 0.9


def run_suite_local_sampled(backend, suite: str, items: list[dict],
                            out_dir: Path, max_tokens: int, batch_size: int,
                            temperature: float, seed_idx: int) -> list[dict]:
    """Run a suite locally with specified temperature."""
    model_slug = backend.model_id.replace("/", "_")
    tag = "greedy" if temperature == 0.0 else f"t{temperature}_s{seed_idx}"
    model_out = out_dir / suite / model_slug / tag
    model_out.mkdir(parents=True, exist_ok=True)
    results_path = model_out / "results.jsonl"

    task_list = []
    for idx, item in enumerate(items):
        for cond in CONDITIONS:
            pv = item[cond]
            task_list.append((idx, cond, pv))

    prompts = [pv.get("prompt_text", "") for _, _, pv in task_list]
    log.info("[%s/%s] Running %d prompts (temp=%.1f)...", suite, tag, len(prompts), temperature)

    raw_outputs = backend.generate_batch(
        prompts, max_tokens=max_tokens, temperature=temperature,
        batch_size=batch_size,
    )

    records = []
    with open(results_path, "w") as fh:
        for (idx, cond, pv), raw in zip(task_list, raw_outputs):
            item = items[idx]
            prompt_text = pv.get("prompt_text", "")
            answer, ok, strategy = parse_response(raw, prompt_text)
            rec = build_record(backend.model_id, item, cond, pv,
                               answer, ok, strategy, raw)
            rec["temperature"] = temperature
            rec["seed_idx"] = seed_idx
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_and_summarize(records, model_out,
                        label=f"{suite} ({tag}) | {backend.model_id}")
    return records


async def run_suite_api_sampled(client, model_id: str, suite: str,
                                items: list[dict], out_dir: Path,
                                max_tokens: int, temperature: float,
                                seed_idx: int) -> list[dict]:
    """Run a suite via API with specified temperature."""
    model_slug = model_id.replace("/", "_")
    tag = "greedy" if temperature == 0.0 else f"t{temperature}_s{seed_idx}"
    model_out = out_dir / suite / model_slug / tag
    model_out.mkdir(parents=True, exist_ok=True)
    results_path = model_out / "results.jsonl"

    task_list = []
    for idx, item in enumerate(items):
        for cond in CONDITIONS:
            pv = item[cond]
            task_list.append((idx, cond, pv))

    prompts = [pv.get("prompt_text", "") for _, _, pv in task_list]

    log.info("[%s/%s] Running %d prompts via API (temp=%.1f)...",
             suite, tag, len(prompts), temperature)
    t0 = time.time()
    results = await client.query_batch(
        model_id, prompts, max_tokens=max_tokens, temperature=temperature,
    )
    log.info("[%s/%s] Done in %.1fs", suite, tag, time.time() - t0)

    records = []
    with open(results_path, "w") as fh:
        for (idx, cond, pv), api_res in zip(task_list, results):
            item = items[idx]
            raw = api_res.get("raw_text", "")
            prompt_text = pv.get("prompt_text", "")
            answer, ok, strategy = parse_response(raw, prompt_text)
            rec = build_record(model_id, item, cond, pv, answer, ok, strategy, raw)
            rec["api_usage"] = api_res.get("usage", {})
            rec["temperature"] = temperature
            rec["seed_idx"] = seed_idx
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_and_summarize(records, model_out,
                        label=f"{suite} ({tag}) | {model_id}")
    return records


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Sampling robustness check")
    p.add_argument("--model_id", required=True)
    p.add_argument("--suites", nargs="+", default=list(SUITE_DATASETS.keys()),
                   choices=list(SUITE_DATASETS.keys()))
    p.add_argument("--out_dir", type=Path, default=Path("results/decoding_sampling_robustness"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_seeds", type=int, default=3,
                   help="Number of sampling runs (seeds)")
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
    from anchorbench.eval.backends import HFBackend, VLLMBackend

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

        run_suite_local_sampled(backend, suite, items, args.out_dir,
                                args.max_tokens, args.batch_size,
                                temperature=0.0, seed_idx=0)

        for si in range(args.n_seeds):
            run_suite_local_sampled(backend, suite, items, args.out_dir,
                                    args.max_tokens, args.batch_size,
                                    temperature=SAMPLING_TEMP, seed_idx=si)


async def _run_api(args):
    from anchorbench.inference.async_api import AsyncOpenRouterClient

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error("No API key.")
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

        await run_suite_api_sampled(client, args.model_id, suite, items,
                                    args.out_dir, args.max_tokens,
                                    temperature=0.0, seed_idx=0)

        for si in range(args.n_seeds):
            await run_suite_api_sampled(client, args.model_id, suite, items,
                                        args.out_dir, args.max_tokens,
                                        temperature=SAMPLING_TEMP, seed_idx=si)

    await client.close()


if __name__ == "__main__":
    main()
