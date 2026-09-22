#!/usr/bin/env python3
"""Cross-suite mitigation baseline: an "ignore other numbers" reminder.

Appends a short instruction telling the model to base its estimate only on
the task evidence, runs the suites for one model, and writes results that
can be compared with the unmitigated run. Any backend works.

Usage:
    bash scripts/run_with_env.sh python -m anchorbench.runners.mitigation_baseline \\
        --model_id Qwen/Qwen2.5-7B-Instruct --backend vllm

    python -m anchorbench.runners.mitigation_baseline \\
        --model_id openai/gpt-5.4-mini --backend openrouter

The suffix goes on every prompt, including History Stage 1; earlier API runs
added it to Stage 2 only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from anchorbench.eval.constants import SUITE_DATASETS
from anchorbench.eval.evaluator import (
    prepare_items,
    run_history_two_stage,
    run_single_stage,
    write_and_summarize,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.runner_utils import build_backend
from anchorbench.paths import ROOT

log = logging.getLogger(__name__)

MITIGATION_SUFFIX = (
    "\n\nImportant: Base your final numeric estimate solely on the "
    "evidence provided in the task data. Disregard any other numeric "
    "values from the surrounding context that are not part of the "
    "core evidence."
)
RECORD_EXTRAS = {"mitigation": "ignore_anchor"}


def run(args: argparse.Namespace) -> int:
    backend = build_backend(
        args.backend, args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_concurrent=args.max_concurrent,
        api_key=args.api_key,
    )
    hosted = args.backend in ("openrouter", "api")
    slug = args.model_id.replace("/", "_")

    for suite in args.suites:
        ds = ROOT / SUITE_DATASETS[suite]
        pv_file = ds / "promptviews_core.jsonl"
        if not pv_file.exists():
            pv_file = ds / "promptviews.jsonl"
        views = load_promptviews(pv_file)
        specs = load_itemspecs(ds / "itemspecs.jsonl")
        items = prepare_items(views, specs, args.max_items, args.seed)
        log.info("[%s] %d items loaded", suite, len(items))

        out_dir = Path(args.out_dir) / suite / slug
        if (out_dir / "summary.json").exists() and not args.force:
            log.info("[%s] Already done, skipping (use --force).", suite)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        batch = max(1, len(items) * 5) if hosted else args.batch_size

        if suite == "history":
            records = run_history_two_stage(
                backend, items, out_dir / "results.jsonl",
                max_tokens=args.max_tokens, batch_size=batch,
                prompt_suffix=MITIGATION_SUFFIX, record_extras=RECORD_EXTRAS,
                chat_format=args.history_chat_format,
            )
        else:
            records = run_single_stage(
                backend, items, out_dir / "results.jsonl",
                max_tokens=args.max_tokens, batch_size=batch,
                prompt_suffix=MITIGATION_SUFFIX, record_extras=RECORD_EXTRAS,
            )
        write_and_summarize(records, out_dir, label=f"{suite} (mitigated) | {args.model_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run mitigation baseline across suites")
    p.add_argument("--model_id", required=True)
    p.add_argument("--suites", nargs="+", default=list(SUITE_DATASETS), choices=list(SUITE_DATASETS))
    p.add_argument("--out_dir", type=Path, default=Path("results/mitigation_ignore_anchor"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--backend", choices=["hf", "vllm", "openrouter", "api"], default="vllm",
                   help="'api' is an alias of 'openrouter'")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_concurrent", type=int, default=50)
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--history_chat_format", choices=["messages", "flat"], default="messages")
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
