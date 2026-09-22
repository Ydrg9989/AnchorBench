#!/usr/bin/env python3
"""Run AnchorBench on a hosted OpenRouter model, for any set of suites.

Usage:
    python -m anchorbench.runners.api --model_id openai/gpt-5.4-mini \\
        --suites external history icl rag tool --out_dir results/api_benchmark

    python -m anchorbench.runners.api --model_id openai/gpt-5.4-mini \\
        --suites external --max_items 10 --out_dir results/api_smoke

The hosted model is wrapped in :class:`OpenRouterBackend`, so every suite runs
through the same evaluator loops as the local models; this module only picks
datasets and output directories. History defaults to the two-stage control
(``--history_baseline_condition``) and sends Stage 2 as a real three-turn
chat; ``--history_chat_format flat`` reproduces the single-message rendering
the published API cells used (docs/RECONCILIATION.md D9).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from anchorbench.eval.constants import SUITE_DATASETS as _CORE
from anchorbench.eval.constants import VARIANT_DATASETS
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

SUITE_DATASETS = {**_CORE, **VARIANT_DATASETS}
HISTORY_ANCHORED = ["irrelevant_low", "irrelevant_high", "plausible_low", "plausible_high"]


def load_items(suite: str, max_items: int | None, seed: int,
               conditions: list[str] | None = None) -> list[dict]:
    """Items of ``suite`` with every required condition present.

    Prefers the full ``promptviews.jsonl`` because ``control_twostage`` (the
    default History baseline) lives only there; ``prepare_items`` keeps just
    the requested conditions.
    """
    ds = ROOT / SUITE_DATASETS[suite]
    pv_file = ds / "promptviews.jsonl"
    if not pv_file.exists():
        pv_file = ds / "promptviews_core.jsonl"
    views = load_promptviews(pv_file)
    specs = load_itemspecs(ds / "itemspecs.jsonl")
    return prepare_items(views, specs, max_items, seed, conditions=conditions)


def run(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error("No API key. Set OPENROUTER_API_KEY or use --api_key")
        return 1
    backend = build_backend("openrouter", args.model_id,
                            max_concurrent=args.max_concurrent, api_key=api_key)
    slug = args.model_id.replace("/", "_")
    t_global = time.time()
    n_records = 0

    for suite in args.suites:
        if suite not in SUITE_DATASETS:
            log.warning("Unknown suite: %s (skipping)", suite)
            continue
        out_dir = Path(args.out_dir) / suite / slug
        if (out_dir / "summary.json").exists() and not args.force:
            log.info("[%s] Already complete (%s), skipping. Use --force to re-run.",
                     suite, out_dir / "summary.json")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        if suite == "history":
            conds = [args.history_baseline_condition, *HISTORY_ANCHORED]
            items = load_items(suite, args.max_items, args.seed, conds)
            log.info("[history] %d items x %d conditions", len(items), len(conds))
            records = run_history_two_stage(
                backend, items, out_dir / "results.jsonl",
                conditions=conds, max_tokens=args.max_tokens,
                batch_size=max(1, len(items) * len(conds)),
                prompt_suffix=args.prompt_suffix,
                chat_format=args.history_chat_format,
            )
            write_and_summarize(records, out_dir, label=f"History | {args.model_id}",
                                baseline_condition=args.history_baseline_condition)
        else:
            items = load_items(suite, args.max_items, args.seed)
            log.info("[%s] %d items (%d prompts)", suite, len(items), len(items) * 5)
            records = run_single_stage(
                backend, items, out_dir / "results.jsonl",
                max_tokens=args.max_tokens, batch_size=max(1, len(items) * 5),
                prompt_suffix=args.prompt_suffix,
            )
            write_and_summarize(records, out_dir, label=f"{suite.capitalize()} | {args.model_id}")
        n_records += len(records)

    log.info("=" * 60)
    log.info("COMPLETE: %d records across %d suites in %.1fs", n_records, len(args.suites),
             time.time() - t_global)
    log.info("Tokens: %s", backend.usage)
    log.info("=" * 60)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run AnchorBench via OpenRouter API")
    p.add_argument("--model_id", type=str, required=True,
                   help="OpenRouter model ID (e.g., openai/gpt-5.4-mini)")
    p.add_argument("--suites", nargs="+",
                   default=["external", "icl", "rag", "tool", "history"],
                   choices=sorted(SUITE_DATASETS), help="Suites to run")
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
                   choices=["control", "control_twostage"], default="control_twostage",
                   help="Baseline condition for History (default: control_twostage)")
    p.add_argument("--history_chat_format", choices=["messages", "flat"], default="messages",
                   help="Stage 2 as a real chat (messages) or quoted in one user turn (flat)")
    p.add_argument("--prompt_suffix", type=str, default="",
                   help="Text appended to every prompt (e.g., a CoT instruction)")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
