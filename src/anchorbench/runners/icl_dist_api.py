#!/usr/bin/env python3
"""Run the ICL-dist (distribution-matching) variant on hosted API models.

Writes ``<out_dir>/<model_slug>/results.jsonl`` for each model, the layout
``anchorbench.paper`` reads for Appendix Table 14 (``DEFAULT_ICL_DIST_API``).

Usage:
    python -m anchorbench.runners.icl_dist_api \\
        --model_ids openai/gpt-5.4-mini anthropic/claude-haiku-4.5 \\
        --out_dir results/icl_dist_api
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from anchorbench.eval.constants import API_MODEL_IDS, VARIANT_DATASETS
from anchorbench.eval.evaluator import prepare_items, run_single_stage, write_and_summarize
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.runner_utils import build_backend
from anchorbench.paths import ROOT

log = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error("No API key. Set OPENROUTER_API_KEY or --api_key")
        return 1

    ds = ROOT / VARIANT_DATASETS["icl_dist"]
    views = load_promptviews(ds / "promptviews_core.jsonl")
    specs = load_itemspecs(ds / "itemspecs.jsonl")
    items = prepare_items(views, specs, args.max_items, args.seed)
    log.info("Loaded %d ICL-dist items (%d prompts)", len(items), len(items) * 5)

    for model_id in args.model_ids:
        out_dir = Path(args.out_dir) / model_id.replace("/", "_")
        if (out_dir / "summary.json").exists() and not args.force:
            log.info("[%s] Already done, skipping.", model_id)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        backend = build_backend("openrouter", model_id,
                                max_concurrent=args.max_concurrent, api_key=api_key)
        records = run_single_stage(
            backend, items, out_dir / "results.jsonl",
            max_tokens=args.max_tokens, batch_size=max(1, len(items) * 5),
        )
        write_and_summarize(records, out_dir, label=f"ICL-dist | {model_id}")
        log.info("[%s] tokens: %s", model_id, backend.usage)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run ICL-dist on API models")
    p.add_argument("--model_ids", nargs="+", default=list(API_MODEL_IDS))
    p.add_argument("--out_dir", type=str, default="results/icl_dist_api")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--max_concurrent", type=int, default=50)
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
