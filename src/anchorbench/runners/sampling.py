#!/usr/bin/env python3
"""Sampling robustness check: greedy versus temperature 0.7, several seeds.

Runs the External, RAG and ICL-dist suites once greedily and ``--n_seeds``
times with sampling, so the anchoring conclusions can be checked for
stability across decoding (Appendix Table 18). Any backend works: local
vLLM/HF or the hosted OpenRouter API.

Usage:
    bash scripts/run_with_env.sh python -m anchorbench.runners.sampling \\
        --model_id Qwen/Qwen2.5-7B-Instruct --backend vllm

    python -m anchorbench.runners.sampling \\
        --model_id openai/gpt-5.4-mini --backend openrouter

Output layout: ``<out_dir>/<suite>/<model_slug>/{greedy,t0.7_s<i>}/results.jsonl``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from anchorbench.eval.constants import SUITE_DATASETS as _ALL
from anchorbench.eval.constants import VARIANT_DATASETS
from anchorbench.eval.evaluator import prepare_items, run_single_stage, write_and_summarize
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.runner_utils import build_backend
from anchorbench.paths import ROOT

log = logging.getLogger(__name__)

SUITE_DATASETS = {
    k: v for k, v in {**_ALL, **VARIANT_DATASETS}.items()
    if k in ("external", "rag", "icl_dist")
}

SAMPLING_TEMP = 0.7
# The paper describes the sampled runs as top-p 0.9, but no backend in this
# package takes a top-p argument and this runner never passed one, so the
# published sampled cells ran at the backends' default top-p. Recorded as
# D10 in docs/RECONCILIATION.md; kept here as documentation of the intent.
SAMPLING_TOP_P = 0.9


def _tag(temperature: float, seed_idx: int) -> str:
    return "greedy" if temperature == 0.0 else f"t{temperature}_s{seed_idx}"


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
    settings = [(0.0, 0)] + [(SAMPLING_TEMP, si) for si in range(args.n_seeds)]

    for suite in args.suites:
        ds = ROOT / SUITE_DATASETS[suite]
        pv_file = ds / "promptviews_core.jsonl"
        if not pv_file.exists():
            pv_file = ds / "promptviews.jsonl"
        views = load_promptviews(pv_file)
        specs = load_itemspecs(ds / "itemspecs.jsonl")
        items = prepare_items(views, specs, args.max_items, args.seed)
        log.info("[%s] %d items loaded", suite, len(items))

        for temperature, seed_idx in settings:
            tag = _tag(temperature, seed_idx)
            out_dir = Path(args.out_dir) / suite / slug / tag
            if (out_dir / "summary.json").exists() and not args.force:
                log.info("[%s/%s] already done, skipping (use --force)", suite, tag)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            records = run_single_stage(
                backend, items, out_dir / "results.jsonl",
                max_tokens=args.max_tokens,
                batch_size=max(1, len(items) * 5) if hosted else args.batch_size,
                temperature=temperature,
                record_extras={"temperature": temperature, "seed_idx": seed_idx},
            )
            write_and_summarize(records, out_dir, label=f"{suite} ({tag}) | {args.model_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sampling robustness check")
    p.add_argument("--model_id", required=True)
    p.add_argument("--suites", nargs="+", default=list(SUITE_DATASETS),
                   choices=list(SUITE_DATASETS))
    p.add_argument("--out_dir", type=Path, default=Path("results/decoding_sampling_robustness"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_seeds", type=int, default=3, help="Number of sampled runs")
    p.add_argument("--backend", choices=["hf", "vllm", "openrouter", "api"], default="vllm",
                   help="'api' is an alias of 'openrouter'")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
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
