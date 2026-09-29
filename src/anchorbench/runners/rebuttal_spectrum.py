#!/usr/bin/env python3
"""Plausibility-spectrum runner.

The core benchmark uses a binary relevance axis (irrelevant vs plausible);
a richer plausibility spectrum lets us measure the dose-response of anchor
credibility.

Evaluates the four ablation conditions (placebo_low/high, authority_low/high)
that already exist in ``datasets/anchorbench_external_core/promptviews_ablation.jsonl``
but were never run for the paper. Combined with the existing core conditions
(control, irrelevant, plausible) this yields a 4-point plausibility curve.

Outputs are written to
``results/rebuttal/spectrum/<model_slug>/results_ablation.jsonl`` and a
combined ``results_extended.jsonl`` that merges the new ablation records
with the existing core records from
``results/full_benchmark/external/<model_slug>/results.jsonl``.

Usage::

    bash scripts/run_with_env.sh \\
        python -m anchorbench.runners.rebuttal_spectrum \\
            --model_id google/gemma-3-4b-it \\
            --backend vllm

For API models, set ``--backend openrouter`` and ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from anchorbench.eval.evaluator import prepare_items, run_single_stage
from anchorbench.eval.io import (
    load_itemspecs,
    load_promptviews,
    splice_core_records,
    write_records,
)
from anchorbench.eval.metrics import compute_extended_metrics
from anchorbench.eval.runner_utils import build_backend

log = logging.getLogger(__name__)

ABLATION_CONDITIONS = [
    "placebo_low", "placebo_high",
    "authority_low", "authority_high",
]

DEFAULT_OUT = Path("results/rebuttal/spectrum")
DEFAULT_DATASET = Path("datasets/anchorbench_external_core")
DEFAULT_CORE_RESULTS = Path("results/full_benchmark/external")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Plausibility-spectrum runner (B2)")
    p.add_argument("--model_id", required=True)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--core_results_dir", type=Path,
                   default=DEFAULT_CORE_RESULTS)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backend",
                   choices=["hf", "vllm", "openrouter"], default="vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_concurrent", type=int, default=16,
                   help="OpenRouter concurrency")
    args = p.parse_args(argv)

    pv_path = args.dataset_dir / "promptviews_ablation.jsonl"
    spec_path = args.dataset_dir / "itemspecs.jsonl"
    items = prepare_items(load_promptviews(pv_path), load_itemspecs(spec_path),
                          args.max_items, args.seed, conditions=ABLATION_CONDITIONS)
    log.info("Loaded %d items x %d ablation conditions = %d prompts",
             len(items), len(ABLATION_CONDITIONS),
             len(items) * len(ABLATION_CONDITIONS))

    model_slug = args.model_id.replace("/", "_")
    out_dir = args.out_dir / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    abl_path = out_dir / "results_ablation.jsonl"

    backend = build_backend(
        args.backend, args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_concurrent=args.max_concurrent,
    )
    new_records = run_single_stage(
        backend, items, abl_path, conditions=ABLATION_CONDITIONS,
        max_tokens=args.max_tokens, batch_size=args.batch_size,
    )
    log.info("Inference complete: %d records", len(new_records))

    all_recs = splice_core_records(
        new_records, args.core_results_dir / model_slug / "results.jsonl",
    )
    write_records(all_recs, out_dir / "results_extended.jsonl")

    metrics = compute_extended_metrics(all_recs, baseline_condition="control")
    summary_path = out_dir / "summary_extended.json"
    summary_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Wrote %s", summary_path)

    # Quick console table of the spectrum
    print("\n=== Plausibility spectrum (4-point curve) ===")
    print(f"  Model: {args.model_id}")
    for key in ("uai_irr_ci", "uai_placebo_ci", "uai_plaus_ci",
                "uai_authority_ci"):
        v = metrics.get(key)
        if v:
            print(f"    {key:>22}: mean={v.get('mean'):.3f}  "
                  f"[{v.get('lo'):.3f}, {v.get('hi'):.3f}]")
    print(f"  control MAE = {metrics.get('mae_control')}")


if __name__ == "__main__":
    main()
