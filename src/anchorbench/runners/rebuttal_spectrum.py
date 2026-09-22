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
import asyncio
import json
import logging
import os
from pathlib import Path

from anchorbench.eval.evaluator import (
    build_record,
    parse_response,
    run_single_stage,
)
from anchorbench.eval.io import load_itemspecs, load_promptviews, load_records
from anchorbench.eval.metrics import compute_extended_metrics

log = logging.getLogger(__name__)

ABLATION_CONDITIONS = [
    "placebo_low", "placebo_high",
    "authority_low", "authority_high",
]

DEFAULT_OUT = Path("results/rebuttal/spectrum")
DEFAULT_DATASET = Path("datasets/anchorbench_external_core")
DEFAULT_CORE_RESULTS = Path("results/full_benchmark/external")


def _build_items(views: dict, specs: dict) -> list[dict]:
    """Build items from ablation promptviews (subset of conditions)."""
    items: list[dict] = []
    cond_set = set(ABLATION_CONDITIONS)
    for item_id, cond_views in views.items():
        if not cond_set.issubset(cond_views.keys()):
            continue
        spec = specs.get(item_id, {})
        item: dict = {
            "item_id": item_id,
            "suite": cond_views[next(iter(cond_set))]["suite"],
            "domain": cond_views[next(iter(cond_set))]["domain"],
            "difficulty": spec.get("difficulty", "standard"),
            "y_star_evidence": spec.get(
                "y_star_evidence", spec.get("y_star")
            ),
            "y_star_theta": spec.get("y_star_theta"),
            "anchors": spec.get("anchors", {}),
            "spec": spec,
        }
        for c in cond_set:
            item[c] = cond_views[c]
        items.append(item)
    return items


def _run_open_weight(args, items: list[dict], out_path: Path) -> list[dict]:
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

    return run_single_stage(
        backend, items, out_path,
        conditions=ABLATION_CONDITIONS,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )


def _run_api(args, items: list[dict], out_path: Path) -> list[dict]:
    """OpenRouter API path mirroring runners.api.run_suite_api."""
    from anchorbench.inference.async_api import AsyncOpenRouterClient

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set")
        raise SystemExit(2)

    async def go():
        client = AsyncOpenRouterClient(
            api_key=api_key, max_concurrent=args.max_concurrent,
        )
        task_list = [(i, c, item[c]) for i, item in enumerate(items)
                     for c in ABLATION_CONDITIONS]
        prompts = [pv.get("prompt_text", "") for _, _, pv in task_list]
        results = await client.query_batch(
            args.model_id, prompts,
            max_tokens=args.max_tokens, temperature=0.0,
        )
        records: list[dict] = []
        with open(out_path, "w", encoding="utf-8") as fh:
            for (i, cond, pv), api_result in zip(task_list, results):
                raw = api_result.get("raw_text", "")
                prompt_text = pv.get("prompt_text", "")
                answer, parsed_ok, strategy = parse_response(raw, prompt_text)
                rec = build_record(
                    args.model_id, items[i], cond, pv,
                    answer, parsed_ok, strategy, raw,
                )
                rec["api_usage"] = api_result.get("usage", {})
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        await client.close()
        return records

    return asyncio.run(go())


def _combine_with_core(
    new_records: list[dict],
    model_slug: str,
    core_results_dir: Path,
    combined_path: Path,
) -> list[dict]:
    """Concatenate the new ablation records with the existing core results
    (control + irrelevant + plausible) so extended metrics can be computed.
    """
    core_path = core_results_dir / model_slug / "results.jsonl"
    if not core_path.exists():
        log.warning(
            "Core results not found at %s; combined results will only "
            "contain placebo/authority", core_path,
        )
        core = []
    else:
        core = load_records(str(core_path))

    all_recs = list(core) + list(new_records)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as fh:
        for r in all_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(
        "Wrote combined %d records (%d core + %d new) to %s",
        len(all_recs), len(core), len(new_records), combined_path,
    )
    return all_recs


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
    views = load_promptviews(pv_path)
    specs = load_itemspecs(spec_path)
    items = _build_items(views, specs)
    if args.max_items and args.max_items < len(items):
        import numpy as np
        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(len(items))[:args.max_items]
        items = [items[i] for i in idx]
    log.info("Loaded %d items x %d ablation conditions = %d prompts",
             len(items), len(ABLATION_CONDITIONS),
             len(items) * len(ABLATION_CONDITIONS))

    model_slug = args.model_id.replace("/", "_")
    out_dir = args.out_dir / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    abl_path = out_dir / "results_ablation.jsonl"

    if args.backend == "openrouter":
        new_records = _run_api(args, items, abl_path)
    else:
        new_records = _run_open_weight(args, items, abl_path)

    log.info("Inference complete: %d records", len(new_records))

    combined_path = out_dir / "results_extended.jsonl"
    all_recs = _combine_with_core(
        new_records, model_slug, args.core_results_dir, combined_path,
    )

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
