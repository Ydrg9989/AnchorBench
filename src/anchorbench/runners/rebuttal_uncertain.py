#!/usr/bin/env python3
"""Uncertain-judgment runner.

Builds 15 conditions per External itemspec by withholding evidence
(see ``anchorbench.data.suites.external_uncertain``) and runs single-stage
inference. The analyzer (``anchorbench.analysis.uncertain``) compares
measured shifts against an information-theoretic Bayesian-rational
baseline.

Outputs (per model):
  datasets/anchorbench_external_uncertain/promptviews_uncertain.jsonl
  results/rebuttal/uncertain/<slug>/results.jsonl
  results/rebuttal/uncertain/<slug>/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from anchorbench.data.generate import render_variant_views
from anchorbench.data.suites.external_uncertain import (
    K_LEVELS,
    _conditions_for_k,
    render_external_uncertain,
)
from anchorbench.eval.evaluator import prepare_items, run_single_stage
from anchorbench.eval.io import load_itemspecs, load_promptviews
from anchorbench.eval.metrics import compute_unified_metrics
from anchorbench.eval.runner_utils import add_backend_args, make_backend
from anchorbench.paths import DATASETS_DIR, RESULTS_DIR

log = logging.getLogger(__name__)

DEFAULT_CORE = DATASETS_DIR / "anchorbench_external_core"
DEFAULT_DATASET = DATASETS_DIR / "anchorbench_external_uncertain"
DEFAULT_OUT = RESULTS_DIR / "rebuttal/uncertain"


def _all_conditions() -> list[str]:
    out: list[str] = []
    for k in K_LEVELS:
        for full, *_ in _conditions_for_k(k):
            out.append(full)
    return out


CONDITIONS = _all_conditions()


def build_uncertain_promptviews(core_dir: Path, out_dir: Path,
                                max_items: int | None = None) -> Path:
    """Render the uncertain-evidence conditions for the committed External items."""
    return render_variant_views(core_dir, out_dir / "promptviews_uncertain.jsonl", "external",
                                render_external_uncertain, max_items=max_items)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="P5 uncertain-judgment runner")
    p.add_argument("--model_id", required=True)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--core_dir", type=Path, default=DEFAULT_CORE)
    p.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--max_items", type=int, default=None,
                   help="Smoke test cap; None = full dataset")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    add_backend_args(p, default="vllm")
    args = p.parse_args(argv)

    model_slug = args.model_id.replace("/", "_")
    out_dir = args.out_dir / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.max_items is not None:
        # A capped smoke run renders its own views under its results directory;
        # datasets/anchorbench_external_uncertain/ is the committed dataset
        # behind Table 2 and must never be overwritten with a truncated file.
        pv_path = build_uncertain_promptviews(args.core_dir, out_dir / "smoke_dataset",
                                              max_items=args.max_items)
    else:
        pv_path = args.dataset_dir / "promptviews_uncertain.jsonl"
        if not pv_path.exists():
            build_uncertain_promptviews(args.core_dir, args.dataset_dir)

    items = prepare_items(load_promptviews(pv_path), load_itemspecs(args.core_dir / "itemspecs.jsonl"),
                          None, 0, conditions=CONDITIONS)
    log.info("Loaded %d items x %d conditions = %d prompts",
             len(items), len(CONDITIONS), len(items) * len(CONDITIONS))

    backend = make_backend(args)
    records = run_single_stage(
        backend, items, out_dir / "results.jsonl", conditions=CONDITIONS,
        max_tokens=args.max_tokens, batch_size=args.batch_size,
    )

    # Compute one summary per k-level so the standard metrics machinery works.
    # The per-k records share the same baseline (uncertain_p{k}_control).
    per_k: dict[int, dict] = {}
    for k in K_LEVELS:
        prefix = f"uncertain_p{k}_"
        k_recs = [r for r in records if r.get("condition", "").startswith(prefix)]
        # Rewrite condition names within the slice to the standard set so
        # compute_unified_metrics can run.
        renamed = []
        for r in k_recs:
            rcopy = dict(r)
            rcopy["condition"] = r["condition"].replace(prefix, "")
            renamed.append(rcopy)
        if not renamed:
            continue
        m = compute_unified_metrics(renamed, baseline_condition="control")
        per_k[k] = m

    summary = {
        "model_id": args.model_id,
        "n_items": len(items),
        "k_levels": list(K_LEVELS),
        "per_k_metrics": {str(k): v for k, v in per_k.items()},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    log.info("[P5] Per-k summary written to %s/summary.json", out_dir)


if __name__ == "__main__":
    main()
