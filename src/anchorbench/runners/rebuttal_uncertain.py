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
import dataclasses
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from anchorbench.data.schema import ItemSpec
from anchorbench.data.suites.external_uncertain import (
    K_LEVELS,
    _conditions_for_k,
    render_external_uncertain,
)
from anchorbench.eval.evaluator import run_single_stage
from anchorbench.eval.metrics import compute_unified_metrics

log = logging.getLogger(__name__)

DEFAULT_CORE = Path("datasets/anchorbench_external_core")
DEFAULT_DATASET = Path("datasets/anchorbench_external_uncertain")
DEFAULT_OUT = Path("results/rebuttal/uncertain")


def _all_conditions() -> list[str]:
    out: list[str] = []
    for k in K_LEVELS:
        for full, *_ in _conditions_for_k(k):
            out.append(full)
    return out


CONDITIONS = _all_conditions()


def _spec_from_dict(d: dict) -> ItemSpec:
    field_names = {f.name for f in dataclasses.fields(ItemSpec)}
    kwargs = {k: v for k, v in d.items() if k in field_names}
    return ItemSpec(**kwargs)


def build_uncertain_promptviews(core_dir: Path, out_dir: Path,
                                max_items: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs_path = core_dir / "itemspecs.jsonl"
    out_path = out_dir / "promptviews_uncertain.jsonl"
    n_items = 0
    n_views = 0
    with open(specs_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            d = json.loads(line)
            spec = _spec_from_dict(d)
            if spec.suite != "external":
                continue
            n_items += 1
            if max_items is not None and n_items > max_items:
                break
            for pv in render_external_uncertain(spec):
                fout.write(json.dumps(pv.__dict__, ensure_ascii=False,
                                       default=str) + "\n")
                n_views += 1
    log.info("[uncertain] Wrote %d promptviews from %d items to %s",
             n_views, n_items, out_path)
    return out_path


def _build_items(promptviews_path: Path, specs_path: Path) -> list[dict]:
    from collections import defaultdict
    views_by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(promptviews_path) as f:
        for line in f:
            pv = json.loads(line)
            views_by_item[pv["item_id"]][pv["condition"]] = pv
    specs = {}
    with open(specs_path) as f:
        for line in f:
            d = json.loads(line)
            specs[d["item_id"]] = d
    items = []
    cond_set = set(CONDITIONS)
    for iid, conds in views_by_item.items():
        if not cond_set.issubset(conds.keys()):
            continue
        spec = specs.get(iid, {})
        item = {
            "item_id": iid, "suite": "external_uncertain",
            "domain": spec.get("domain"),
            "difficulty": spec.get("difficulty", "standard"),
            "y_star_evidence": spec.get(
                "y_star_evidence", spec.get("y_star")
            ),
            "y_star_theta": spec.get("y_star_theta"),
            "anchors": spec.get("anchors", {}),
            "spec": spec,
        }
        for c in cond_set:
            item[c] = conds[c]
        items.append(item)
    return items


def _run_inference(args, items: list[dict], out_path: Path) -> list[dict]:
    from anchorbench.eval.backends import HFBackend, VLLMBackend
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
        conditions=CONDITIONS,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )


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
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    args = p.parse_args(argv)

    pv_path = args.dataset_dir / "promptviews_uncertain.jsonl"
    # Always regenerate when --max_items is set so smoke caps are honored.
    if args.max_items is not None or not pv_path.exists():
        build_uncertain_promptviews(args.core_dir, args.dataset_dir,
                                    max_items=args.max_items)

    items = _build_items(pv_path, args.core_dir / "itemspecs.jsonl")
    log.info("Loaded %d items x %d conditions = %d prompts",
             len(items), len(CONDITIONS), len(items) * len(CONDITIONS))

    model_slug = args.model_id.replace("/", "_")
    out_dir = args.out_dir / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"
    records = _run_inference(args, items, res_path)

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
