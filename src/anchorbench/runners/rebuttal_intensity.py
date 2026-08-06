#!/usr/bin/env python3
"""Plausible-intensity probe, extended for the cross-pathway comparison.

Renders four new conditions on top of the existing core itemspecs for the
requested suite::

    plausible_mild_low / plausible_mild_high
    plausible_strong_low / plausible_strong_high

Combined with the existing ``control`` (or ``control_twostage`` for History)
and ``plausible_low`` / ``plausible_high`` (standard intensity), this gives
a 3-point plausibility-intensity curve mild -> standard -> strong.

Suites supported:
  external (default, original D1)
  rag      (P1: anchor-slot doc body swapped for intensity preamble)
  history  (P1: intensity preamble injected before Stage-2 question)

Outputs (per suite, per model):
  results/rebuttal/intensity_<suite>/<slug>/results_d1.jsonl
  results/rebuttal/intensity_<suite>/<slug>/results_combined.jsonl
  results/rebuttal/intensity_<suite>/<slug>/summary_combined.json
  results/rebuttal/intensity_<suite>/<slug>/intensity_curve.json

External keeps its legacy output location (results/rebuttal/intensity/...)
for backward compatibility with the existing D1 artifacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from anchorbench.data.schema import ItemSpec
from anchorbench.data.suites._shared import INTENSITY_CONDITIONS
from anchorbench.data.suites.external import (
    _build_prompt as _external_build_prompt,
)
from anchorbench.data.suites.history import (
    build_intensity_promptviews as _history_intensity_views,
)
from anchorbench.data.suites.rag import (
    build_intensity_promptviews as _rag_intensity_views,
)
from anchorbench.eval.evaluator import (
    run_history_two_stage,
    run_single_stage,
)
from anchorbench.eval.io import load_records

log = logging.getLogger(__name__)

NEW_CONDITIONS = [
    "plausible_mild_low", "plausible_mild_high",
    "plausible_strong_low", "plausible_strong_high",
]

# Per-suite configuration. The ``core_dir`` is the dataset directory
# containing itemspecs.jsonl; ``core_results_dir`` is where the standard
# benchmark results.jsonl already lives for the suite (so we can splice
# control + standard-plausible into the combined intensity records).
SUITE_CONFIG = {
    "external": {
        "core_dir": Path("datasets/anchorbench_external_core"),
        "dataset_dir": Path("datasets/anchorbench_external_d1"),
        "out_dir": Path("results/rebuttal/intensity"),
        "core_results_dir": Path("results/full_benchmark/external"),
        "baseline_cond": "control",
    },
    "rag": {
        "core_dir": Path("datasets/anchorbench_rag_core"),
        "dataset_dir": Path("datasets/anchorbench_rag_d1"),
        "out_dir": Path("results/rebuttal/intensity_rag"),
        "core_results_dir": Path("results/full_benchmark/rag"),
        "baseline_cond": "control",
    },
    "history": {
        "core_dir": Path("datasets/anchorbench_history_core"),
        "dataset_dir": Path("datasets/anchorbench_history_d1"),
        "out_dir": Path("results/rebuttal/intensity_history"),
        "core_results_dir": Path("results/full_benchmark/history"),
        "baseline_cond": "control_twostage",
    },
}


def _spec_from_dict(d: dict) -> ItemSpec:
    field_names = {f.name for f in dataclasses.fields(ItemSpec)}
    kwargs = {k: v for k, v in d.items() if k in field_names}
    return ItemSpec(**kwargs)


def _build_external_intensity_views(spec: ItemSpec):
    """Render the 4 intensity conditions on External using the standard
    plausible-preamble pathway (same mechanism as the published plausible
    condition, just with mild/strong preamble pools)."""
    out = []
    for cond, rel, direction in INTENSITY_CONDITIONS:
        if cond not in NEW_CONDITIONS:
            continue
        anchor = spec.anchors.get(direction) if direction else None
        pv = _external_build_prompt(spec, cond, rel, anchor)
        out.append(pv)
    return out


SUITE_VIEW_BUILDER: dict[str, Callable[[ItemSpec], list]] = {
    "external": _build_external_intensity_views,
    "rag":      _rag_intensity_views,
    "history":  _history_intensity_views,
}


def build_d1_promptviews(suite: str, core_dir: Path, out_dir: Path) -> Path:
    """Render the 4 new intensity conditions for every existing core itemspec."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs_path = core_dir / "itemspecs.jsonl"
    out_path = out_dir / "promptviews_d1.jsonl"
    builder = SUITE_VIEW_BUILDER[suite]
    n = 0
    with open(specs_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            d = json.loads(line)
            spec = _spec_from_dict(d)
            if spec.suite != suite:
                continue
            for pv in builder(spec):
                fout.write(json.dumps(pv.__dict__, ensure_ascii=False,
                                       default=str) + "\n")
                n += 1
    log.info("[%s] Wrote %d D1 promptviews to %s", suite, n, out_path)
    return out_path


def _build_items(suite: str, promptviews_path: Path, specs_path: Path) -> list[dict]:
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

    items: list[dict] = []
    cond_set = set(NEW_CONDITIONS)
    for iid, conds in views_by_item.items():
        if not cond_set.issubset(conds.keys()):
            continue
        spec = specs.get(iid, {})
        item = {
            "item_id": iid,
            "suite": suite,
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


def _run_inference(suite: str, args, items: list[dict], out_path: Path) -> list[dict]:
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

    if suite == "history":
        return run_history_two_stage(
            backend, items, out_path,
            conditions=NEW_CONDITIONS,
            max_tokens=args.max_tokens,
        )
    return run_single_stage(
        backend, items, out_path,
        conditions=NEW_CONDITIONS,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )


def _combine_with_core(
    suite: str,
    new_records: list[dict],
    model_slug: str,
    core_results_dir: Path,
    combined_path: Path,
) -> list[dict]:
    core_path = core_results_dir / model_slug / "results.jsonl"
    if not core_path.exists():
        log.warning("[%s] Core results not found at %s; combined will only "
                    "include NEW conditions", suite, core_path)
        core = []
    else:
        core = load_records(str(core_path))
    keep = {SUITE_CONFIG[suite]["baseline_cond"], "control",
            "plausible_low", "plausible_high"}
    core_subset = [r for r in core if r.get("condition") in keep]
    all_recs = list(core_subset) + list(new_records)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as fh:
        for r in all_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("[%s] Wrote combined %d records (%d core_subset + %d new) to %s",
             suite, len(all_recs), len(core_subset), len(new_records),
             combined_path)
    return all_recs


def _compute_intensity_curve(
    all_recs: list[dict], baseline_cond: str,
) -> dict:
    """Per-intensity mean UAI vs the per-item baseline.

    Tries ``baseline_cond`` first; if no records match (e.g. History
    full_benchmark only has single-stage ``control``), falls back to
    ``control``.
    """
    def _collect(cond: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in all_recs:
            if not r.get("parsed_ok"):
                continue
            if r["condition"] == cond:
                ai = r.get("answer_int")
                if ai is not None:
                    out[r["item_id"]] = float(ai)
        return out

    by_item_ctrl = _collect(baseline_cond)
    if not by_item_ctrl and baseline_cond != "control":
        log.info("No %s baseline; falling back to 'control'", baseline_cond)
        by_item_ctrl = _collect("control")

    by_cond: dict[str, list[float]] = {c: [] for c in (
        "plausible_low", "plausible_high",
        "plausible_mild_low", "plausible_mild_high",
        "plausible_strong_low", "plausible_strong_high",
    )}
    for r in all_recs:
        if not r.get("parsed_ok"):
            continue
        cond = r["condition"]
        if cond not in by_cond:
            continue
        ai = r.get("answer_int")
        ctrl = by_item_ctrl.get(r["item_id"])
        anchor = r.get("anchor_value")
        if ai is None or ctrl is None or anchor is None:
            continue
        denom = abs(anchor - ctrl)
        if denom < 1e-6:
            continue
        uai = (float(ai) - ctrl) / denom * (1 if anchor > ctrl else -1)
        by_cond[cond].append(uai)

    intensity: dict[str, float | None] = {}
    for k in ("plausible_mild", "plausible", "plausible_strong"):
        vals = by_cond[f"{k}_low"] + by_cond[f"{k}_high"]
        intensity[k] = (sum(vals) / len(vals)) if vals else None
        intensity[f"{k}_n"] = len(vals)
    return intensity


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Plausibility-intensity probe (D1/P1)")
    p.add_argument("--model_id", required=True)
    p.add_argument("--suite", choices=list(SUITE_CONFIG.keys()),
                   default="external")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="override; defaults to SUITE_CONFIG[suite].out_dir")
    p.add_argument("--core_dir", type=Path, default=None,
                   help="override; defaults to SUITE_CONFIG[suite].core_dir")
    p.add_argument("--core_results_dir", type=Path, default=None,
                   help="override; defaults to SUITE_CONFIG[suite].core_results_dir")
    p.add_argument("--d1_dataset_dir", type=Path, default=None,
                   help="override; defaults to SUITE_CONFIG[suite].dataset_dir")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    args = p.parse_args(argv)

    cfg = SUITE_CONFIG[args.suite]
    core_dir = args.core_dir or cfg["core_dir"]
    out_dir_base = args.out_dir or cfg["out_dir"]
    core_results_dir = args.core_results_dir or cfg["core_results_dir"]
    d1_dataset_dir = args.d1_dataset_dir or cfg["dataset_dir"]
    baseline_cond = cfg["baseline_cond"]

    pv_path = d1_dataset_dir / "promptviews_d1.jsonl"
    if not pv_path.exists():
        log.info("[%s] D1 promptviews not found; generating from %s",
                 args.suite, core_dir)
        build_d1_promptviews(args.suite, core_dir, d1_dataset_dir)

    items = _build_items(args.suite, pv_path, core_dir / "itemspecs.jsonl")
    log.info("[%s] Loaded %d items x %d NEW conditions = %d prompts",
             args.suite, len(items), len(NEW_CONDITIONS),
             len(items) * len(NEW_CONDITIONS))

    model_slug = args.model_id.replace("/", "_")
    out_dir = out_dir_base / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    new_path = out_dir / "results_d1.jsonl"
    new_records = _run_inference(args.suite, args, items, new_path)
    log.info("[%s] D1 inference complete: %d records",
             args.suite, len(new_records))

    combined_path = out_dir / "results_combined.jsonl"
    all_recs = _combine_with_core(
        args.suite, new_records, model_slug,
        core_results_dir, combined_path,
    )

    from anchorbench.eval.metrics import compute_unified_metrics
    metrics = compute_unified_metrics(all_recs, baseline_condition=baseline_cond)
    (out_dir / "summary_combined.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )

    intensity = _compute_intensity_curve(all_recs, baseline_cond)
    (out_dir / "intensity_curve.json").write_text(
        json.dumps(intensity, indent=2)
    )
    log.info("[%s] Intensity curve: mild=%s, standard=%s, strong=%s",
             args.suite,
             intensity.get("plausible_mild"),
             intensity.get("plausible"),
             intensity.get("plausible_strong"))


if __name__ == "__main__":
    main()
