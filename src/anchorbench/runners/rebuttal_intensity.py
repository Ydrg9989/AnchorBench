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
import json
import logging
from collections.abc import Callable
from pathlib import Path

from anchorbench.data.generate import render_variant_views
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
    prepare_items,
    run_history_two_stage,
    run_single_stage,
)
from anchorbench.eval.io import (
    load_itemspecs,
    load_promptviews,
    splice_core_records,
    write_records,
)
from anchorbench.eval.runner_utils import add_backend_args, make_backend
from anchorbench.paths import DATASETS_DIR, RESULTS_DIR

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
        "core_dir": DATASETS_DIR / "anchorbench_external_core",
        "dataset_dir": DATASETS_DIR / "anchorbench_external_d1",
        "out_dir": RESULTS_DIR / "rebuttal/intensity",
        "core_results_dir": RESULTS_DIR / "full_benchmark/external",
        "baseline_cond": "control",
    },
    "rag": {
        "core_dir": DATASETS_DIR / "anchorbench_rag_core",
        "dataset_dir": DATASETS_DIR / "anchorbench_rag_d1",
        "out_dir": RESULTS_DIR / "rebuttal/intensity_rag",
        "core_results_dir": RESULTS_DIR / "full_benchmark/rag",
        "baseline_cond": "control",
    },
    "history": {
        "core_dir": DATASETS_DIR / "anchorbench_history_core",
        "dataset_dir": DATASETS_DIR / "anchorbench_history_d1",
        "out_dir": RESULTS_DIR / "rebuttal/intensity_history",
        "core_results_dir": RESULTS_DIR / "full_benchmark/history",
        "baseline_cond": "control_twostage",
    },
}


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
    return render_variant_views(core_dir, out_dir / "promptviews_d1.jsonl", suite,
                                SUITE_VIEW_BUILDER[suite])


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
    add_backend_args(p, default="vllm")
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

    items = prepare_items(load_promptviews(pv_path), load_itemspecs(core_dir / "itemspecs.jsonl"),
                          None, 0, conditions=NEW_CONDITIONS)
    log.info("[%s] Loaded %d items x %d NEW conditions = %d prompts",
             args.suite, len(items), len(NEW_CONDITIONS),
             len(items) * len(NEW_CONDITIONS))

    model_slug = args.model_id.replace("/", "_")
    out_dir = out_dir_base / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    new_path = out_dir / "results_d1.jsonl"
    backend = make_backend(args)
    if args.suite == "history":
        new_records = run_history_two_stage(
            backend, items, new_path, conditions=NEW_CONDITIONS, max_tokens=args.max_tokens,
        )
    else:
        new_records = run_single_stage(
            backend, items, new_path, conditions=NEW_CONDITIONS,
            max_tokens=args.max_tokens, batch_size=args.batch_size,
        )
    log.info("[%s] D1 inference complete: %d records", args.suite, len(new_records))

    all_recs = splice_core_records(
        new_records, core_results_dir / model_slug / "results.jsonl",
        keep={baseline_cond, "control", "plausible_low", "plausible_high"},
    )
    write_records(all_recs, out_dir / "results_combined.jsonl")

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
