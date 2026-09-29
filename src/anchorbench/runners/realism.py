"""RAG and Tool realism ablations (Appendix tables tab:rag_realism and tab:tool_realism).

Both experiments render extra anchored conditions on the committed core
items, run them, splice in the published control and standard-anchor
records, and summarise a per-variant UAI curve. They differ only in the
suite's renderer and variant names, which the SUITES table carries.

    python -m anchorbench.runners.realism --suite rag  --model_id Qwen/Qwen2.5-7B-Instruct
    python -m anchorbench.runners.realism --suite tool --model_id Qwen/Qwen2.5-7B-Instruct

Outputs (per model, under results/rebuttal/<suite>_realism/<slug>/):
  results_<tag>.jsonl, results_combined.jsonl, summary_combined.json,
  realism_curve.json; the rendered prompt views go to
  datasets/anchorbench_<suite>_<tag>/promptviews_<tag>.jsonl once.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

from anchorbench.data.generate import render_variant_views
from anchorbench.data.schema import ItemSpec, PromptView
from anchorbench.data.suites import rag, tool
from anchorbench.eval.evaluator import prepare_items, run_single_stage
from anchorbench.eval.io import (
    load_itemspecs,
    load_promptviews,
    splice_core_records,
    write_records,
)
from anchorbench.eval.metrics import compute_unified_metrics, item_uai
from anchorbench.eval.runner_utils import add_backend_args, make_backend
from anchorbench.paths import RESULTS_DIR

log = logging.getLogger(__name__)

CORE_CONDITIONS = {"control", "plausible_low", "plausible_high",
                   "irrelevant_low", "irrelevant_high"}


class Suite:
    def __init__(self, tag: str, variants: tuple[str, ...],
                 build: Callable[[ItemSpec], list[PromptView]]) -> None:
        self.tag = tag
        self.variants = variants
        self.build = build

    def conditions(self) -> list[str]:
        return [f"{rel}_{direction}_{v}"
                for rel in ("plausible", "irrelevant")
                for direction in ("low", "high")
                for v in self.variants]


SUITES: dict[str, Suite] = {
    # anchor document at rank 1 / rank 5 / rank 5 with two distractors and scores
    "rag": Suite("p2", ("rank1", "rank5", "rank5_distract"), rag.build_realism_promptviews),
    # model-planned tool call / tool response wrapped in metadata
    "tool": Suite("p3", ("elicited", "noisy"), tool.build_realism_promptviews),
}


def build_promptviews(suite: str, core_dir: Path, out_dir: Path) -> Path:
    """Render the realism conditions for every committed core item of ``suite``."""
    cfg = SUITES[suite]
    return render_variant_views(core_dir, out_dir / f"promptviews_{cfg.tag}.jsonl", suite, cfg.build)


def realism_curve(records: list[dict], variants: tuple[str, ...]) -> dict:
    """Mean UAI per (relevance, variant) over the spliced records, with the
    published conditions as the ``baseline`` variant. UAI is the paper's
    (metrics.item_uai, epsilon = 3); see RECONCILIATION D11 for the curves
    the rebuttal was computed with.
    """
    ctrl: dict[str, float] = {}
    for r in records:
        if r["parsed_ok"] and r["condition"] == "control" and r["answer_int"] is not None:
            ctrl[r["item_id"]] = float(r["answer_int"])

    tags = ("baseline",) + variants
    buckets: dict[str, list[float]] = {f"{rel}__{t}": [] for rel in ("plausible", "irrelevant") for t in tags}
    for r in records:
        if not r["parsed_ok"]:
            continue
        cond, ai, y_ctrl, anchor = r["condition"], r["answer_int"], ctrl.get(r["item_id"]), r["anchor_value"]
        if ai is None or y_ctrl is None or anchor is None:
            continue
        uai = item_uai(float(ai), y_ctrl, anchor)
        if uai is None:
            continue
        rel = next((x for x in ("plausible", "irrelevant") if cond.startswith(f"{x}_")), None)
        if rel is None:
            continue
        if cond in (f"{rel}_low", f"{rel}_high"):
            tag = "baseline"
        else:
            tag = next((v for v in variants if cond.endswith(f"_{v}")), None)
            if tag is None:
                continue
        buckets[f"{rel}__{tag}"].append(uai)

    out: dict[str, float | int | None] = {}
    for k, vs in buckets.items():
        out[k] = (sum(vs) / len(vs)) if vs else None
        out[f"{k}_n"] = len(vs)
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="RAG / Tool realism ablation")
    p.add_argument("--suite", choices=sorted(SUITES), required=True)
    p.add_argument("--model_id", required=True)
    p.add_argument("--out_dir", type=Path, default=None,
                   help="default: results/rebuttal/<suite>_realism")
    p.add_argument("--core_dir", type=Path, default=None,
                   help="default: datasets/anchorbench_<suite>_core")
    p.add_argument("--core_results_dir", type=Path, default=None,
                   help="default: results/full_benchmark/<suite>")
    p.add_argument("--dataset_dir", type=Path, default=None,
                   help="where the rendered views live; default: datasets/anchorbench_<suite>_<tag>")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    add_backend_args(p, default="vllm")
    args = p.parse_args(argv)

    suite, cfg = args.suite, SUITES[args.suite]
    core_dir = args.core_dir or Path(f"datasets/anchorbench_{suite}_core")
    dataset_dir = args.dataset_dir or Path(f"datasets/anchorbench_{suite}_{cfg.tag}")
    core_results_dir = args.core_results_dir or RESULTS_DIR / f"full_benchmark/{suite}"
    out_root = args.out_dir or RESULTS_DIR / f"rebuttal/{suite}_realism"
    conditions = cfg.conditions()

    pv_path = dataset_dir / f"promptviews_{cfg.tag}.jsonl"
    if not pv_path.exists():
        log.info("[%s realism] promptviews not found; generating from %s", suite, core_dir)
        build_promptviews(suite, core_dir, dataset_dir)

    items = prepare_items(load_promptviews(pv_path), load_itemspecs(core_dir / "itemspecs.jsonl"),
                          None, 0, conditions=conditions)
    log.info("[%s realism] %d items x %d conditions", suite, len(items), len(conditions))

    model_slug = args.model_id.replace("/", "_")
    out_dir = out_root / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args)
    new_records = run_single_stage(
        backend, items, out_dir / f"results_{cfg.tag}.jsonl",
        conditions=conditions, max_tokens=args.max_tokens, batch_size=args.batch_size,
    )

    records = splice_core_records(
        new_records, core_results_dir / model_slug / "results.jsonl", keep=CORE_CONDITIONS,
    )
    write_records(records, out_dir / "results_combined.jsonl")
    metrics = compute_unified_metrics(records, baseline_condition="control")
    (out_dir / "summary_combined.json").write_text(json.dumps(metrics, indent=2, default=str))
    curve = realism_curve(records, cfg.variants)
    (out_dir / "realism_curve.json").write_text(json.dumps(curve, indent=2))
    log.info("[%s realism] curve written to %s", suite, out_dir / "realism_curve.json")


if __name__ == "__main__":
    main()
