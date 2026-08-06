#!/usr/bin/env python3
"""Tool realism ablation.

Renders 8 new conditions per Tool item (plausible/irrelevant x low/high x
elicited/noisy) and combines them with the existing core Tool baseline
for analysis.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from anchorbench.data.schema import ItemSpec
from anchorbench.data.suites.tool import build_realism_promptviews
from anchorbench.eval.evaluator import run_single_stage
from anchorbench.eval.io import load_records

log = logging.getLogger(__name__)

DEFAULT_CORE = Path("datasets/anchorbench_tool_core")
DEFAULT_DATASET = Path("datasets/anchorbench_tool_p3")
DEFAULT_OUT = Path("results/rebuttal/tool_realism")
DEFAULT_CORE_RESULTS = Path("results/full_benchmark/tool")

NEW_CONDITIONS: list[str] = []
for rel in ("plausible", "irrelevant"):
    for direction in ("low", "high"):
        for tag in ("elicited", "noisy"):
            NEW_CONDITIONS.append(f"{rel}_{direction}_{tag}")


def _spec_from_dict(d: dict) -> ItemSpec:
    field_names = {f.name for f in dataclasses.fields(ItemSpec)}
    kwargs = {k: v for k, v in d.items() if k in field_names}
    return ItemSpec(**kwargs)


def build_p3_promptviews(core_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs_path = core_dir / "itemspecs.jsonl"
    out_path = out_dir / "promptviews_p3.jsonl"
    n = 0
    with open(specs_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            d = json.loads(line)
            spec = _spec_from_dict(d)
            if spec.suite != "tool":
                continue
            for pv in build_realism_promptviews(spec):
                fout.write(json.dumps(pv.__dict__, ensure_ascii=False,
                                       default=str) + "\n")
                n += 1
    log.info("[tool P3] Wrote %d promptviews to %s", n, out_path)
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
    cond_set = set(NEW_CONDITIONS)
    items = []
    for iid, conds in views_by_item.items():
        if not cond_set.issubset(conds.keys()):
            continue
        spec = specs.get(iid, {})
        item = {
            "item_id": iid, "suite": "tool",
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
        conditions=NEW_CONDITIONS,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )


def _combine_with_core(
    new_records: list[dict],
    model_slug: str,
    core_results_dir: Path,
    combined_path: Path,
) -> list[dict]:
    core_path = core_results_dir / model_slug / "results.jsonl"
    if not core_path.exists():
        log.warning("[P3] Core Tool results not found at %s", core_path)
        core = []
    else:
        core = load_records(str(core_path))
    keep = {"control", "plausible_low", "plausible_high",
            "irrelevant_low", "irrelevant_high"}
    core_subset = [r for r in core if r.get("condition") in keep]
    all_recs = list(core_subset) + list(new_records)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as fh:
        for r in all_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("[P3] Wrote combined %d records (%d core + %d new) to %s",
             len(all_recs), len(core_subset), len(new_records),
             combined_path)
    return all_recs


def _realism_curve(all_recs: list[dict]) -> dict:
    by_item_ctrl: dict[str, float] = {}
    for r in all_recs:
        if not r.get("parsed_ok"):
            continue
        if r["condition"] == "control":
            ai = r.get("answer_int")
            if ai is not None:
                by_item_ctrl[r["item_id"]] = float(ai)

    buckets: dict[str, list[float]] = {}
    for rel in ("plausible", "irrelevant"):
        for tag in ("baseline", "elicited", "noisy"):
            buckets[f"{rel}__{tag}"] = []

    for r in all_recs:
        if not r.get("parsed_ok"):
            continue
        cond = r["condition"]
        ai = r.get("answer_int")
        ctrl = by_item_ctrl.get(r["item_id"])
        anchor = r.get("anchor_value")
        if ai is None or ctrl is None or anchor is None:
            continue
        denom = abs(anchor - ctrl)
        if denom < 1e-6:
            continue
        uai = (float(ai) - ctrl) / denom * (1 if anchor > ctrl else -1)
        rel = None
        for rcand in ("plausible", "irrelevant"):
            if cond.startswith(f"{rcand}_"):
                rel = rcand
                break
        if rel is None:
            continue
        if cond in (f"{rel}_low", f"{rel}_high"):
            tag = "baseline"
        elif cond.endswith("_elicited"):
            tag = "elicited"
        elif cond.endswith("_noisy"):
            tag = "noisy"
        else:
            continue
        buckets[f"{rel}__{tag}"].append(uai)

    out: dict[str, float | None] = {}
    for k, vs in buckets.items():
        out[k] = (sum(vs) / len(vs)) if vs else None
        out[f"{k}_n"] = len(vs)
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="P3 Tool realism ablation")
    p.add_argument("--model_id", required=True)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--core_dir", type=Path, default=DEFAULT_CORE)
    p.add_argument("--core_results_dir", type=Path,
                   default=DEFAULT_CORE_RESULTS)
    p.add_argument("--p3_dataset_dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    args = p.parse_args(argv)

    pv_path = args.p3_dataset_dir / "promptviews_p3.jsonl"
    if not pv_path.exists():
        log.info("P3 promptviews not found; generating from %s", args.core_dir)
        build_p3_promptviews(args.core_dir, args.p3_dataset_dir)

    items = _build_items(pv_path, args.core_dir / "itemspecs.jsonl")
    log.info("Loaded %d items x %d NEW conditions = %d prompts",
             len(items), len(NEW_CONDITIONS),
             len(items) * len(NEW_CONDITIONS))

    model_slug = args.model_id.replace("/", "_")
    out_dir = args.out_dir / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    new_path = out_dir / "results_p3.jsonl"
    new_records = _run_inference(args, items, new_path)

    combined_path = out_dir / "results_combined.jsonl"
    all_recs = _combine_with_core(
        new_records, model_slug, args.core_results_dir, combined_path,
    )

    from anchorbench.eval.metrics import compute_unified_metrics
    metrics = compute_unified_metrics(all_recs, baseline_condition="control")
    (out_dir / "summary_combined.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )

    curve = _realism_curve(all_recs)
    (out_dir / "realism_curve.json").write_text(
        json.dumps(curve, indent=2)
    )
    log.info("[P3] Realism curve written")


if __name__ == "__main__":
    main()
