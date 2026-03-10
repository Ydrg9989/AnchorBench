#!/usr/bin/env python3
"""Run inference on External v2 dataset (5 conditions, no mitigation) and evaluate.

Usage:
    PYTHONPATH=src python scripts/eval/run_external_v2.py \\
        --promptviews datasets/anchorbench_v2_external_pilot/promptviews.jsonl \\
        --itemspecs datasets/anchorbench_v2_external_pilot/itemspecs.jsonl \\
        --model_id Qwen/Qwen2.5-1.5B-Instruct \\
        --out_dir results/external_v2_pilot \\
        [--max_items 30]

Outputs:
    - results.jsonl (one row per item x condition)
    - summary.json (accuracy, NAI by relevance, discrimination ratio)

Parse rate: For CoT models (e.g. Llama 3.2), use --llm_fallback and --max_tokens 512
  (default). To avoid long reasoning and get a clear final line, add
  --request_final_line so the prompt asks for "Answer: [number]" on the last line.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mitigation_eval.runner import (
    HFRunner,
    LLMFallbackExtractor,
    load_promptviews,
    load_itemspecs,
    parse_answer_int,
    _parse_with_fallback,
)

log = logging.getLogger(__name__)

V2_CONDITIONS = [
    "control",
    "irrelevant_low",
    "irrelevant_high",
    "plausible_low",
    "plausible_high",
]


def prepare_items_v2(views: dict, specs: dict, max_items: int | None, seed: int) -> list[dict]:
    """Build list of items with all 5 v2 conditions."""
    items = []
    for item_id, cond_views in views.items():
        if set(cond_views.keys()) != set(V2_CONDITIONS):
            continue
        spec = specs.get(item_id, {})
        item = {
            "item_id": item_id,
            "suite": cond_views["control"]["suite"],
            "domain": cond_views["control"]["domain"],
            "difficulty": spec.get("difficulty", "standard"),
            "y_star_evidence": spec.get("y_star_evidence", spec.get("y_star")),
            "y_star_theta": spec.get("y_star_theta"),
            "anchors": spec.get("anchors", {}),
        }
        for c in V2_CONDITIONS:
            item[c] = cond_views[c]
        items.append(item)

    if max_items and max_items < len(items):
        rng = np.random.RandomState(seed)
        rng.shuffle(items)
        items = items[:max_items]

    return items


def run_inference(
    runner: HFRunner,
    items: list[dict],
    out_path: Path,
    max_tokens: int = 64,
    batch_size: int = 8,
    fallback_extractor: LLMFallbackExtractor | None = None,
    request_final_line: bool = False,
    request_reasoning: bool = False,
) -> list[dict]:
    """Run model on every item x condition; write results and return records."""
    records = []
    total = len(items) * len(V2_CONDITIONS)
    done = 0
    use_fallback = fallback_extractor is not None

    # Optional suffix: request clear final line and/or full reasoning (for sanity-check raw_text)
    final_line_suffix = ""
    if request_reasoning:
        final_line_suffix = (
            "\n\nShow your reasoning step by step, then give your final numeric estimate (0-100) at the end."
        )
    if request_final_line:
        final_line_suffix += (
            "\n\nPut your final numeric estimate (0-100) on the last line only, in this exact form: Answer: [number]"
        )

    with open(out_path, "w", encoding="utf-8") as fh:
        for item in items:
            for cond in V2_CONDITIONS:
                pv = item[cond]
                prompt = pv["prompt_text"] + final_line_suffix

                try:
                    raw = runner.generate(prompt, max_tokens=max_tokens)
                except Exception as e:
                    log.warning("Failed %s/%s: %s", item["item_id"], cond, e)
                    raw = f"ERROR: {e}"

                if use_fallback:
                    answer, parsed_ok, parse_strategy = _parse_with_fallback(
                        raw, prompt, is_cot=False, fallback=fallback_extractor
                    )
                else:
                    answer, parsed_ok = parse_answer_int(raw, prompt)
                    parse_strategy = "regex" if parsed_ok else "failed"
                assert parse_strategy in ("regex", "llm_fallback", "failed")

                rec = {
                    "model_id": runner.model_id,
                    "item_id": item["item_id"],
                    "suite": item["suite"],
                    "domain": item["domain"],
                    "difficulty": item["difficulty"],
                    "condition": cond,
                    "anchor_relevance": pv.get("anchor_relevance", "none"),
                    "anchor_value": pv.get("anchor_value"),
                    "y_star_evidence": item["y_star_evidence"],
                    "y_star_theta": item["y_star_theta"],
                    "answer_int": answer,
                    "parsed_ok": parsed_ok,
                    "parse_strategy": parse_strategy,
                    "raw_text": raw,
                }
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

                done += 1
                if done % 50 == 0:
                    log.info("Inference %d/%d", done, total)

    return records


def compute_metrics(records: list[dict]) -> dict:
    """Compute accuracy, NAI by relevance, and discrimination ratio."""
    by_item: dict[str, dict[str, int | None]] = {}
    for r in records:
        if not r["parsed_ok"] or r["answer_int"] is None:
            continue
        iid = r["item_id"]
        if iid not in by_item:
            by_item[iid] = {}
        by_item[iid][r["condition"]] = r["answer_int"]

    # Accuracy vs y_star_evidence (overall and per condition)
    mae_list = []
    mae_by_cond = {c: [] for c in V2_CONDITIONS}
    for r in records:
        if not r["parsed_ok"] or r["answer_int"] is None:
            continue
        y = r.get("y_star_evidence")
        if y is None:
            continue
        mae_list.append(abs(r["answer_int"] - y))
        mae_by_cond[r["condition"]].append(abs(r["answer_int"] - y))

    n_valid = len(mae_list)
    n_total = len(records)
    parse_rate = n_valid / n_total if n_total else 0.0
    mae_mean = float(np.mean(mae_list)) if mae_list else None
    accuracy_10 = sum(1 for e in mae_list if e <= 10) / len(mae_list) if mae_list else None

    # NAI per anchored condition (normalized by distance from gold)
    nai_irrelevant_low = []
    nai_irrelevant_high = []
    nai_plausible_low = []
    nai_plausible_high = []

    for iid, conds in by_item.items():
        control_val = conds.get("control")
        if control_val is None:
            continue

        # Get y_star_evidence and anchor_value from records
        y_star = None
        anchor_by_cond = {}
        for r in records:
            if r["item_id"] != iid:
                continue
            if r.get("y_star_evidence") is not None:
                y_star = r["y_star_evidence"]
            if r.get("anchor_value") is not None:
                anchor_by_cond[r["condition"]] = r["anchor_value"]
        if y_star is None:
            continue

        for cond in ("irrelevant_low", "irrelevant_high", "plausible_low", "plausible_high"):
            anchored_response = conds.get(cond)
            if anchored_response is None:
                continue
            anchor_val = anchor_by_cond.get(cond)
            if anchor_val is None:
                continue
            denom = anchor_val - y_star
            if abs(denom) < 1:
                continue
            nai = (anchored_response - control_val) / denom
            if cond == "irrelevant_low":
                nai_irrelevant_low.append(nai)
            elif cond == "irrelevant_high":
                nai_irrelevant_high.append(nai)
            elif cond == "plausible_low":
                nai_plausible_low.append(nai)
            elif cond == "plausible_high":
                nai_plausible_high.append(nai)

    def mean_or_none(x):
        return float(np.mean(x)) if x else None

    nai_irr = mean_or_none(nai_irrelevant_low + nai_irrelevant_high)
    nai_plas = mean_or_none(nai_plausible_low + nai_plausible_high)

    discrimination = None
    if nai_plas is not None and abs(nai_plas) > 1e-6 and nai_irr is not None:
        discrimination = 1.0 - (nai_irr / nai_plas)

    return {
        "n_items": len(by_item),
        "n_records": n_total,
        "parse_rate": round(parse_rate, 4),
        "mae_overall": round(mae_mean, 2) if mae_mean is not None else None,
        "accuracy_within_10": round(accuracy_10, 4) if accuracy_10 is not None else None,
        "mae_by_condition": {
            c: round(float(np.mean(mae_by_cond[c])), 2) if mae_by_cond[c] else None
            for c in V2_CONDITIONS
        },
        "nai_irrelevant_low": round(mean_or_none(nai_irrelevant_low), 4) if nai_irrelevant_low else None,
        "nai_irrelevant_high": round(mean_or_none(nai_irrelevant_high), 4) if nai_irrelevant_high else None,
        "nai_plausible_low": round(mean_or_none(nai_plausible_low), 4) if nai_plausible_low else None,
        "nai_plausible_high": round(mean_or_none(nai_plausible_high), 4) if nai_plausible_high else None,
        "nai_irrelevant_mean": round(nai_irr, 4) if nai_irr is not None else None,
        "nai_plausible_mean": round(nai_plas, 4) if nai_plas is not None else None,
        "discrimination_ratio": round(discrimination, 4) if discrimination is not None else None,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run External v2 inference and evaluation")
    p.add_argument("--promptviews", type=Path, required=True)
    p.add_argument("--itemspecs", type=Path, required=True)
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out_dir", type=Path, default=Path("results/external_v2_pilot"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512,
                   help="Max new tokens per reply (512 for CoT; reduce if using --request_final_line)")
    p.add_argument("--request_final_line", action="store_true",
                   help="Append instruction to request 'Answer: [number]' on last line (reduces needed tokens)")
    p.add_argument("--request_reasoning", action="store_true",
                   help="Ask model to show step-by-step reasoning so raw_text contains full generation (for sanity check)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--llm_fallback", action="store_true",
                   help="Use small HF model to extract answer when regex fails (improves parse rate for CoT models)")
    p.add_argument("--fallback_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HF model for fallback extraction (used with --llm_fallback)")
    p.add_argument("--fallback_device", type=str, default="auto",
                   help="Device for fallback model (auto=GPU when available; use cpu to avoid GPU contention)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset...")
    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items_v2(views, specs, args.max_items, args.seed)
    log.info("Loaded %d items (%d prompts)", len(items), len(items) * len(V2_CONDITIONS))

    if not items:
        log.error("No v2 items found (need 5 conditions per item)")
        sys.exit(1)

    fallback = None
    if args.llm_fallback:
        log.info("Loading LLM fallback extractor: %s on %s", args.fallback_model, args.fallback_device)
        fallback = LLMFallbackExtractor(
            model_id=args.fallback_model,
            device=args.fallback_device,
            dtype="bfloat16",
        )

    log.info("Loading model %s...", args.model_id)
    runner = HFRunner(args.model_id, device=args.device, dtype=args.dtype)

    results_path = args.out_dir / "results.jsonl"
    log.info("Running inference -> %s (llm_fallback=%s)", results_path, bool(fallback))
    records = run_inference(
        runner, items, results_path,
        max_tokens=args.max_tokens,
        fallback_extractor=fallback,
        request_final_line=getattr(args, "request_final_line", False),
        request_reasoning=getattr(args, "request_reasoning", False),
    )

    metrics = compute_metrics(records)
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Summary -> %s", summary_path)

    print("\n--- External v2 evaluation summary ---")
    print(f"Model:           {args.model_id}")
    print(f"Items:           {metrics['n_items']}")
    print(f"Parse rate:      {metrics['parse_rate']:.2%}")
    print(f"MAE (overall):   {metrics['mae_overall']}")
    print(f"Accuracy (±10):  {metrics['accuracy_within_10']}")
    print(f"NAI irrelevant:  {metrics['nai_irrelevant_mean']}")
    print(f"NAI plausible:   {metrics['nai_plausible_mean']}")
    print(f"Discrimination:  {metrics['discrimination_ratio']}")
    print()


if __name__ == "__main__":
    main()
