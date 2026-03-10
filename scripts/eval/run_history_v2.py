#!/usr/bin/env python3
"""Run inference on History v2 dataset (5 conditions, two-stage for non-control) and evaluate.

Usage:
    PYTHONPATH=src python scripts/eval/run_history_v2.py \\
        --promptviews datasets/anchorbench_v2_history_pilot/promptviews.jsonl \\
        --itemspecs datasets/anchorbench_v2_history_pilot/itemspecs.jsonl \\
        --model_id Qwen/Qwen2.5-1.5B-Instruct \\
        --out_dir results/history_v2_pilot

Outputs:
    - results.jsonl (one row per item x condition; two-stage conditions include stage1_answer, stage1_raw_text)
    - summary.json (ACR/RR for plausible; shift and toward-anchor rate for irrelevant; MAE/accuracy)

Control is single-stage; plausible_low/high and irrelevant_low/high run Stage 1 then Stage 2 (multi-turn).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

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

HISTORY_V2_CONDITIONS = [
    "control",
    "plausible_low",
    "plausible_high",
    "irrelevant_low",
    "irrelevant_high",
]


def prepare_items_history_v2(
    views: dict, specs: dict, max_items: int | None, seed: int
) -> list[dict]:
    """Build list of items with all 5 History v2 conditions."""
    items = []
    for item_id, cond_views in views.items():
        if set(cond_views.keys()) != set(HISTORY_V2_CONDITIONS):
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
        for c in HISTORY_V2_CONDITIONS:
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
    max_tokens: int = 512,
    fallback_extractor: LLMFallbackExtractor | None = None,
    request_final_line: bool = False,
) -> list[dict]:
    """Run model: single-stage for control; two-stage (Stage 1 -> Stage 2) for others."""
    records = []
    total = len(items) * len(HISTORY_V2_CONDITIONS)
    done = 0
    use_fallback = fallback_extractor is not None

    final_line_suffix = ""
    if request_final_line:
        final_line_suffix = (
            "\n\nPut your final numeric estimate (0-100) on the last line only, "
            "in this exact form: Answer: [number]"
        )

    with open(out_path, "w", encoding="utf-8") as fh:
        for item in items:
            for cond in HISTORY_V2_CONDITIONS:
                pv = item[cond]
                comp = pv.get("prompt_components", {})

                if cond == "control":
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

                    rec = {
                        "model_id": runner.model_id,
                        "item_id": item["item_id"],
                        "suite": item["suite"],
                        "domain": item["domain"],
                        "difficulty": item["difficulty"],
                        "condition": cond,
                        "anchor_relevance": pv.get("anchor_relevance", "none"),
                        "anchor_value": None,
                        "y_star_evidence": item["y_star_evidence"],
                        "y_star_theta": item["y_star_theta"],
                        "answer_int": answer,
                        "parsed_ok": parsed_ok,
                        "parse_strategy": parse_strategy,
                        "raw_text": raw,
                        "stage1_answer": None,
                        "stage1_raw_text": None,
                    }
                else:
                    stage1_msg = comp.get("stage1_user_message", "")
                    stage2_msg = comp.get("stage2_user_message", "")
                    if not stage1_msg or not stage2_msg:
                        rec = {
                            "model_id": runner.model_id,
                            "item_id": item["item_id"],
                            "suite": item["suite"],
                            "domain": item["domain"],
                            "difficulty": item["difficulty"],
                            "condition": cond,
                            "anchor_relevance": pv.get("anchor_relevance", "none"),
                            "anchor_value": None,
                            "y_star_evidence": item["y_star_evidence"],
                            "y_star_theta": item["y_star_theta"],
                            "answer_int": None,
                            "parsed_ok": False,
                            "parse_strategy": "failed",
                            "raw_text": "MISSING_STAGE_COMPONENTS",
                            "stage1_answer": None,
                            "stage1_raw_text": None,
                        }
                        records.append(rec)
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        done += 1
                        continue

                    stage1_prompt = stage1_msg + final_line_suffix
                    try:
                        stage1_raw = runner.generate(
                            stage1_prompt, max_tokens=max_tokens
                        )
                    except Exception as e:
                        log.warning("Failed %s/%s stage1: %s", item["item_id"], cond, e)
                        stage1_raw = f"ERROR: {e}"
                    if use_fallback:
                        stage1_answer, s1_ok, _ = _parse_with_fallback(
                            stage1_raw, stage1_prompt,
                            is_cot=False, fallback=fallback_extractor
                        )
                    else:
                        stage1_answer, s1_ok = parse_answer_int(
                            stage1_raw, stage1_prompt
                        )

                    messages = [
                        {"role": "user", "content": stage1_msg},
                        {"role": "assistant", "content": stage1_raw},
                        {"role": "user", "content": stage2_msg + final_line_suffix},
                    ]
                    try:
                        stage2_raw = runner.generate_chat(
                            messages, max_tokens=max_tokens
                        )
                    except Exception as e:
                        log.warning("Failed %s/%s stage2: %s", item["item_id"], cond, e)
                        stage2_raw = f"ERROR: {e}"
                    if use_fallback:
                        stage2_answer, parsed_ok, parse_strategy = _parse_with_fallback(
                            stage2_raw, messages[-1]["content"],
                            is_cot=False, fallback=fallback_extractor
                        )
                    else:
                        stage2_answer, parsed_ok = parse_answer_int(
                            stage2_raw, messages[-1]["content"]
                        )
                        parse_strategy = "regex" if parsed_ok else "failed"

                    rec = {
                        "model_id": runner.model_id,
                        "item_id": item["item_id"],
                        "suite": item["suite"],
                        "domain": item["domain"],
                        "difficulty": item["difficulty"],
                        "condition": cond,
                        "anchor_relevance": pv.get("anchor_relevance", "none"),
                        "anchor_value": stage1_answer,
                        "y_star_evidence": item["y_star_evidence"],
                        "y_star_theta": item["y_star_theta"],
                        "answer_int": stage2_answer,
                        "parsed_ok": parsed_ok,
                        "parse_strategy": parse_strategy,
                        "raw_text": stage2_raw,
                        "stage1_answer": stage1_answer,
                        "stage1_raw_text": stage1_raw,
                    }
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                if done % 50 == 0:
                    log.info("Inference %d/%d", done, total)

    return records


def compute_metrics(records: list[dict]) -> dict:
    """Compute ACR/RR for plausible; shift and toward-anchor for irrelevant; MAE/accuracy."""
    by_item: dict[str, dict[str, int | None]] = {}
    by_item_full: dict[str, dict] = {}
    for r in records:
        iid = r["item_id"]
        if iid not in by_item_full:
            by_item_full[iid] = {}
        by_item_full[iid][r["condition"]] = r
        if not r["parsed_ok"] or r["answer_int"] is None:
            continue
        if iid not in by_item:
            by_item[iid] = {}
        by_item[iid][r["condition"]] = r["answer_int"]

    n_total = len(records)
    n_valid = sum(1 for r in records if r["parsed_ok"] and r["answer_int"] is not None)
    parse_rate = n_valid / n_total if n_total else 0.0

    # Stage 2 MAE vs y_star_evidence (all conditions)
    mae_list = []
    mae_by_cond = {c: [] for c in HISTORY_V2_CONDITIONS}
    for r in records:
        if not r["parsed_ok"] or r["answer_int"] is None:
            continue
        y = r.get("y_star_evidence")
        if y is None:
            continue
        mae_list.append(abs(r["answer_int"] - y))
        mae_by_cond[r["condition"]].append(abs(r["answer_int"] - y))
    mae_mean = float(np.mean(mae_list)) if mae_list else None
    accuracy_10 = (
        sum(1 for e in mae_list if e <= 10) / len(mae_list) if mae_list else None
    )

    def mean_or_none(x):
        return float(np.mean(x)) if x else None

    # Plausible: ACR and RR (only where stage1 and stage2 and y_star valid)
    acr_list = []
    rr_list = []
    for iid, conds in by_item_full.items():
        control_val = conds.get("control", {}).get("answer_int")
        y_star = conds.get("control", {}).get("y_star_evidence")
        if control_val is None or y_star is None:
            continue
        for cond in ("plausible_low", "plausible_high"):
            r = conds.get(cond, {})
            stage1 = r.get("stage1_answer")
            stage2 = r.get("answer_int")
            if stage1 is None or stage2 is None:
                continue
            denom = y_star - stage1
            if abs(denom) >= 1:
                acr = (stage2 - stage1) / denom
                acr_list.append(acr)
            err1 = abs(stage1 - y_star)
            err2 = abs(stage2 - y_star)
            if err1 >= 1:
                rr = 1.0 - err2 / err1
                rr_list.append(rr)

    acr_mean = mean_or_none(acr_list)
    rr_mean = mean_or_none(rr_list)

    # Irrelevant: shift vs control, mean shift low/high, high-low contrast, toward-anchor rate
    shift_irr_low = []
    shift_irr_high = []
    toward_anchor_low = []
    toward_anchor_high = []
    for iid, conds in by_item_full.items():
        control_val = conds.get("control", {}).get("answer_int")
        if control_val is None:
            continue
        for cond, shift_list, toward_list in (
            ("irrelevant_low", shift_irr_low, toward_anchor_low),
            ("irrelevant_high", shift_irr_high, toward_anchor_high),
        ):
            r = conds.get(cond, {})
            stage2 = r.get("answer_int")
            stage1 = r.get("stage1_answer")
            if stage2 is not None:
                shift_list.append(stage2 - control_val)
            if stage2 is not None and stage1 is not None:
                # Toward anchor: stage2 moved toward stage1 relative to control
                if stage1 > control_val and stage2 > control_val:
                    toward_list.append(1.0)
                elif stage1 < control_val and stage2 < control_val:
                    toward_list.append(1.0)
                else:
                    toward_list.append(0.0)

    shift_low_mean = mean_or_none(shift_irr_low)
    shift_high_mean = mean_or_none(shift_irr_high)
    high_low_contrast = (
        (shift_high_mean - shift_low_mean) if (shift_high_mean is not None and shift_low_mean is not None) else None
    )
    toward_low = mean_or_none(toward_anchor_low)
    toward_high = mean_or_none(toward_anchor_high)
    toward_overall = mean_or_none(toward_anchor_low + toward_anchor_high)

    return {
        "n_items": len(by_item_full),
        "n_records": n_total,
        "parse_rate": round(parse_rate, 4),
        "mae_overall": round(mae_mean, 2) if mae_mean is not None else None,
        "accuracy_within_10": round(accuracy_10, 4) if accuracy_10 is not None else None,
        "mae_by_condition": {
            c: round(mean_or_none(mae_by_cond[c]), 2) if mae_by_cond[c] else None
            for c in HISTORY_V2_CONDITIONS
        },
        "acr_mean_plausible": round(acr_mean, 4) if acr_mean is not None else None,
        "rr_mean_plausible": round(rr_mean, 4) if rr_mean is not None else None,
        "shift_irrelevant_low": round(shift_low_mean, 2) if shift_low_mean is not None else None,
        "shift_irrelevant_high": round(shift_high_mean, 2) if shift_high_mean is not None else None,
        "shift_high_low_contrast": round(high_low_contrast, 2) if high_low_contrast is not None else None,
        "toward_anchor_rate_irrelevant_low": round(toward_low, 4) if toward_low is not None else None,
        "toward_anchor_rate_irrelevant_high": round(toward_high, 4) if toward_high is not None else None,
        "toward_anchor_rate_irrelevant": round(toward_overall, 4) if toward_overall is not None else None,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Run History v2 inference (two-stage for non-control) and evaluation"
    )
    p.add_argument("--promptviews", type=Path, required=True)
    p.add_argument("--itemspecs", type=Path, required=True)
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out_dir", type=Path, default=Path("results/history_v2_pilot"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--request_final_line", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--llm_fallback", action="store_true")
    p.add_argument("--fallback_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--fallback_device", type=str, default="auto")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset...")
    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items_history_v2(
        views, specs, args.max_items, args.seed
    )
    log.info(
        "Loaded %d items (%d prompts)",
        len(items),
        len(items) * len(HISTORY_V2_CONDITIONS),
    )

    if not items:
        log.error("No History v2 items found (need 5 conditions per item)")
        sys.exit(1)

    fallback = None
    if args.llm_fallback:
        log.info(
            "Loading LLM fallback extractor: %s on %s",
            args.fallback_model,
            args.fallback_device,
        )
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
        runner,
        items,
        results_path,
        max_tokens=args.max_tokens,
        fallback_extractor=fallback,
        request_final_line=getattr(args, "request_final_line", False),
    )

    metrics = compute_metrics(records)
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Summary -> %s", summary_path)

    print("\n--- History v2 evaluation summary ---")
    print(f"Model:              {args.model_id}")
    print(f"Items:              {metrics['n_items']}")
    print(f"Parse rate:         {metrics['parse_rate']:.2%}")
    print(f"MAE (overall):      {metrics['mae_overall']}")
    print(f"Accuracy (±10):     {metrics['accuracy_within_10']}")
    print(f"ACR (plausible):    {metrics['acr_mean_plausible']}")
    print(f"RR (plausible):     {metrics['rr_mean_plausible']}")
    print(f"Shift irr low:      {metrics['shift_irrelevant_low']}")
    print(f"Shift irr high:     {metrics['shift_irrelevant_high']}")
    print(f"Toward-anchor rate: {metrics['toward_anchor_rate_irrelevant']}")
    print()


if __name__ == "__main__":
    main()
