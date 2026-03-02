#!/usr/bin/env python3
"""SynAnchors inference + evaluation: numeric priming anchoring experiment.

For each item, generates model outputs under two conditions:
  (1) WITH-ANCHOR:    anchor_text + "\n\n" + question
  (2) WITHOUT-ANCHOR: question alone

Then evaluates anchoring with:
  - Paired t-test (with-anchor vs without-anchor predictions)
  - R-Error = mean(|v_anchor - v_orig| / |v_orig|)
  - Occurrence criterion: p < 0.05 AND R-Error > 0.2

Usage:
  python scripts/eval/run_syn_anchors.py \
    --dataset data/processed/syn_anchors_v0/dataset.jsonl \
    --models meta-llama/Llama-3.2-1B-Instruct \
             meta-llama/Llama-3.2-3B-Instruct \
             meta-llama/Llama-3.1-8B-Instruct \
    --out_dir results/syn_anchors \
    --n_samples 5 --seed 0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from llm_anchoring.eval.inference import (
    build_gen_kwargs,
    generate_batch,
    load_model,
    prepare_prompt,
)
from llm_anchoring.eval.parsers import parse_hierarchical

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SynAnchors inference + eval")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, default=Path("results/syn_anchors"))
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--skip_inference", action="store_true",
                   help="Skip inference, only re-run evaluation on existing generations")
    return p.parse_args()


# ── Prompt construction ──────────────────────────────────────────────

INSTRUCTION_SUFFIX = (
    "\n\nProvide your best numerical estimate as a single number. "
    "Output only the number, nothing else."
)


def build_prompts(item: dict) -> dict[str, str]:
    """Build WITH-ANCHOR and WITHOUT-ANCHOR prompts for one item."""
    q = item["question"] + INSTRUCTION_SUFFIX
    return {
        "without_anchor": q,
        "with_anchor": item["anchor_text"] + "\n\n" + q,
    }


# ── Inference ────────────────────────────────────────────────────────

def run_inference(
    model_id: str, items: list[dict], args: argparse.Namespace
) -> list[dict]:
    """Generate outputs for all items under both conditions."""
    model, tokenizer = load_model(model_id, dtype="auto")
    gen_kwargs = build_gen_kwargs(
        {"temperature": args.temperature, "top_p": 0.9},
        args.max_new_tokens,
    )
    gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    all_prompts = []
    prompt_meta = []
    for item in items:
        prompts = build_prompts(item)
        for condition, raw_prompt in prompts.items():
            chat_prompt = prepare_prompt(raw_prompt, tokenizer, use_chat_template=True)
            all_prompts.append(chat_prompt)
            prompt_meta.append({"item": item, "condition": condition})

    records = []
    run_id = str(uuid.uuid4())[:8]

    for si in range(args.n_samples):
        torch.manual_seed(args.seed + si)
        logger.info("[%s] sample %d/%d (%d prompts)",
                     model_id.split("/")[-1], si + 1, args.n_samples, len(all_prompts))

        for bs in range(0, len(all_prompts), args.batch_size):
            be = min(bs + args.batch_size, len(all_prompts))
            outputs = generate_batch(
                all_prompts[bs:be], model, tokenizer, gen_kwargs
            )
            for i, raw_text in enumerate(outputs):
                meta = prompt_meta[bs + i]
                item = meta["item"]
                trace = parse_hierarchical(raw_text, prefer_last=True)
                pv = trace["parsed_value"]
                records.append({
                    "run_id": run_id,
                    "model_id": model_id,
                    "seed": args.seed,
                    "sample_idx": si,
                    "item_id": item["id"],
                    "topic": item["topic"],
                    "condition": meta["condition"],
                    "true_value": item["true_value"],
                    "anchor_value": item["anchor_value"],
                    "unit": item["unit"],
                    "difficulty": item["difficulty"],
                    "raw_output_text": raw_text,
                    "parsed_value": pv,
                    "parse_ok": trace["parse_ok"],
                    "parse_method": trace["strategy_used"],
                })

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return records


# ── Evaluation metrics ───────────────────────────────────────────────

def evaluate(records: list[dict]) -> dict:
    """Compute R-Error, paired t-test, occurrence criterion, per-item stats."""

    # Group by item_id: collect parsed values per condition
    by_item: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["parse_ok"] and r["parsed_value"] is not None:
            by_item[r["item_id"]][r["condition"]].append(r["parsed_value"])

    item_meta = {}
    for r in records:
        if r["item_id"] not in item_meta:
            item_meta[r["item_id"]] = {
                "topic": r["topic"], "true_value": r["true_value"],
                "anchor_value": r["anchor_value"], "unit": r["unit"],
                "difficulty": r["difficulty"],
            }

    per_item = []
    v_anchor_list = []
    v_orig_list = []
    r_errors = []

    for item_id in sorted(by_item):
        wa = by_item[item_id].get("with_anchor", [])
        wo = by_item[item_id].get("without_anchor", [])
        meta = item_meta[item_id]

        median_wa = float(np.median(wa)) if wa else None
        median_wo = float(np.median(wo)) if wo else None
        mean_wa = float(np.mean(wa)) if wa else None
        mean_wo = float(np.mean(wo)) if wo else None

        row = {
            "item_id": item_id,
            **meta,
            "n_with_anchor": len(wa),
            "n_without_anchor": len(wo),
            "median_with_anchor": median_wa,
            "median_without_anchor": median_wo,
            "mean_with_anchor": mean_wa,
            "mean_without_anchor": mean_wo,
        }

        if mean_wa is not None and mean_wo is not None:
            v_anchor_list.append(mean_wa)
            v_orig_list.append(mean_wo)

            denom = max(abs(mean_wo), 1e-9)
            r_err = abs(mean_wa - mean_wo) / denom
            r_errors.append(r_err)
            row["r_error_item"] = r_err

            shift = mean_wa - mean_wo
            row["shift"] = shift
            toward_anchor = (meta["anchor_value"] - mean_wo)
            row["shift_toward_anchor"] = (
                shift * toward_anchor > 0 if toward_anchor != 0 else False
            )

        per_item.append(row)

    r_error_mean = float(np.mean(r_errors)) if r_errors else None
    r_error_median = float(np.median(r_errors)) if r_errors else None

    # Trimmed R-Error (drop top 5% of outliers for robust estimate)
    r_error_trimmed = None
    if r_errors:
        sorted_re = sorted(r_errors)
        trim_n = max(1, int(len(sorted_re) * 0.95))
        r_error_trimmed = float(np.mean(sorted_re[:trim_n]))

    # Paired t-test on per-item medians (robust to within-item outliers)
    med_anchor = [row["median_with_anchor"] for row in per_item
                  if row.get("median_with_anchor") is not None
                  and row.get("median_without_anchor") is not None]
    med_orig = [row["median_without_anchor"] for row in per_item
                if row.get("median_with_anchor") is not None
                and row.get("median_without_anchor") is not None]
    v_a = np.array(med_anchor)
    v_o = np.array(med_orig)
    if len(v_a) >= 2:
        t_stat, p_value = stats.ttest_rel(v_a, v_o)
        t_stat = float(t_stat)
        p_value = float(p_value)
    else:
        t_stat, p_value = None, None

    # Wilcoxon signed-rank test (non-parametric, robust to outliers)
    if len(v_a) >= 2:
        diffs = v_a - v_o
        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 10:
            w_stat, w_pvalue = stats.wilcoxon(nonzero)
            w_stat, w_pvalue = float(w_stat), float(w_pvalue)
        else:
            w_stat, w_pvalue = None, None
    else:
        w_stat, w_pvalue = None, None

    # Occurrence criterion (use trimmed R-Error for robustness)
    occurrence = (
        p_value is not None
        and p_value < 0.05
        and r_error_trimmed is not None
        and r_error_trimmed > 0.2
    )

    # Shift-toward-anchor rate
    n_toward = sum(1 for row in per_item if row.get("shift_toward_anchor"))
    n_with_shift = sum(1 for row in per_item if "shift" in row)

    # Parse rate
    total = len(records)
    parsed = sum(1 for r in records if r["parse_ok"])

    # Per-topic breakdown
    topic_groups: dict[str, list[float]] = defaultdict(list)
    for row in per_item:
        if "r_error_item" in row:
            topic_groups[row["topic"]].append(row["r_error_item"])
    topic_breakdown = {}
    for topic in sorted(topic_groups):
        vals = topic_groups[topic]
        topic_breakdown[topic] = {
            "n_items": len(vals),
            "r_error_mean": float(np.mean(vals)),
            "r_error_median": float(np.median(vals)),
        }

    # Per-difficulty breakdown
    diff_groups: dict[str, list[float]] = defaultdict(list)
    for row in per_item:
        if "r_error_item" in row:
            diff_groups[row["difficulty"]].append(row["r_error_item"])
    difficulty_breakdown = {}
    for diff in ["easy", "medium", "hard"]:
        vals = diff_groups.get(diff, [])
        if vals:
            difficulty_breakdown[diff] = {
                "n_items": len(vals),
                "r_error_mean": float(np.mean(vals)),
            }

    summary = {
        "n_items": len(per_item),
        "n_records": total,
        "parse_rate": parsed / total if total else 0,
        "r_error_mean": r_error_mean,
        "r_error_median": r_error_median,
        "r_error_trimmed_95": r_error_trimmed,
        "paired_t_stat": t_stat,
        "paired_p_value": p_value,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_pvalue,
        "anchoring_present_ttest": p_value < 0.05 if p_value is not None else None,
        "anchoring_present_wilcoxon": w_pvalue < 0.05 if w_pvalue is not None else None,
        "occurrence": occurrence,
        "shift_toward_anchor_rate": n_toward / n_with_shift if n_with_shift else None,
        "topic_breakdown": topic_breakdown,
        "difficulty_breakdown": difficulty_breakdown,
    }
    return {"summary": summary, "per_item": per_item}


# ── Report formatting ────────────────────────────────────────────────

def format_report(model_id: str, result: dict) -> str:
    s = result["summary"]
    lines = [
        f"{'=' * 70}",
        f"Model: {model_id.split('/')[-1]}",
        f"{'=' * 70}",
        f"Items: {s['n_items']}   Records: {s['n_records']}   Parse rate: {s['parse_rate']:.1%}",
        "",
        f"  R-Error (mean):         {s['r_error_mean']:.4f}" if s['r_error_mean'] is not None else "  R-Error: N/A",
        f"  R-Error (median):       {s['r_error_median']:.4f}" if s['r_error_median'] is not None else "",
        f"  R-Error (trimmed 95%):  {s['r_error_trimmed_95']:.4f}" if s.get('r_error_trimmed_95') is not None else "",
        "",
        f"  Paired t-test:          t={s['paired_t_stat']:.3f}   p={s['paired_p_value']:.4f}" if s['paired_t_stat'] is not None else "  Paired t-test: N/A",
        f"  Wilcoxon signed-rank:   W={s['wilcoxon_stat']:.1f}   p={s['wilcoxon_p_value']:.4f}" if s.get('wilcoxon_stat') is not None else "  Wilcoxon: N/A (too few non-zero diffs)",
        "",
        f"  Anchoring (t-test):     {'YES' if s.get('anchoring_present_ttest') else 'NO'} (p < 0.05)",
        f"  Anchoring (Wilcoxon):   {'YES' if s.get('anchoring_present_wilcoxon') else 'NO'} (p < 0.05)",
        f"  Occurrence criterion:   {'YES' if s['occurrence'] else 'NO'} (p < 0.05 AND R-Error_trim > 0.2)",
        f"  Shift toward anchor:    {s['shift_toward_anchor_rate']:.1%}" if s['shift_toward_anchor_rate'] is not None else "",
        "",
        "  Per-topic R-Error (median):",
    ]
    for topic, vals in sorted(s["topic_breakdown"].items()):
        lines.append(f"    {topic:45s}  mean={vals['r_error_mean']:.4f}  med={vals['r_error_median']:.4f}  (n={vals['n_items']})")
    lines.append("")
    lines.append("  Per-difficulty R-Error:")
    for diff in ["easy", "medium", "hard"]:
        if diff in s["difficulty_breakdown"]:
            d = s["difficulty_breakdown"][diff]
            lines.append(f"    {diff:12s}  {d['r_error_mean']:.4f}  (n={d['n_items']})")
    lines.append("")
    return "\n".join(lines)


def format_comparison(all_results: dict[str, dict]) -> str:
    lines = [
        "",
        "=" * 80,
        "CROSS-MODEL COMPARISON",
        "=" * 80,
        "",
        f"  {'Model':28s} {'R-Err(med)':>10s} {'R-Err(trim)':>11s} {'t-test p':>10s} {'Wilcox p':>10s} {'Occur?':>7s} {'Toward%':>8s}",
        f"  {'-'*28} {'-'*10} {'-'*11} {'-'*10} {'-'*10} {'-'*7} {'-'*8}",
    ]
    for model_id, result in all_results.items():
        s = result["summary"]
        name = model_id.split("/")[-1]
        med_str = f"{s['r_error_median']:.4f}" if s['r_error_median'] is not None else "N/A"
        trim_str = f"{s['r_error_trimmed_95']:.4f}" if s.get('r_error_trimmed_95') is not None else "N/A"
        pt_str = f"{s['paired_p_value']:.4f}" if s['paired_p_value'] is not None else "N/A"
        pw_str = f"{s['wilcoxon_p_value']:.4f}" if s.get('wilcoxon_p_value') is not None else "N/A"
        oc_str = "YES" if s["occurrence"] else "NO"
        tw_str = f"{s['shift_toward_anchor_rate']:.1%}" if s['shift_toward_anchor_rate'] is not None else "N/A"
        lines.append(f"  {name:28s} {med_str:>10s} {trim_str:>11s} {pt_str:>10s} {pw_str:>10s} {oc_str:>7s} {tw_str:>8s}")
    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in Path(args.dataset).read_text().strip().split("\n")]
    logger.info("Loaded %d items from %s", len(items), args.dataset)

    all_results: dict[str, dict] = {}

    for model_id in args.models:
        model_short = model_id.split("/")[-1]
        gen_path = args.out_dir / f"{model_short}_generations.jsonl"

        if args.skip_inference and gen_path.exists():
            logger.info("Loading existing generations for %s", model_short)
            records = [json.loads(l) for l in gen_path.read_text().strip().split("\n")]
        else:
            logger.info("Running inference for %s", model_short)
            records = run_inference(model_id, items, args)
            with gen_path.open("w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            logger.info("Saved %d records to %s", len(records), gen_path)

        result = evaluate(records)
        all_results[model_id] = result

        summary_path = args.out_dir / f"{model_short}_summary.json"
        summary_path.write_text(json.dumps(result["summary"], indent=2))

        per_item_path = args.out_dir / f"{model_short}_per_item.jsonl"
        with per_item_path.open("w") as f:
            for row in result["per_item"]:
                f.write(json.dumps(row) + "\n")

        report = format_report(model_id, result)
        print(report)

    if len(all_results) > 1:
        comparison = format_comparison(all_results)
        print(comparison)

        comp_summary = {}
        for model_id, result in all_results.items():
            name = model_id.split("/")[-1]
            s = result["summary"]
            comp_summary[name] = {
                "r_error_mean": s["r_error_mean"],
                "r_error_median": s["r_error_median"],
                "r_error_trimmed_95": s.get("r_error_trimmed_95"),
                "paired_t_stat": s["paired_t_stat"],
                "paired_p_value": s["paired_p_value"],
                "wilcoxon_stat": s.get("wilcoxon_stat"),
                "wilcoxon_p_value": s.get("wilcoxon_p_value"),
                "anchoring_present_ttest": s.get("anchoring_present_ttest"),
                "anchoring_present_wilcoxon": s.get("anchoring_present_wilcoxon"),
                "occurrence": s["occurrence"],
                "shift_toward_anchor_rate": s["shift_toward_anchor_rate"],
                "parse_rate": s["parse_rate"],
            }
        comp_path = args.out_dir / "comparison_summary.json"
        comp_path.write_text(json.dumps(comp_summary, indent=2))

    print(f"Outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
