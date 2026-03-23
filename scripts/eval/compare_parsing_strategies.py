#!/usr/bin/env python3
"""Compare parsing strategies on existing Llama model results.

Re-parses existing results.jsonl files with multiple extraction methods
to measure parse rate, agreement, and identify which strategy works best
for verbose chain-of-thought outputs.

Usage:
    PYTHONPATH=src python scripts/eval/compare_parsing_strategies.py \
        --results_dir results/smoke_test \
        --models meta-llama_Llama-3.1-8B-Instruct \
                 meta-llama_Llama-3.2-3B-Instruct \
                 meta-llama_Llama-3.2-1B-Instruct \
        --use_llm_extractor \
        --extractor_model meta-llama/Llama-3.2-3B-Instruct

    # Quick mode (no LLM extractor, regex strategies only):
    PYTHONPATH=src python scripts/eval/compare_parsing_strategies.py \
        --results_dir results/smoke_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench_eval.parsing import (
    STRATEGIES,
    LLMFallbackExtractor,
    apply_strategy,
    parse_answer_int,
    parse_final_answer,
    parse_last_number,
    parse_xml_answer,
)

log = logging.getLogger(__name__)

DETERMINISTIC_STRATEGIES = ("regex", "xml_tag", "last_number", "final_answer")


def find_llama_results(results_dir: Path, models: list[str] | None) -> list[dict]:
    """Discover all results.jsonl files for Llama models."""
    found = []
    for p in sorted(results_dir.rglob("results.jsonl")):
        rel = str(p.relative_to(results_dir))
        parts = rel.split("/")

        model_slug = None
        suite = None
        for part in parts:
            if "llama" in part.lower() or "Llama" in part:
                model_slug = part
            if part in ("external", "icl", "rag", "history", "tool"):
                suite = part

        if model_slug is None:
            continue
        if models and not any(m in model_slug for m in models):
            continue

        found.append({
            "path": p,
            "model": model_slug,
            "suite": suite or "unknown",
            "rel": rel,
        })
    return found


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def reparse_records(
    records: list[dict],
    strategies: tuple[str, ...],
    fallback: LLMFallbackExtractor | None = None,
) -> list[dict]:
    """Re-parse each record with all strategies, return enriched records."""
    enriched = []
    for rec in records:
        raw = rec.get("raw_text", "")
        prompt = rec.get("prompt_text", "")
        if not prompt:
            for field in ("prompt_text", "prompt"):
                if rec.get(field):
                    prompt = rec[field]
                    break

        row = {
            "item_id": rec.get("item_id", ""),
            "suite": rec.get("suite", ""),
            "condition": rec.get("condition", ""),
            "domain": rec.get("domain", ""),
            "original_answer": rec.get("answer_int"),
            "original_strategy": rec.get("parse_strategy", ""),
            "original_parsed_ok": rec.get("parsed_ok", False),
            "raw_text_len": len(raw),
        }

        for strat in strategies:
            if strat == "llm_fallback" and fallback is None:
                row[f"{strat}_answer"] = None
                row[f"{strat}_ok"] = False
                continue
            answer, ok = apply_strategy(raw, prompt, strat, fallback)
            row[f"{strat}_answer"] = answer
            row[f"{strat}_ok"] = ok

        enriched.append(row)
    return enriched


def compute_strategy_stats(
    enriched: list[dict],
    strategies: tuple[str, ...],
) -> dict:
    """Compute per-strategy parse rate and agreement metrics."""
    n = len(enriched)
    if n == 0:
        return {}

    stats = {}
    for strat in strategies:
        ok_count = sum(1 for r in enriched if r.get(f"{strat}_ok", False))
        answers = [r.get(f"{strat}_answer") for r in enriched if r.get(f"{strat}_ok")]

        stats[strat] = {
            "parse_rate": ok_count / n if n else 0,
            "parsed": ok_count,
            "failed": n - ok_count,
            "total": n,
        }

    # Agreement matrix between strategies
    agreement = {}
    for s1 in strategies:
        for s2 in strategies:
            if s1 >= s2:
                continue
            both_ok = 0
            agree = 0
            disagree_pairs = []
            for r in enriched:
                a1, ok1 = r.get(f"{s1}_answer"), r.get(f"{s1}_ok", False)
                a2, ok2 = r.get(f"{s2}_answer"), r.get(f"{s2}_ok", False)
                if ok1 and ok2:
                    both_ok += 1
                    if a1 == a2:
                        agree += 1
                    else:
                        disagree_pairs.append({
                            "item_id": r["item_id"],
                            "condition": r["condition"],
                            f"{s1}": a1,
                            f"{s2}": a2,
                        })
            agreement[f"{s1}_vs_{s2}"] = {
                "both_parsed": both_ok,
                "agree": agree,
                "disagree": both_ok - agree,
                "agreement_rate": agree / both_ok if both_ok else 0,
                "disagreements": disagree_pairs[:5],
            }

    # "Rescue" rate: how many items each strategy recovers that regex failed
    regex_fails = [r for r in enriched if not r.get("regex_ok", False)]
    rescue = {}
    for strat in strategies:
        if strat == "regex":
            continue
        rescued = sum(1 for r in regex_fails if r.get(f"{strat}_ok", False))
        rescue[strat] = {
            "regex_failures": len(regex_fails),
            "rescued": rescued,
            "rescue_rate": rescued / len(regex_fails) if regex_fails else 0,
        }

    return {"per_strategy": stats, "agreement": agreement, "rescue": rescue}


def print_report(
    all_stats: dict[str, dict],
    strategies: tuple[str, ...],
) -> None:
    """Print formatted comparison report."""
    print("\n" + "=" * 80)
    print("PARSING STRATEGY COMPARISON REPORT")
    print("=" * 80)

    for key, stats_data in all_stats.items():
        model, suite = key.rsplit("/", 1)
        stats = stats_data["stats"]
        n = stats_data["n_records"]

        print(f"\n{'─' * 70}")
        print(f"  Model: {model}  |  Suite: {suite}  |  N={n}")
        print(f"{'─' * 70}")

        per_strat = stats.get("per_strategy", {})
        print(f"\n  {'Strategy':<20s} {'Parsed':>8s} {'Failed':>8s} {'Rate':>8s}")
        print(f"  {'─' * 48}")
        for strat in strategies:
            s = per_strat.get(strat, {})
            if s:
                print(
                    f"  {strat:<20s} {s['parsed']:>8d} {s['failed']:>8d} "
                    f"{s['parse_rate']:>7.1%}"
                )

        rescue = stats.get("rescue", {})
        if rescue:
            print(f"\n  Rescue from regex failures:")
            for strat, r in rescue.items():
                print(
                    f"    {strat:<18s}: {r['rescued']:>4d} / {r['regex_failures']} "
                    f"rescued ({r['rescue_rate']:.1%})"
                )

        agreement = stats.get("agreement", {})
        if agreement:
            disagreements_found = False
            for pair_key, ag in agreement.items():
                if ag["disagree"] > 0:
                    if not disagreements_found:
                        print(f"\n  Disagreements between strategies:")
                        disagreements_found = True
                    print(
                        f"    {pair_key}: {ag['disagree']} disagreements "
                        f"(agreement: {ag['agreement_rate']:.1%})"
                    )
                    for d in ag.get("disagreements", [])[:3]:
                        vals = {k: v for k, v in d.items()
                                if k not in ("item_id", "condition")}
                        print(f"      {d['item_id']}/{d['condition']}: {vals}")

    # Summary table across all
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: Parse Rates Across All Models/Suites")
    print(f"{'=' * 80}")
    print(f"\n  {'Model/Suite':<50s}", end="")
    for strat in strategies:
        print(f" {strat:>12s}", end="")
    print()
    print(f"  {'─' * (50 + 13 * len(strategies))}")
    for key, stats_data in all_stats.items():
        per_strat = stats_data["stats"].get("per_strategy", {})
        print(f"  {key:<50s}", end="")
        for strat in strategies:
            s = per_strat.get(strat, {})
            rate = s.get("parse_rate", 0)
            print(f" {rate:>11.1%}", end="")
        print()
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Compare parsing strategies on Llama results")
    p.add_argument("--results_dir", type=Path, default=Path("results/smoke_test"))
    p.add_argument("--models", nargs="*", default=None,
                   help="Filter to these model slugs (substring match). "
                        "Default: all Llama models found.")
    p.add_argument("--suites", nargs="*", default=None,
                   help="Filter to these suites. Default: all.")
    p.add_argument("--use_llm_extractor", action="store_true",
                   help="Include the 2-stage LLM extractor (requires GPU)")
    p.add_argument("--extractor_model", type=str,
                   default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--extractor_device", type=str, default="auto")
    p.add_argument("--out_json", type=Path, default=None,
                   help="Path to write detailed JSON results")
    args = p.parse_args()

    result_files = find_llama_results(args.results_dir, args.models)
    if args.suites:
        result_files = [r for r in result_files if r["suite"] in args.suites]

    if not result_files:
        log.error("No Llama results found in %s", args.results_dir)
        sys.exit(1)

    log.info("Found %d result files:", len(result_files))
    for rf in result_files:
        log.info("  %s (%s / %s)", rf["rel"], rf["model"], rf["suite"])

    strategies = DETERMINISTIC_STRATEGIES
    fallback = None
    if args.use_llm_extractor:
        strategies = STRATEGIES
        log.info("Loading LLM extractor: %s", args.extractor_model)
        fallback = LLMFallbackExtractor(
            model_id=args.extractor_model,
            device=args.extractor_device,
        )

    all_stats: dict[str, dict] = {}
    all_enriched: dict[str, list[dict]] = {}

    for rf in result_files:
        key = f"{rf['model']}/{rf['suite']}"
        log.info("Processing %s ...", key)

        records = load_records(rf["path"])
        if not records:
            log.warning("No records in %s", rf["path"])
            continue

        enriched = reparse_records(records, strategies, fallback)
        stats = compute_strategy_stats(enriched, strategies)

        all_stats[key] = {"stats": stats, "n_records": len(records)}
        all_enriched[key] = enriched

    print_report(all_stats, strategies)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "strategies": list(strategies),
            "stats": all_stats,
            "enriched_samples": {
                k: v[:10] for k, v in all_enriched.items()
            },
        }
        with open(args.out_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        log.info("Detailed results -> %s", args.out_json)


if __name__ == "__main__":
    main()
