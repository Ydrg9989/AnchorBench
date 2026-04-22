#!/usr/bin/env python3
"""Re-parse failed records using the LLM fallback extractor.

Reads a results.jsonl file, identifies records where parsed_ok is False,
runs LLMFallbackExtractor on them, writes updated results alongside the
original, and prints a comparison of metrics before/after.

Usage:
    PYTHONPATH=src python scripts/eval/reparse_with_fallback.py \
        results/full_benchmark/tool/meta-llama_Llama-3.1-8B-Instruct/results.jsonl

    # Custom extractor model:
    PYTHONPATH=src python scripts/eval/reparse_with_fallback.py \
        results/full_benchmark/tool/meta-llama_Llama-3.1-8B-Instruct/results.jsonl \
        --extractor_model meta-llama/Llama-3.2-3B-Instruct

    # Batch mode: scan all results under a directory
    PYTHONPATH=src python scripts/eval/reparse_with_fallback.py \
        --results_dir results/full_benchmark --only_failures
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench_eval.metrics import compute_unified_metrics
from anchorbench_eval.parsing import LLMFallbackExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

from anchorbench_eval.runner_utils import discover_results as _discover_all, fmt, SUITES


def _discover_results_local(results_dir: Path) -> list[tuple[str, str, Path]]:
    return [(s, slug, p) for s, slug, _, p in _discover_all(results_dir)]


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt_pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "---"


def reparse_failures(
    records: list[dict],
    extractor: LLMFallbackExtractor,
) -> tuple[list[dict], int, int]:
    """Re-parse failed records using the LLM fallback extractor.

    Returns (updated_records, n_attempted, n_recovered).
    """
    updated = []
    n_attempted = 0
    n_recovered = 0
    for r in records:
        if r.get("parsed_ok"):
            updated.append(r)
            continue

        r2 = copy.copy(r)
        raw = r.get("raw_text", "") or ""
        n_attempted += 1

        answer, ok = extractor.try_extract(raw)
        if ok:
            r2["answer_int"] = answer
            r2["parsed_ok"] = True
            r2["parse_strategy"] = "llm_fallback"
            n_recovered += 1
            log.info(
                "Recovered %s/%s: %d (raw: %.60s...)",
                r.get("item_id", "?"), r.get("condition", "?"),
                answer, raw,
            )
        updated.append(r2)
    return updated, n_attempted, n_recovered


def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-parse failed records using LLM fallback extractor",
    )
    p.add_argument("results", type=Path, nargs="*", help="Path(s) to results.jsonl")
    p.add_argument(
        "--results_dir", type=Path, default=None,
        help="Scan all results.jsonl under this directory (batch mode)",
    )
    p.add_argument(
        "--extractor_model", type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model for fallback extraction",
    )
    p.add_argument("--epsilon", type=float, default=3.0)
    p.add_argument(
        "--only_failures", action="store_true",
        help="In batch mode, only process files with parse failures",
    )
    p.add_argument(
        "--write", action="store_true",
        help="Write updated results to results_with_fallback.jsonl",
    )
    args = p.parse_args()

    paths: list[tuple[str, Path]] = []
    if args.results_dir:
        for suite, model_slug, rpath in _discover_results_local(args.results_dir):
            paths.append((f"{suite}/{model_slug}", rpath))
    for rpath in (args.results or []):
        paths.append((str(rpath.name), rpath))

    if not paths:
        print("No results files found.")
        sys.exit(1)

    if args.only_failures:
        filtered = []
        for label, rpath in paths:
            records = load_records(rpath)
            n_fail = sum(1 for r in records if not r.get("parsed_ok"))
            if n_fail > 0:
                filtered.append((label, rpath, n_fail))
        if not filtered:
            print("All results have 100% parse rate. Nothing to do.")
            sys.exit(0)
        print(f"Found {len(filtered)} files with parse failures:")
        for label, _, n in filtered:
            print(f"  {label}: {n} failures")
        print()
        paths = [(label, rpath) for label, rpath, _ in filtered]

    log.info("Loading LLM fallback extractor: %s", args.extractor_model)
    extractor = LLMFallbackExtractor(model_id=args.extractor_model)

    summary_rows = []
    for label, results_path in paths:
        records = load_records(results_path)
        if not records:
            continue

        n_fail = sum(1 for r in records if not r.get("parsed_ok"))
        if n_fail == 0:
            continue

        print(f"\n{'=' * 90}")
        print(f"  Re-parsing failures: {label}")
        print(f"  ({len(records)} records, {n_fail} failures)")
        print(f"{'=' * 90}")

        original_m = compute_unified_metrics(records, epsilon=args.epsilon)
        updated, n_attempted, n_recovered = reparse_failures(records, extractor)
        updated_m = compute_unified_metrics(updated, epsilon=args.epsilon)

        print(f"\n  Attempted: {n_attempted}, Recovered: {n_recovered}")
        header = (
            f"  {'Variant':<16} {'Parse%':>7} {'MAE_c':>6} {'UAI_irr':>8} "
            f"{'UAI_pls':>8} {'TAR_irr':>8} {'TAR_pls':>8} {'Disc_Δ':>8}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for tag, m in [("original", original_m), ("with_fallback", updated_m)]:
            print(
                f"  {tag:<16} "
                f"{fmt_pct(m['parse_rate']):>7} "
                f"{fmt(m.get('mae_control'), 2):>6} "
                f"{fmt(m['uai_irr']):>8} "
                f"{fmt(m['uai_plaus']):>8} "
                f"{fmt(m['tar_irr']):>8} "
                f"{fmt(m['tar_plaus']):>8} "
                f"{fmt(m['disc_delta']):>8}"
            )

        summary_rows.append({
            "label": label,
            "n_records": len(records),
            "n_failures": n_fail,
            "n_recovered": n_recovered,
            "parse_rate_before": original_m.get("parse_rate"),
            "parse_rate_after": updated_m.get("parse_rate"),
        })

        if args.write:
            out_path = results_path.parent / "results_with_fallback.jsonl"
            with open(out_path, "w") as f:
                for r in updated:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Written to {out_path}")

    if summary_rows:
        print(f"\n{'=' * 90}")
        print("  LLM Fallback Recovery Summary")
        print(f"{'=' * 90}")
        print(
            f"  {'Label':<35} {'Fail':>5} {'Recov':>5} "
            f"{'Rate_before':>12} {'Rate_after':>12}"
        )
        print("  " + "-" * 78)
        for row in summary_rows:
            print(
                f"  {row['label']:<35} "
                f"{row['n_failures']:>5} "
                f"{row['n_recovered']:>5} "
                f"{fmt_pct(row['parse_rate_before']):>12} "
                f"{fmt_pct(row['parse_rate_after']):>12}"
            )


if __name__ == "__main__":
    main()
