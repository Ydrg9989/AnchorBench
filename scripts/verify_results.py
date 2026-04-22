#!/usr/bin/env python3
"""Verify AnchorBench results for completeness and consistency.

Checks:
  1. Every model x suite has results.jsonl with expected record count
  2. Item ID prefixes match expected suite
  3. Parse rates are above threshold
  4. summary.json exists and has valid metrics
  5. No NaN values in critical metrics

Usage:
    PYTHONPATH=src python scripts/verify_results.py --results_dir results/full_benchmark
    PYTHONPATH=src python scripts/verify_results.py --results_dir results/api_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUITE_PREFIXES = {
    "external": "EXT-",
    "icl": "ICL-",
    "rag": "RAG-",
    "tool": "TOOL-",
    "history": "HIST-",
}

_DEFAULT_EXPECTED_RECORDS = 1800
MIN_PARSE_RATE = 0.50

CRITICAL_METRICS = [
    "parse_rate", "n_records", "n_items",
    "uai_irr", "uai_plaus", "tar_irr", "tar_plaus", "disc_delta",
]


def verify_suite(suite_dir: Path, suite: str, expected_prefix: str,
                 expected_records: int) -> list[str]:
    """Verify all model results for one suite. Returns list of issues."""
    issues = []
    if not suite_dir.is_dir():
        issues.append(f"  MISSING: {suite_dir} does not exist")
        return issues

    model_dirs = sorted(p for p in suite_dir.iterdir() if p.is_dir())
    if not model_dirs:
        issues.append(f"  EMPTY: no model directories in {suite_dir}")
        return issues

    for model_dir in model_dirs:
        model = model_dir.name
        results_file = model_dir / "results.jsonl"
        summary_file = model_dir / "summary.json"

        if not results_file.is_file():
            issues.append(f"  {suite}/{model}: results.jsonl MISSING")
            continue

        records = [json.loads(line) for line in results_file.open()]
        n = len(records)

        if n != expected_records:
            issues.append(f"  {suite}/{model}: {n} records (expected {expected_records})")

        bad_prefix = [r for r in records[:5] if not r.get("item_id", "").startswith(expected_prefix)]
        if bad_prefix:
            first_id = records[0].get("item_id", "???")
            issues.append(f"  {suite}/{model}: wrong prefix (first={first_id}, expected {expected_prefix}*)")

        parsed = sum(1 for r in records if r.get("parsed_ok"))
        rate = parsed / n if n > 0 else 0
        if rate < MIN_PARSE_RATE:
            issues.append(f"  {suite}/{model}: parse rate {rate:.1%} < {MIN_PARSE_RATE:.0%}")

        if not summary_file.is_file():
            issues.append(f"  {suite}/{model}: summary.json MISSING")
        else:
            summary = json.load(summary_file.open())
            for metric in CRITICAL_METRICS:
                val = summary.get(metric)
                if val is None:
                    issues.append(f"  {suite}/{model}: {metric} is None in summary.json")
                elif isinstance(val, float) and (val != val):  # NaN check
                    issues.append(f"  {suite}/{model}: {metric} is NaN in summary.json")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Verify AnchorBench results")
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--expected_records", type=int, default=_DEFAULT_EXPECTED_RECORDS)
    args = parser.parse_args()

    expected_records = args.expected_records

    results_dir = args.results_dir
    if not results_dir.is_dir():
        print(f"ERROR: {results_dir} does not exist")
        sys.exit(1)

    print(f"Verifying results in: {results_dir}")
    print(f"Expected records per suite: {expected_records}")
    print()

    all_issues = []
    for suite, prefix in SUITE_PREFIXES.items():
        suite_dir = results_dir / suite
        if not suite_dir.exists():
            continue
        print(f"Checking {suite}...")
        issues = verify_suite(suite_dir, suite, prefix, expected_records)
        if issues:
            for issue in issues:
                print(issue)
            all_issues.extend(issues)
        else:
            model_count = len([p for p in suite_dir.iterdir() if p.is_dir()])
            print(f"  OK ({model_count} models, all {expected_records} records)")

    print()
    if all_issues:
        print(f"FAILED: {len(all_issues)} issue(s) found")
        sys.exit(1)
    else:
        print("PASSED: all checks OK")


if __name__ == "__main__":
    main()
