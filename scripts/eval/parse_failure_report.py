#!/usr/bin/env python3
"""Generate a comprehensive parse failure diagnostic report.

Scans all results.jsonl under a results directory, categorizes parse
failures, and produces a markdown report.

Failure categories:
  out_of_range   — extractable integer outside [0, 100]
  tool_call_json — model emits tool-call JSON instead of an answer
  prose_no_int   — prose/text with no parseable integer
  error_prefix   — raw_text starts with "ERROR:"
  empty          — empty or very short raw_text
  other          — uncategorized

Usage:
    PYTHONPATH=src python scripts/eval/parse_failure_report.py \
        --results_dir results/full_benchmark
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SUITES = ("external", "icl", "rag", "tool", "history")

_INT_PAT = re.compile(r"(?<![\d.])-?\d+(?!\d)(?!\.\d)")

MODEL_SHORT = {
    "meta-llama_Llama-3.2-1B-Instruct": "Llama-1B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-8B",
    "Qwen_Qwen2.5-1.5B-Instruct": "Qwen-1.5B",
    "Qwen_Qwen2.5-3B-Instruct": "Qwen-3B",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen-7B",
    "google_gemma-3-1b-it": "Gemma-1B",
    "google_gemma-3-4b-it": "Gemma-4B",
    "allenai_OLMo-2-1124-13B-Instruct": "OLMo-13B",
    "allenai_OLMo-2-0325-32B-Instruct": "OLMo-32B",
}


def categorize_failure(raw_text: str) -> str:
    """Categorize a parse failure based on raw_text content."""
    if not raw_text or len(raw_text.strip()) < 2:
        return "empty"
    text = raw_text.strip()
    if text.startswith("ERROR:"):
        return "error_prefix"

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and ("name" in obj or "function" in obj or "tool_calls" in obj):
            return "tool_call_json"
    except (json.JSONDecodeError, ValueError):
        pass

    all_ints = [int(m.group()) for m in _INT_PAT.finditer(text)]
    oor = [v for v in all_ints if v < 0 or v > 100]
    in_range = [v for v in all_ints if 0 <= v <= 100]
    if oor and not in_range:
        return "out_of_range"

    if not all_ints:
        return "prose_no_int"

    return "other"


def discover_results(results_dir: Path) -> list[tuple[str, str, Path]]:
    found = []
    for suite in SUITES:
        suite_dir = results_dir / suite
        if not suite_dir.is_dir():
            continue
        for p in sorted(suite_dir.rglob("results.jsonl")):
            rel = p.relative_to(suite_dir)
            parts = list(rel.parts)
            model_slug = parts[0] if len(parts) >= 2 else suite_dir.name
            if model_slug in SUITES:
                continue
            found.append((suite, model_slug, p))
    return found


def truncate(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def generate_report(results_dir: Path) -> str:
    """Generate the markdown report and return it as a string."""
    runs = discover_results(results_dir)

    lines = [
        "# Parse Failure Diagnostic Report",
        "",
        f"Results directory: `{results_dir}`",
        "",
    ]

    # Global stats
    total_records = 0
    total_failures = 0
    global_cats: dict[str, int] = defaultdict(int)

    suite_model_data: list[dict] = []

    for suite, model_slug, rpath in runs:
        with open(rpath) as f:
            records = [json.loads(line) for line in f if line.strip()]

        n = len(records)
        failures = [r for r in records if not r.get("parsed_ok")]
        n_fail = len(failures)
        total_records += n
        total_failures += n_fail

        cats: dict[str, list[dict]] = defaultdict(list)
        cond_counts: dict[str, int] = defaultdict(int)

        for r in failures:
            raw = r.get("raw_text", "") or ""
            cat = categorize_failure(raw)
            cats[cat].append(r)
            global_cats[cat] += 1
            cond_counts[r.get("condition", "?")] += 1

        short = MODEL_SHORT.get(model_slug, model_slug)
        suite_model_data.append({
            "suite": suite,
            "model_slug": model_slug,
            "short": short,
            "n_records": n,
            "n_failures": n_fail,
            "parse_rate": (n - n_fail) / n if n else 1.0,
            "categories": {k: len(v) for k, v in cats.items()},
            "condition_counts": dict(cond_counts),
            "failure_records": dict(cats),
        })

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total records:** {total_records:,}")
    lines.append(f"- **Total failures:** {total_failures:,}")
    lines.append(
        f"- **Overall parse rate:** "
        f"{(total_records - total_failures) / total_records * 100:.2f}%"
    )
    lines.append("")

    if global_cats:
        lines.append("### Failure Categories (global)")
        lines.append("")
        lines.append("| Category | Count | % of failures |")
        lines.append("|----------|------:|:-------------:|")
        for cat in sorted(global_cats, key=global_cats.get, reverse=True):
            cnt = global_cats[cat]
            pct = cnt / total_failures * 100 if total_failures else 0
            lines.append(f"| {cat} | {cnt} | {pct:.1f}% |")
        lines.append("")

    # Per suite/model table
    imperfect = [d for d in suite_model_data if d["n_failures"] > 0]
    if imperfect:
        lines.append("## Models with Parse Failures")
        lines.append("")
        lines.append(
            "| Suite | Model | Records | Failures | Parse Rate | Categories |"
        )
        lines.append("|-------|-------|--------:|---------:|:----------:|:-----------|")
        for d in sorted(imperfect, key=lambda x: x["n_failures"], reverse=True):
            cat_str = ", ".join(
                f"{k}={v}" for k, v in
                sorted(d["categories"].items(), key=lambda x: x[1], reverse=True)
            )
            lines.append(
                f"| {d['suite']} | {d['short']} | {d['n_records']} | "
                f"{d['n_failures']} | {d['parse_rate'] * 100:.1f}% | {cat_str} |"
            )
        lines.append("")

    # Detailed failure examples
    lines.append("## Detailed Failure Examples")
    lines.append("")
    for d in sorted(imperfect, key=lambda x: (-x["n_failures"], x["suite"])):
        lines.append(f"### {d['suite'].capitalize()} / {d['short']}")
        lines.append(f"  {d['n_failures']} failures out of {d['n_records']} records")
        lines.append("")

        if d["condition_counts"]:
            lines.append("**By condition:**")
            for cond, cnt in sorted(
                d["condition_counts"].items(),
                key=lambda x: x[1], reverse=True,
            ):
                lines.append(f"- {cond}: {cnt}")
            lines.append("")

        for cat, recs in sorted(
            d["failure_records"].items(),
            key=lambda x: len(x[1]), reverse=True,
        ):
            lines.append(f"**Category: {cat}** ({len(recs)} records)")
            lines.append("")
            examples = recs[:3]
            for r in examples:
                raw = r.get("raw_text", "") or ""
                item_id = r.get("item_id", "?")
                cond = r.get("condition", "?")
                lines.append(f"- `{item_id}` / `{cond}`:")
                lines.append(f"  ```")
                lines.append(f"  {truncate(raw, 200)}")
                lines.append(f"  ```")
            if len(recs) > 3:
                lines.append(f"  *(... {len(recs) - 3} more)*")
            lines.append("")

    # Perfect models
    perfect = [d for d in suite_model_data if d["n_failures"] == 0]
    if perfect:
        lines.append("## Models with Perfect Parse Rate (100%)")
        lines.append("")
        lines.append("| Suite | Model | Records |")
        lines.append("|-------|-------|--------:|")
        for d in sorted(perfect, key=lambda x: (x["suite"], x["short"])):
            lines.append(f"| {d['suite']} | {d['short']} | {d['n_records']} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate parse failure diagnostic report",
    )
    p.add_argument(
        "--results_dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "full_benchmark",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: <results_dir>/parse_failure_report.md)",
    )
    args = p.parse_args()

    report = generate_report(args.results_dir)

    out_path = args.out or (args.results_dir / "parse_failure_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Report written to {out_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
