#!/usr/bin/env python3
"""Parse-policy sensitivity analysis for AnchorBench.

Re-parses raw model outputs under different parsing strategies and
recomputes metrics to show how robust results are to parsing choices.

Policies:
  strict_regex     — only regex (parse_answer_int), no fallback
  regex_final      — regex, then final_answer patterns
  full_cascade     — regex → xml_tag → final_answer → last_number
  last_only        — last integer in [0,100] only
  clamped_cascade  — full_cascade with out-of-range values clamped to [0,100]

Output: a sensitivity table showing UAI/TAR under each policy.

Usage:
    PYTHONPATH=src python scripts/eval/parse_sensitivity.py \\
        results/full_benchmark/external/model_slug/results.jsonl

    # Batch mode: scan all results under a directory
    PYTHONPATH=src python scripts/eval/parse_sensitivity.py \\
        --results_dir results/full_benchmark --out results/full_benchmark/sensitivity.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench_eval.metrics import compute_unified_metrics
from anchorbench_eval.parsing import (
    parse_answer_int,
    parse_final_answer,
    parse_last_number,
    parse_xml_answer,
)

POLICIES = {
    "strict_regex": ["regex"],
    "regex_final": ["regex", "final_answer"],
    "full_cascade": ["regex", "xml_tag", "final_answer", "last_number"],
    "last_only": ["last_number"],
    "clamped_cascade": ["regex", "xml_tag", "final_answer", "last_number"],
}

CLAMP_POLICIES = {"clamped_cascade"}

STRATEGY_FNS = {
    "regex": lambda raw, prompt, clamp=False: parse_answer_int(raw, prompt, clamp=clamp),
    "xml_tag": lambda raw, _, clamp=False: parse_xml_answer(raw, clamp=clamp),
    "final_answer": lambda raw, _, clamp=False: parse_final_answer(raw, clamp=clamp),
    "last_number": lambda raw, _, clamp=False: parse_last_number(raw, clamp=clamp),
}

SUITES = ("external", "icl", "rag", "tool", "history")


def reparse_records(
    records: list[dict], policy_name: str,
) -> list[dict]:
    """Re-parse all records under a given policy, returning modified copies."""
    strategies = POLICIES[policy_name]
    use_clamp = policy_name in CLAMP_POLICIES
    out = []
    for r in records:
        r2 = copy.copy(r)
        raw = r.get("raw_text", "") or r.get("raw_output", "") or r.get("model_output", "") or ""
        prompt = r.get("prompt_text", "")

        answer = None
        parsed_ok = False
        used_strategy = "failed"

        for strat_name in strategies:
            fn = STRATEGY_FNS[strat_name]
            answer, parsed_ok = fn(raw, prompt, clamp=use_clamp)
            if parsed_ok:
                if use_clamp:
                    strict_answer, strict_ok = STRATEGY_FNS[strat_name](raw, prompt, clamp=False)
                    used_strategy = f"{strat_name}_clamped" if not strict_ok else strat_name
                else:
                    used_strategy = strat_name
                break

        r2["answer_int"] = answer
        r2["parsed_ok"] = parsed_ok
        r2["parse_strategy"] = used_strategy
        out.append(r2)
    return out


def fmt(v: float | None, d: int = 3) -> str:
    return f"{v:.{d}f}" if v is not None else "---"


def fmt_pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "---"


def discover_results(results_dir: Path) -> list[tuple[str, str, Path]]:
    """Find all results.jsonl under results_dir/<suite>/<model>/."""
    found = []
    for suite in SUITES:
        suite_dir = results_dir / suite
        if not suite_dir.is_dir():
            continue
        for p in sorted(suite_dir.rglob("results.jsonl")):
            rel = p.relative_to(suite_dir)
            parts = list(rel.parts)
            if len(parts) >= 2:
                model_slug = parts[0]
            elif len(parts) == 1:
                model_slug = suite_dir.name
                if model_slug in SUITES:
                    continue
            else:
                continue
            found.append((suite, model_slug, p))
    return found


def run_sensitivity(
    results_path: Path, epsilon: float,
) -> dict:
    """Run sensitivity analysis on a single results file, return metrics dict."""
    with open(results_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        return {}

    all_policy_metrics = {}
    for policy_name in POLICIES:
        reparsed = reparse_records(records, policy_name)
        m = compute_unified_metrics(reparsed, epsilon=epsilon)
        all_policy_metrics[policy_name] = m

    original_m = compute_unified_metrics(records, epsilon=epsilon)

    return {
        "source": str(results_path),
        "n_records": len(records),
        "epsilon": epsilon,
        "original": original_m,
        "policies": all_policy_metrics,
    }


def print_sensitivity(label: str, data: dict) -> None:
    """Print a sensitivity table for a single results file."""
    print(f"\n{'=' * 100}")
    print(f"  Parse-policy sensitivity: {label}")
    print(f"  ({data['n_records']} records)")
    print(f"{'=' * 100}")

    header = (
        f"  {'Policy':<20} {'Parse%':>7} {'MAE_c':>6} {'UAI_irr':>8} {'UAI_pls':>8} "
        f"{'TAR_irr':>8} {'TAR_pls':>8} {'Disc_Δ':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for policy_name in POLICIES:
        m = data["policies"][policy_name]
        print(
            f"  {policy_name:<20} "
            f"{fmt_pct(m['parse_rate']):>7} "
            f"{fmt(m.get('mae_control'), 2):>6} "
            f"{fmt(m['uai_irr']):>8} "
            f"{fmt(m['uai_plaus']):>8} "
            f"{fmt(m['tar_irr']):>8} "
            f"{fmt(m['tar_plaus']):>8} "
            f"{fmt(m['disc_delta']):>8}"
        )

    m = data["original"]
    print(
        f"  {'(original)':<20} "
        f"{fmt_pct(m['parse_rate']):>7} "
        f"{fmt(m.get('mae_control'), 2):>6} "
        f"{fmt(m['uai_irr']):>8} "
        f"{fmt(m['uai_plaus']):>8} "
        f"{fmt(m['tar_irr']):>8} "
        f"{fmt(m['tar_plaus']):>8} "
        f"{fmt(m['disc_delta']):>8}"
    )
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parse-policy sensitivity analysis for AnchorBench",
    )
    p.add_argument("results", type=Path, nargs="*", help="Path(s) to results.jsonl")
    p.add_argument(
        "--results_dir", type=Path, default=None,
        help="Scan all results.jsonl under this directory (batch mode)",
    )
    p.add_argument("--epsilon", type=float, default=3.0)
    p.add_argument("--out", type=Path, default=None, help="Write sensitivity JSON")
    p.add_argument(
        "--only_imperfect", action="store_true",
        help="In batch mode, only show models with parse_rate < 100%%",
    )
    args = p.parse_args()

    paths: list[tuple[str, Path]] = []
    if args.results_dir:
        for suite, model_slug, rpath in discover_results(args.results_dir):
            paths.append((f"{suite}/{model_slug}", rpath))
    for rpath in (args.results or []):
        paths.append((str(rpath.name), rpath))

    if not paths:
        print("No results files found.")
        sys.exit(1)

    all_data = {}
    for label, results_path in paths:
        data = run_sensitivity(results_path, args.epsilon)
        if not data:
            print(f"WARNING: empty {results_path}")
            continue

        orig_rate = data["original"].get("parse_rate", 1.0)
        if args.only_imperfect and orig_rate >= 1.0:
            continue

        all_data[label] = data
        print_sensitivity(label, data)

    # Strict vs clamped comparison summary
    print(f"\n{'=' * 110}")
    print("  Strict vs Clamped Comparison (full_cascade vs clamped_cascade)")
    print(f"{'=' * 110}")
    header = (
        f"  {'Suite/Model':<35} {'Parse_strict':>12} {'Parse_clamp':>12} "
        f"{'ΔUAI_irr':>9} {'ΔUAI_pls':>9} {'ΔMAE_c':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, data in sorted(all_data.items()):
        strict = data["policies"]["full_cascade"]
        clamped = data["policies"]["clamped_cascade"]

        def delta(key: str) -> str:
            s = strict.get(key)
            c = clamped.get(key)
            if s is None or c is None:
                return "---"
            d = c - s
            return f"{d:+.4f}"

        print(
            f"  {label:<35} "
            f"{fmt_pct(strict['parse_rate']):>12} "
            f"{fmt_pct(clamped['parse_rate']):>12} "
            f"{delta('uai_irr'):>9} "
            f"{delta('uai_plaus'):>9} "
            f"{delta('mae_control'):>8}"
        )
    print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
        print(f"  Written to {args.out}")


if __name__ == "__main__":
    main()
