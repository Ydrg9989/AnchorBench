#!/usr/bin/env python3
"""Package extension experiment results into summary tables and markdown.

Reads results from:
  - results/mitigation_ignore_anchor/
  - results/icl_numeric_api/
  - results/decoding_sampling_robustness/

Compares with baseline results from:
  - results/full_benchmark/
  - results/api_benchmark/
  - results/icl_dist_core/

Outputs to results/extension_summary/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorbench_eval.metrics import compute_unified_metrics

log = logging.getLogger(__name__)


def load_records(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def metrics_row(records: list[dict], label: str = "") -> dict:
    if not records:
        return {"label": label, "n": 0}
    m = compute_unified_metrics(records)
    return {
        "label": label,
        "n": len(records),
        "parse_rate": m.get("parse_rate", 0),
        "mae_control": m.get("mae_control"),
        "acc10_control": m.get("acc10_control"),
        "uai_irr": m.get("uai_irr"),
        "uai_plaus": m.get("uai_plaus"),
        "disc_delta": m.get("disc_delta"),
        "tar_irr": m.get("tar_irr"),
        "tar_plaus": m.get("tar_plaus"),
    }


def _fmt(v, prec=3):
    if v is None:
        return "---"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def package_mitigation(results_root: Path):
    """Compare mitigation vs baseline for each model x suite."""
    mit_dir = results_root / "mitigation_ignore_anchor"
    if not mit_dir.exists():
        log.warning("No mitigation results found at %s", mit_dir)
        return []

    rows = []
    for suite_dir in sorted(mit_dir.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite = suite_dir.name
        for model_dir in sorted(suite_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_slug = model_dir.name

            mit_records = load_records(model_dir / "results.jsonl")
            mit_m = metrics_row(mit_records, f"{suite}/{model_slug} (mitigated)")

            base_path = _find_baseline(suite, model_slug, results_root)
            base_records = load_records(base_path) if base_path else []
            base_m = metrics_row(base_records, f"{suite}/{model_slug} (baseline)")

            rows.append({
                "suite": suite,
                "model": model_slug,
                "condition": "baseline",
                **{k: base_m.get(k) for k in ["uai_irr", "uai_plaus", "disc_delta", "mae_control", "parse_rate"]},
            })
            rows.append({
                "suite": suite,
                "model": model_slug,
                "condition": "mitigated",
                **{k: mit_m.get(k) for k in ["uai_irr", "uai_plaus", "disc_delta", "mae_control", "parse_rate"]},
            })

    return rows


def _find_baseline(suite: str, model_slug: str, results_root: Path) -> Path | None:
    """Find baseline results.jsonl for a model+suite."""
    candidates = [
        results_root / "full_benchmark" / suite / model_slug / "results.jsonl",
        results_root / "api_benchmark" / suite / model_slug / "results.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def package_icl_dist_api(results_root: Path):
    """Compare standard ICL vs ICL-dist for API models."""
    icl_api_dir = results_root / "icl_numeric_api"
    if not icl_api_dir.exists():
        log.warning("No ICL-dist API results at %s", icl_api_dir)
        return []

    rows = []
    for model_dir in sorted(icl_api_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_slug = model_dir.name

        dist_records = load_records(model_dir / "results.jsonl")
        dist_m = metrics_row(dist_records)

        std_path = results_root / "api_benchmark" / "icl" / model_slug / "results.jsonl"
        std_records = load_records(std_path)
        std_m = metrics_row(std_records)

        rows.append({
            "model": model_slug,
            "variant": "ICL-std",
            **{k: std_m.get(k) for k in ["uai_irr", "uai_plaus", "disc_delta", "mae_control", "parse_rate"]},
        })
        rows.append({
            "model": model_slug,
            "variant": "ICL-dist",
            **{k: dist_m.get(k) for k in ["uai_irr", "uai_plaus", "disc_delta", "mae_control", "parse_rate"]},
        })

    return rows


def package_sampling(results_root: Path):
    """Compare greedy vs sampling for each model x suite."""
    samp_dir = results_root / "decoding_sampling_robustness"
    if not samp_dir.exists():
        log.warning("No sampling results at %s", samp_dir)
        return []

    rows = []
    for suite_dir in sorted(samp_dir.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite = suite_dir.name
        for model_dir in sorted(suite_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_slug = model_dir.name

            greedy_records = load_records(model_dir / "greedy" / "results.jsonl")
            greedy_m = metrics_row(greedy_records)

            sample_runs = []
            for seed_dir in sorted(model_dir.iterdir()):
                if seed_dir.name.startswith("t0.7"):
                    recs = load_records(seed_dir / "results.jsonl")
                    if recs:
                        sample_runs.append(metrics_row(recs))

            rows.append({
                "suite": suite,
                "model": model_slug,
                "decoding": "greedy",
                **{k: greedy_m.get(k) for k in ["uai_irr", "uai_plaus", "disc_delta", "mae_control", "parse_rate"]},
            })

            if sample_runs:
                import numpy as np
                for key in ["uai_irr", "uai_plaus", "disc_delta", "mae_control"]:
                    vals = [r[key] for r in sample_runs if r.get(key) is not None]
                    if vals:
                        mean_val = float(np.mean(vals))
                        std_val = float(np.std(vals))
                    else:
                        mean_val, std_val = None, None
                for si, sr in enumerate(sample_runs):
                    rows.append({
                        "suite": suite,
                        "model": model_slug,
                        "decoding": f"sample_s{si}",
                        **{k: sr.get(k) for k in ["uai_irr", "uai_plaus", "disc_delta", "mae_control", "parse_rate"]},
                    })

    return rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %s (%d rows)", path, len(rows))


def write_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Wrote %s", path)


def write_markdown_report(
    mit_rows,
    icl_rows,
    samp_rows,
    path: Path,
    results_root: Path,
    out_dir: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Extension Experiment Results\n"]

    lines.append("## 1. Mitigation Baseline (ignore-anchor reminder)\n")
    if mit_rows:
        lines.append("| Suite | Model | Condition | UAI_irr | UAI_pls | Disc_Δ | MAE_ctrl |")
        lines.append("|-------|-------|-----------|---------|---------|--------|----------|")
        for r in mit_rows:
            lines.append(f"| {r['suite']} | {r['model'][:20]} | {r['condition']} | "
                          f"{_fmt(r.get('uai_irr'))} | {_fmt(r.get('uai_plaus'))} | "
                          f"{_fmt(r.get('disc_delta'))} | {_fmt(r.get('mae_control'), 1)} |")
        lines.append("")
    else:
        lines.append("*No results available.*\n")

    lines.append("## 2. ICL Numeric Positive Control (API Models)\n")
    if icl_rows:
        lines.append("| Model | Variant | UAI_irr | UAI_pls | Disc_Δ | MAE_ctrl |")
        lines.append("|-------|---------|---------|---------|--------|----------|")
        for r in icl_rows:
            lines.append(f"| {r['model'][:25]} | {r['variant']} | "
                          f"{_fmt(r.get('uai_irr'))} | {_fmt(r.get('uai_plaus'))} | "
                          f"{_fmt(r.get('disc_delta'))} | {_fmt(r.get('mae_control'), 1)} |")
        lines.append("")
    else:
        lines.append("*No results available.*\n")

    lines.append("## 3. Sampling Robustness (greedy vs temp=0.7)\n")
    if samp_rows:
        lines.append("| Suite | Model | Decoding | UAI_irr | UAI_pls | Disc_Δ | MAE_ctrl |")
        lines.append("|-------|-------|----------|---------|---------|--------|----------|")
        for r in samp_rows:
            lines.append(f"| {r['suite']} | {r['model'][:20]} | {r['decoding']} | "
                          f"{_fmt(r.get('uai_irr'))} | {_fmt(r.get('uai_plaus'))} | "
                          f"{_fmt(r.get('disc_delta'))} | {_fmt(r.get('mae_control'), 1)} |")
        lines.append("")
    else:
        lines.append("*No results available.*\n")

    lines.append("## File Locations\n")
    rr = results_root.as_posix()
    lines.append(f"- Mitigation: `{rr}/mitigation_ignore_anchor/`")
    lines.append(f"- ICL-dist API: `{rr}/icl_numeric_api/`")
    lines.append(f"- Sampling: `{rr}/decoding_sampling_robustness/`")
    lines.append(f"- Summaries: `{out_dir.as_posix()}/`")
    lines.append("")

    path.write_text("\n".join(lines))
    log.info("Wrote %s", path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_root",
        default="results",
        help="Root directory containing mitigation, ICL-dist, sampling, and baseline runs",
    )
    p.add_argument(
        "--out_dir",
        default="results/extension_summary",
        help="Output directory for CSV, JSON, and markdown report",
    )
    args = p.parse_args()
    results_root = ROOT / args.results_root
    out_dir = ROOT / args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    mit_rows = package_mitigation(results_root)
    icl_rows = package_icl_dist_api(results_root)
    samp_rows = package_sampling(results_root)

    write_csv(mit_rows, out_dir / "mitigation_comparison.csv")
    write_csv(icl_rows, out_dir / "icl_dist_api_comparison.csv")
    write_csv(samp_rows, out_dir / "sampling_robustness.csv")

    all_data = {
        "mitigation": mit_rows,
        "icl_dist_api": icl_rows,
        "sampling": samp_rows,
    }
    write_json(all_data, out_dir / "all_extension_results.json")
    write_markdown_report(
        mit_rows,
        icl_rows,
        samp_rows,
        out_dir / "EXTENSION_REPORT.md",
        results_root,
        out_dir,
    )

    log.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
