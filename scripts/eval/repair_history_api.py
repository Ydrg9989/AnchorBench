#!/usr/bin/env python3
"""Offline repair of API History results: inject missing anchor_value.

The API History runner (run_api_benchmark.py) stored stage1_raw but never
parsed it into an integer or set anchor_value.  This script retroactively
parses stage1_raw using the same parse_response function as the local
evaluator and injects anchor_value + stage1_answer, then recomputes
summary metrics.

No API calls or GPU required.

Usage:
    PYTHONPATH=src python scripts/eval/repair_history_api.py
    PYTHONPATH=src python scripts/eval/repair_history_api.py --results_dirs results/api_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench_eval.evaluator import parse_response, write_and_summarize
from anchorbench_eval.metrics import compute_unified_metrics

log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIRS = [
    "results/api_benchmark",
    "results/api_smoke",
]


def repair_model_dir(model_dir: Path, *, dry_run: bool = False) -> dict | None:
    """Repair a single model's History results.jsonl in place."""
    results_path = model_dir / "results.jsonl"
    if not results_path.exists():
        return None

    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return None

    n_repaired = 0
    n_already_ok = 0
    n_parse_fail = 0

    for rec in records:
        if rec.get("condition") == "control":
            continue

        if rec.get("anchor_value") is not None:
            n_already_ok += 1
            continue

        stage1_raw = rec.get("stage1_raw", "")
        if not stage1_raw:
            continue

        stage1_answer, _, _ = parse_response(stage1_raw, "")

        if stage1_answer is None:
            n_parse_fail += 1
            continue

        use_as_anchor = rec["condition"].startswith(
            ("plausible_", "irrelevant_"),
        )
        rec["anchor_value"] = stage1_answer if use_as_anchor else None
        rec["stage1_answer"] = stage1_answer
        n_repaired += 1

    model_slug = model_dir.name
    log.info(
        "  %s: repaired=%d, already_ok=%d, parse_fail=%d",
        model_slug, n_repaired, n_already_ok, n_parse_fail,
    )

    if dry_run or (n_repaired == 0 and n_already_ok > 0):
        return {"model": model_slug, "repaired": n_repaired, "records": records}

    backup_path = results_path.with_suffix(".jsonl.bak")
    if not backup_path.exists():
        shutil.copy2(results_path, backup_path)
        log.info("  Backed up -> %s", backup_path)

    with open(results_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_and_summarize(records, model_dir, label=f"History (repaired) | {model_slug}")

    return {"model": model_slug, "repaired": n_repaired, "records": records}


def ceiling_diagnosis(all_results: list[dict]) -> None:
    """Print ceiling / saturation analysis across all repaired models."""
    print("\n" + "=" * 70)
    print("CEILING DIAGNOSIS: API History Suite")
    print("=" * 70)

    EPSILON = 3.0

    for entry in all_results:
        model = entry["model"]
        records = entry["records"]

        items: dict[str, dict] = {}
        for r in records:
            items.setdefault(r["item_id"], {})[r["condition"]] = r

        uai_irr, uai_plaus = [], []
        uai_plaus_easy, uai_plaus_hard = [], []
        n_excluded_irr, n_excluded_plaus = 0, 0
        s1_deviations = []

        for iid, conds in items.items():
            ctrl = conds.get("control")
            if not ctrl or ctrl.get("answer_int") is None:
                continue
            y_ctrl = ctrl["answer_int"]

            for cond_name in [
                "irrelevant_low", "irrelevant_high",
                "plausible_low", "plausible_high",
            ]:
                rec = conds.get(cond_name)
                if not rec:
                    continue
                y_anchor = rec.get("answer_int")
                a = rec.get("anchor_value")
                if y_anchor is None or a is None:
                    continue

                s1_deviations.append(abs(a - rec["y_star_evidence"]))

                denom = a - y_ctrl
                shift = y_anchor - y_ctrl

                is_irr = cond_name.startswith("irr")
                if abs(denom) < EPSILON:
                    if is_irr:
                        n_excluded_irr += 1
                    else:
                        n_excluded_plaus += 1
                    continue

                uai_val = shift / denom
                if is_irr:
                    uai_irr.append(uai_val)
                else:
                    uai_plaus.append(uai_val)
                    diff = rec.get("difficulty", "unknown")
                    if diff == "easy":
                        uai_plaus_easy.append(uai_val)
                    elif diff == "hard":
                        uai_plaus_hard.append(uai_val)

        def sm(lst):
            return f"{np.mean(lst):+.4f}" if lst else "N/A"

        disc = None
        if uai_irr and uai_plaus:
            disc = np.mean(uai_plaus) - np.mean(uai_irr)

        print(f"\n--- {model} ---")
        print(f"  UAI_irr:   {sm(uai_irr):>8s}  (n={len(uai_irr)}, excluded={n_excluded_irr})")
        print(f"  UAI_plaus: {sm(uai_plaus):>8s}  (n={len(uai_plaus)}, excluded={n_excluded_plaus})")
        print(f"  Disc_Δ:    {disc:+.4f}" if disc is not None else "  Disc_Δ:    N/A")
        if s1_deviations:
            arr = np.array(s1_deviations)
            print(f"  |stage1 - y*|: mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
                  f"<3: {(arr < 3).sum()}/{len(arr)} ({100*(arr < 3).mean():.0f}%), "
                  f">=10: {(arr >= 10).sum()}/{len(arr)} ({100*(arr >= 10).mean():.0f}%)")
        if uai_plaus_easy or uai_plaus_hard:
            print(f"  Easy UAI_plaus: {sm(uai_plaus_easy):>8s}  (n={len(uai_plaus_easy)})")
            print(f"  Hard UAI_plaus: {sm(uai_plaus_hard):>8s}  (n={len(uai_plaus_hard)})")

    print("\n" + "=" * 70)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Repair API History anchor_value offline")
    p.add_argument(
        "--results_dirs", nargs="+", default=DEFAULT_RESULTS_DIRS,
        help="Result directories containing history/<model>/ subdirs",
    )
    p.add_argument("--dry_run", action="store_true",
                   help="Parse and diagnose without writing changes")
    args = p.parse_args()

    all_results = []

    for results_dir in args.results_dirs:
        history_dir = Path(results_dir) / "history"
        if not history_dir.is_dir():
            log.warning("No history dir at %s, skipping", history_dir)
            continue

        log.info("Repairing %s ...", history_dir)
        for model_dir in sorted(history_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            result = repair_model_dir(model_dir, dry_run=args.dry_run)
            if result:
                all_results.append(result)

    if all_results:
        ceiling_diagnosis(all_results)
    else:
        log.warning("No results found to repair.")


if __name__ == "__main__":
    main()
