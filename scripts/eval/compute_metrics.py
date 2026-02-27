#!/usr/bin/env python3
"""CLI: compute anchoring metrics from generations, save summaries, log to W&B."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llm_anchoring.eval.metrics import compute_paired_anchoring, condition_summary
from llm_anchoring.eval.wandb_logger import (
    finish,
    init_run,
    log_condition_metrics,
    log_dataset_info,
    log_per_base_table,
    log_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute anchoring metrics")
    p.add_argument("--generations", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.generations.parent

    records = [
        json.loads(l)
        for l in args.generations.read_text().strip().split("\n")
    ]
    logger.info("loaded %d generation records", len(records))

    cond_summ = condition_summary(records)
    logger.info("=== Condition Summary ===")
    for cond, m in cond_summ.items():
        mae_str = f"{m['mae']:.2f}" if m["mae"] is not None else "N/A"
        acr = m.get("anchor_collision_rate", 0)
        logger.info(
            "  %-25s parse=%.3f  fmt=%.3f  mae=%s  acol=%.3f  n=%d",
            cond, m["parse_rate"], m["format_rate"], mae_str, acr, m["n"],
        )

    anchoring = compute_paired_anchoring(records)
    logger.info("=== Anchoring (paired by base_id) ===")
    logger.info(
        "  IAS  mean=%.4f  median=%.4f",
        anchoring.get("ias_mean") or 0,
        anchoring.get("ias_median") or 0,
    )
    if anchoring.get("ias_ci_95"):
        logger.info("  IAS  95%% CI: [%.4f, %.4f]", *anchoring["ias_ci_95"])
    logger.info(
        "  OutlierShift  mean=%.4f  median=%.4f",
        anchoring.get("outlier_shift_mean") or 0,
        anchoring.get("outlier_shift_median") or 0,
    )

    summary = {
        "condition_summary": cond_summ,
        "ias_mean": anchoring.get("ias_mean"),
        "ias_median": anchoring.get("ias_median"),
        "ias_ci_95": anchoring.get("ias_ci_95"),
        "outlier_shift_mean": anchoring.get("outlier_shift_mean"),
        "outlier_shift_median": anchoring.get("outlier_shift_median"),
        "n_records": len(records),
        "n_base_ids": len(anchoring["per_base"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    per_base = anchoring["per_base"]
    if per_base:
        csv_path = out_dir / "per_base.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_base[0].keys())
            w.writeheader()
            w.writerows(per_base)
        logger.info("per-base table → %s (%d rows)", csv_path, len(per_base))

    if not args.no_wandb:
        _log_wandb(cfg, out_dir, cond_summ, anchoring, summary)

    logger.info("done")


def _log_wandb(
    cfg: dict, out_dir: Path, cond_summ: dict, anchoring: dict, summary: dict
) -> None:
    run_cfg_path = out_dir / "run_config.json"
    run_cfg = json.loads(run_cfg_path.read_text()) if run_cfg_path.exists() else {}
    wb = cfg.get("wandb", {})
    model_short = run_cfg.get("model", "unknown").split("/")[-1]
    run_name = (
        f"{model_short}_{run_cfg.get('decoding', '?')}"
        f"_s{run_cfg.get('seed', '?')}_n{run_cfg.get('n_samples', '?')}"
    )
    run = init_run(
        config={**run_cfg, **cfg},
        run_name=run_name,
        project=wb.get("project", "llm-anchoring"),
        entity=wb.get("entity"),
    )
    log_dataset_info(cfg.get("dataset_path", ""), run)
    log_summary(
        {
            "ias_mean": summary.get("ias_mean"),
            "ias_median": summary.get("ias_median"),
            "outlier_shift_mean": summary.get("outlier_shift_mean"),
        },
        run,
    )
    log_condition_metrics(cond_summ, run)
    log_per_base_table(anchoring["per_base"], run)
    finish(run)
    logger.info("W&B run logged")


if __name__ == "__main__":
    main()
