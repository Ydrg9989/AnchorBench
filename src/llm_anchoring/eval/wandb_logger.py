"""Weights & Biases logging helpers for anchoring evaluation."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    HAS_WANDB = False


def init_run(
    config: dict[str, Any],
    run_name: str,
    project: str,
    entity: str | None = None,
) -> Any | None:
    """Start a W&B run. Returns the run object, or None if wandb is missing."""
    if not HAS_WANDB:
        logger.warning("wandb not installed — skipping logging")
        return None
    return wandb.init(project=project, entity=entity, name=run_name, config=config)


def log_summary(summary: dict[str, Any], run: Any | None = None) -> None:
    """Log scalar metrics to W&B summary."""
    if run is None:
        return
    flat = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
    wandb.summary.update(flat)


def log_condition_metrics(
    cond_summary: dict[str, dict], run: Any | None = None
) -> None:
    """Log per-condition parse/format rates and MAE."""
    if run is None:
        return
    for cond, m in cond_summary.items():
        wandb.summary[f"{cond}/parse_rate"] = m["parse_rate"]
        wandb.summary[f"{cond}/format_rate"] = m["format_rate"]
        if m["mae"] is not None:
            wandb.summary[f"{cond}/mae"] = m["mae"]
        if m.get("anchor_collision_rate") is not None:
            wandb.summary[f"{cond}/anchor_collision_rate"] = m["anchor_collision_rate"]


def log_per_base_table(per_base: list[dict], run: Any | None = None) -> None:
    """Upload the per-base IAS table as a W&B Table."""
    if run is None or not per_base:
        return
    cols = list(per_base[0].keys())
    table = wandb.Table(columns=cols)
    for row in per_base:
        table.add_data(*[row.get(c) for c in cols])
    wandb.log({"per_base_ias": table})


def log_dataset_info(dataset_path: str, run: Any | None = None) -> None:
    """Record dataset path and SHA-1 hash in W&B config."""
    if run is None:
        return
    p = Path(dataset_path)
    if p.exists():
        sha = hashlib.sha1(p.read_bytes()).hexdigest()
        wandb.config.update({"dataset_sha1": sha, "dataset_path": str(p)})


def finish(run: Any | None = None) -> None:
    if run is not None:
        wandb.finish()
