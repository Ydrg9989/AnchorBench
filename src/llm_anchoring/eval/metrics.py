"""Anchoring metrics: condition summaries, paired IAS, bootstrap CIs."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np


def condition_summary(records: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-condition aggregate: parse/format rates, MAE, median absolute error."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cond[r["icl_condition"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for cond, recs in sorted(by_cond.items()):
        n = len(recs)
        pok = sum(1 for r in recs if r["parse_ok"])
        fok = sum(1 for r in recs if r["format_ok"])
        errs = [r["abs_error"] for r in recs if r["abs_error"] is not None]
        acol = sum(1 for r in recs if r.get("anchor_collision", False))
        out[cond] = {
            "n": n,
            "parse_rate": pok / n if n else 0,
            "format_rate": fok / n if n else 0,
            "mae": float(np.mean(errs)) if errs else None,
            "median_abs_error": float(np.median(errs)) if errs else None,
            "anchor_collision_rate": acol / n if n else 0,
        }
    return out


def compute_paired_anchoring(records: list[dict]) -> dict[str, Any]:
    """Compute IAS and OutlierShift per base_id, with bootstrap 95 % CI.

    Returns a dict with ``per_base`` table, aggregate stats, and CI.
    """
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    base_meta: dict[str, dict] = {}
    for r in records:
        bid = r["base_id"]
        if bid not in base_meta:
            base_meta[bid] = {
                "domain": r.get("domain"),
                "template_id": r.get("template_id"),
            }
        if r["parse_ok"]:
            groups[(bid, r["icl_condition"])].append(r["parsed_value"])

    medians = {k: float(np.median(v)) for k, v in groups.items()}

    per_base: list[dict] = []
    ias_vals: list[float] = []
    osh_vals: list[float] = []

    for bid in sorted(base_meta):
        meta = base_meta[bid]
        low = medians.get((bid, "low_anchor_demo"))
        high = medians.get((bid, "high_anchor_demo"))
        outlier = medians.get((bid, "outlier_anchor_demo"))
        no_demo = medians.get((bid, "no_demo"))
        placeholder = medians.get((bid, "placeholder_demo"))

        row: dict[str, Any] = {
            "base_id": bid,
            "domain": meta.get("domain"),
            "template_id": meta.get("template_id"),
            "median_no_demo": no_demo,
            "median_low": low,
            "median_high": high,
            "median_outlier": outlier,
            "median_placeholder": placeholder,
        }

        if low is not None and high is not None:
            ias = high - low
            row["IAS"] = ias
            ias_vals.append(ias)

        if low is not None and outlier is not None:
            osh = outlier - low
            row["OutlierShift"] = osh
            osh_vals.append(osh)

        per_base.append(row)

    result: dict[str, Any] = {
        "per_base": per_base,
        "ias_mean": float(np.mean(ias_vals)) if ias_vals else None,
        "ias_median": float(np.median(ias_vals)) if ias_vals else None,
        "outlier_shift_mean": float(np.mean(osh_vals)) if osh_vals else None,
        "outlier_shift_median": float(np.median(osh_vals)) if osh_vals else None,
    }
    if ias_vals:
        result["ias_ci_95"] = _bootstrap_ci(ias_vals)
    return result


def _bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[float]:
    """Percentile bootstrap 95 % CI for the mean over *values*."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_boot)
    )
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return [round(lo, 6), round(hi, 6)]
