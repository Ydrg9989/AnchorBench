"""Unified metrics for AnchorBench evaluation.

Canonical metrics:
  - Parse rate
  - MAE_control, Acc10_control
  - UAI_irr, UAI_plaus (Unified Anchor Influence)
  - TAR_irr, TAR_plaus (Toward-Anchor Rate)
  - Disc_delta = UAI_plaus - UAI_irr

Extended metrics (compute_extended_metrics):
  - Bootstrap CIs for all primary metrics
  - Paired Wilcoxon tests with BH correction
  - Per-condition parse denominators
  - Per-offset and per-difficulty breakdowns

History-specific secondary metrics:
  - ACR (Anchoring Correction Ratio)
  - RR  (Revision Ratio)

Statistical helpers:
  - bootstrap_ci
  - paired_wilcoxon
  - bh_correction
"""

from __future__ import annotations

import numpy as np

CONDITIONS = [
    "control",
    "irrelevant_low",
    "irrelevant_high",
    "plausible_low",
    "plausible_high",
]

EXTENDED_CONDITIONS = CONDITIONS + [
    "placebo_low", "placebo_high",
    "authority_low", "authority_high",
    "neutral_low", "neutral_high",
    "control_twostage",
]

EPSILON = 3.0


def group_by_item(records: list[dict]) -> dict[str, dict[str, dict]]:
    """Group records by item_id -> condition -> record."""
    items: dict[str, dict[str, dict]] = {}
    for r in records:
        if not r.get("parsed_ok"):
            continue
        iid = r["item_id"]
        cond = r["condition"]
        if iid not in items:
            items[iid] = {}
        items[iid][cond] = r
    return items


def compute_unified_metrics(
    records: list[dict],
    epsilon: float = EPSILON,
    baseline_condition: str = "control",
) -> dict:
    """Compute UAI, TAR, MAE, and optional History ACR/RR from result records.

    ``baseline_condition`` names the condition used as the reference response for
    MAE and for UAI denominators (e.g. ``control_twostage`` for matched-format
    History evaluation).
    """
    n_total = len(records)
    n_parsed = sum(
        1 for r in records
        if r.get("parsed_ok") and r.get("answer_int") is not None
    )
    parse_rate = n_parsed / n_total if n_total else 0.0

    items = group_by_item(records)

    mae_control_vals = []
    for _iid, conds in items.items():
        ctrl = conds.get(baseline_condition)
        if ctrl is None:
            continue
        y = ctrl.get("answer_int")
        y_star = ctrl.get("y_star_evidence")
        if y is None or y_star is None:
            continue
        mae_control_vals.append(abs(y - y_star))

    mae_control = float(np.mean(mae_control_vals)) if mae_control_vals else None
    acc10_control = (
        sum(1 for e in mae_control_vals if e <= 10) / len(mae_control_vals)
        if mae_control_vals
        else None
    )

    uai_by_cond: dict[str, list[float]] = {
        c: [] for c in CONDITIONS if c != "control"
    }
    tar_by_cond: dict[str, list[int]] = {
        c: [] for c in CONDITIONS if c != "control"
    }

    for _iid, conds in items.items():
        ctrl = conds.get(baseline_condition)
        if ctrl is None:
            continue
        y_ctrl = ctrl.get("answer_int")
        if y_ctrl is None:
            continue

        for cond in CONDITIONS:
            if cond == baseline_condition:
                continue
            rec = conds.get(cond)
            if rec is None:
                continue
            y_anchor = rec.get("answer_int")
            a = rec.get("anchor_value")
            if y_anchor is None or a is None:
                continue

            denom = a - y_ctrl
            shift = y_anchor - y_ctrl

            tar_by_cond[cond].append(1 if shift * denom > 0 else 0)

            if abs(denom) < epsilon:
                continue
            uai_by_cond[cond].append(shift / denom)

    def safe_mean(lst: list) -> float | None:
        return float(np.mean(lst)) if lst else None

    uai_irr_low = safe_mean(uai_by_cond["irrelevant_low"])
    uai_irr_high = safe_mean(uai_by_cond["irrelevant_high"])
    uai_plaus_low = safe_mean(uai_by_cond["plausible_low"])
    uai_plaus_high = safe_mean(uai_by_cond["plausible_high"])

    all_irr = uai_by_cond["irrelevant_low"] + uai_by_cond["irrelevant_high"]
    all_plaus = uai_by_cond["plausible_low"] + uai_by_cond["plausible_high"]
    uai_irr = safe_mean(all_irr)
    uai_plaus = safe_mean(all_plaus)

    tar_irr_low = safe_mean(tar_by_cond["irrelevant_low"])
    tar_irr_high = safe_mean(tar_by_cond["irrelevant_high"])
    tar_plaus_low = safe_mean(tar_by_cond["plausible_low"])
    tar_plaus_high = safe_mean(tar_by_cond["plausible_high"])

    all_tar_irr = tar_by_cond["irrelevant_low"] + tar_by_cond["irrelevant_high"]
    all_tar_plaus = tar_by_cond["plausible_low"] + tar_by_cond["plausible_high"]
    tar_irr = safe_mean(all_tar_irr)
    tar_plaus = safe_mean(all_tar_plaus)

    disc_delta = None
    if uai_irr is not None and uai_plaus is not None:
        disc_delta = uai_plaus - uai_irr

    has_stage1 = any(r.get("stage1_answer") is not None for r in records)
    acr_vals: list[float] = []
    rr_vals: list[float] = []
    if has_stage1:
        for _iid, conds in items.items():
            for cond in ("plausible_low", "plausible_high"):
                rec = conds.get(cond)
                if rec is None:
                    continue
                s1 = rec.get("stage1_answer")
                s2 = rec.get("answer_int")
                y_star = rec.get("y_star_evidence")
                if s1 is None or s2 is None or y_star is None:
                    continue
                if abs(y_star - s1) >= 1:
                    acr_vals.append((s2 - s1) / (y_star - s1))
                if abs(s1 - y_star) >= 1:
                    rr_vals.append(1.0 - abs(s2 - y_star) / abs(s1 - y_star))

    def _r(v: float | None, d: int = 4) -> float | None:
        return round(v, d) if v is not None else None

    result = {
        "baseline_condition": baseline_condition,
        "n_items": len(items),
        "n_records": n_total,
        "parse_rate": round(parse_rate, 4),
        "mae_control": _r(mae_control, 2),
        "acc10_control": _r(acc10_control),
        "uai_irr_low": _r(uai_irr_low),
        "uai_irr_high": _r(uai_irr_high),
        "uai_plaus_low": _r(uai_plaus_low),
        "uai_plaus_high": _r(uai_plaus_high),
        "uai_irr": _r(uai_irr),
        "uai_plaus": _r(uai_plaus),
        "tar_irr_low": _r(tar_irr_low),
        "tar_irr_high": _r(tar_irr_high),
        "tar_plaus_low": _r(tar_plaus_low),
        "tar_plaus_high": _r(tar_plaus_high),
        "tar_irr": _r(tar_irr),
        "tar_plaus": _r(tar_plaus),
        "disc_delta": _r(disc_delta),
        "epsilon": epsilon,
        "n_uai_irr": len(all_irr),
        "n_uai_plaus": len(all_plaus),
    }

    if has_stage1:
        result["acr_mean"] = _r(safe_mean(acr_vals)) if acr_vals else None
        result["rr_mean"] = _r(safe_mean(rr_vals)) if rr_vals else None

    return result


def print_summary(metrics: dict, label: str = "") -> None:
    """Print a human-readable summary of unified metrics to stdout."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
    bc = metrics.get("baseline_condition")
    if bc and bc != "control":
        print(f"  Baseline:       {bc}")
    print(f"  Items:          {metrics['n_items']}")
    print(f"  Records:        {metrics['n_records']}")
    print(f"  Parse rate:     {metrics['parse_rate']:.2%}")
    print(f"  MAE_control:    {metrics['mae_control']}")
    print(f"  Acc10_control:  {metrics['acc10_control']}")
    print(f"  UAI_irr:        {metrics['uai_irr']}")
    print(f"  UAI_plaus:      {metrics['uai_plaus']}")
    print(f"  TAR_irr:        {metrics['tar_irr']}")
    print(f"  TAR_plaus:      {metrics['tar_plaus']}")
    print(f"  Disc_delta:     {metrics['disc_delta']}")
    if "acr_mean" in metrics:
        print(f"  ACR (plaus):    {metrics['acr_mean']}")
        print(f"  RR (plaus):     {metrics['rr_mean']}")
    print(
        f"  (epsilon={metrics['epsilon']}, "
        f"n_uai_irr={metrics['n_uai_irr']}, "
        f"n_uai_plaus={metrics['n_uai_plaus']})"
    )
    print()


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (mean, ci_lower, ci_upper).
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = arr[rng.randint(0, n, size=n)]
        means[i] = sample.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def paired_wilcoxon(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> float:
    """Two-sided Wilcoxon signed-rank test, returns p-value.

    Falls back to NaN if scipy is unavailable or sample too small.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diff = x_arr - y_arr
    nonzero = diff[diff != 0]
    if len(nonzero) < 6:
        return float("nan")
    _, p = wilcoxon(nonzero)
    return float(p)


def bh_correction(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values in the original order.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [0.0] * n
    prev = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        adj = min(prev, p * n / (rank_minus_1 + 1))
        adjusted[orig_idx] = adj
        prev = adj
    return adjusted


# ---------------------------------------------------------------------------
# Extended metrics: CIs, tests, breakdowns
# ---------------------------------------------------------------------------

def _collect_item_uai_vectors(
    records: list[dict],
    epsilon: float = EPSILON,
    control_key: str = "control",
) -> dict[str, dict[tuple[str, str], float]]:
    """Per-item UAI by condition group, keyed by (item_id, anchor direction).

    The key carries the direction because an item contributes one UAI per
    direction -- irrelevant_low and irrelevant_high both land in "irr". Keying
    on item_id alone would collide, and pairing plausible against irrelevant
    then has to match on both: same item *and* same anchor value, so that only
    the framing sentence differs. That is what Disc_delta is defined to
    isolate.
    """
    items = group_by_item(records)
    vectors: dict[str, dict[tuple[str, str], float]] = {
        "irr": {}, "plaus": {}, "placebo": {}, "authority": {}, "neutral": {},
    }

    for iid, conds in items.items():
        ctrl = conds.get(control_key)
        if ctrl is None:
            continue
        y_ctrl = ctrl.get("answer_int")
        if y_ctrl is None:
            continue

        for cond, rec in conds.items():
            if cond.startswith("control"):
                continue
            y_anchor = rec.get("answer_int")
            a = rec.get("anchor_value")
            if y_anchor is None or a is None:
                continue
            denom = a - y_ctrl
            if abs(denom) < epsilon:
                continue
            uai = (y_anchor - y_ctrl) / denom

            key = (iid, cond.rsplit("_", 1)[-1])
            if "placebo" in cond:
                vectors["placebo"][key] = uai
            elif "authority" in cond:
                vectors["authority"][key] = uai
            elif "neutral" in cond:
                vectors["neutral"][key] = uai
            elif "irrelevant" in cond:
                vectors["irr"][key] = uai
            elif "plausible" in cond:
                vectors["plaus"][key] = uai

    return vectors


def _parse_rate_by_condition(records: list[dict]) -> dict[str, dict[str, int]]:
    """Per-condition n_parsed / n_total."""
    by_cond: dict[str, dict[str, int]] = {}
    for r in records:
        c = r.get("condition", "unknown")
        if c not in by_cond:
            by_cond[c] = {"n_total": 0, "n_parsed": 0}
        by_cond[c]["n_total"] += 1
        if r.get("parsed_ok") and r.get("answer_int") is not None:
            by_cond[c]["n_parsed"] += 1
    return by_cond


def compute_extended_metrics(
    records: list[dict],
    epsilon: float = EPSILON,
    baseline_condition: str = "control",
) -> dict:
    """Compute standard metrics plus bootstrap CIs, paired tests, and
    per-condition parse denominators.

    Superset of compute_unified_metrics output.
    """
    base = compute_unified_metrics(
        records, epsilon=epsilon, baseline_condition=baseline_condition,
    )

    items = group_by_item(records)
    vectors = _collect_item_uai_vectors(
        records, epsilon, control_key=baseline_condition,
    )

    ci_results = {}
    for key in ("irr", "plaus", "placebo", "authority", "neutral"):
        if vectors[key]:
            mean, lo, hi = bootstrap_ci(list(vectors[key].values()))
            ci_results[f"uai_{key}_ci"] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
            ci_results[f"n_uai_{key}"] = len(vectors[key])
        else:
            ci_results[f"uai_{key}_ci"] = None
            ci_results[f"n_uai_{key}"] = 0

    if base.get("mae_control") is not None:
        mae_vals = []
        for _iid, conds in items.items():
            ctrl = conds.get(baseline_condition)
            if ctrl and ctrl.get("answer_int") is not None and ctrl.get("y_star_evidence") is not None:
                mae_vals.append(abs(ctrl["answer_int"] - ctrl["y_star_evidence"]))
        if mae_vals:
            mean, lo, hi = bootstrap_ci(mae_vals)
            ci_results["mae_control_ci"] = {"mean": round(mean, 2), "lo": round(lo, 2), "hi": round(hi, 2)}

    # Pair on (item_id, direction). This used to slice both lists to the
    # shorter length in dict-iteration order, which is not a pairing at all:
    # the epsilon exclusion drops different items from each vector, so the
    # "paired" rows were unrelated. Neither output below is read by any paper
    # artifact, so this corrects the computation without changing a published
    # number.
    paired_keys = sorted(vectors["plaus"].keys() & vectors["irr"].keys())
    if base.get("disc_delta") is not None and paired_keys:
        disc_vals = [vectors["plaus"][k] - vectors["irr"][k] for k in paired_keys]
        mean, lo, hi = bootstrap_ci(disc_vals)
        ci_results["disc_delta_ci"] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}

    p_values = {}
    test_labels = []

    if paired_keys:
        p = paired_wilcoxon([vectors["plaus"][k] for k in paired_keys],
                            [vectors["irr"][k] for k in paired_keys])
        p_values["p_plaus_vs_irr"] = round(p, 6) if not np.isnan(p) else None
        test_labels.append("p_plaus_vs_irr")

    for key in ("irr", "plaus", "placebo", "authority", "neutral"):
        if vectors[key]:
            vals = list(vectors[key].values())
            p = paired_wilcoxon(vals, [0.0] * len(vals))
            p_values[f"p_{key}_vs_zero"] = round(p, 6) if not np.isnan(p) else None
            test_labels.append(f"p_{key}_vs_zero")

    raw_ps = [p_values.get(k) for k in test_labels]
    valid_ps = [p for p in raw_ps if p is not None and not np.isnan(p)]
    if valid_ps:
        adjusted = bh_correction(valid_ps)
        adj_idx = 0
        for k in test_labels:
            if p_values.get(k) is not None and not np.isnan(p_values[k]):
                p_values[f"{k}_bh"] = round(adjusted[adj_idx], 6)
                adj_idx += 1

    parse_denom = _parse_rate_by_condition(records)

    has_twostage_ctrl = any(r.get("condition") == "control_twostage" for r in records)
    twostage_metrics = {}
    if has_twostage_ctrl and baseline_condition != "control_twostage":
        ts_vectors = _collect_item_uai_vectors(
            records, epsilon, control_key="control_twostage",
        )
        for key in ("irr", "plaus"):
            if ts_vectors[key]:
                mean, lo, hi = bootstrap_ci(ts_vectors[key])
                twostage_metrics[f"uai_{key}_ts"] = round(mean, 4)
                twostage_metrics[f"uai_{key}_ts_ci"] = {
                    "mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                }

    result = {**base, **ci_results, **p_values, **twostage_metrics}
    result["parse_by_condition"] = {
        c: {"n_parsed": v["n_parsed"], "n_total": v["n_total"],
            "rate": round(v["n_parsed"] / v["n_total"], 4) if v["n_total"] else 0.0}
        for c, v in parse_denom.items()
    }
    return result


def _parse_offset_from_item_id(item_id: str) -> int | None:
    """Extract offset from item_id like 'EXT-pricing_wtp-e-off15-001'."""
    import re
    m = re.search(r"-off(\d+)-", item_id or "")
    return int(m.group(1)) if m else None


def compute_by_offset(
    records: list[dict],
    epsilon: float = EPSILON,
    baseline_condition: str = "control",
) -> dict[int, dict]:
    """Compute metrics grouped by anchor offset distance {15, 25, 40}."""
    by_offset: dict[int, list[dict]] = {}
    for r in records:
        offset = None
        anchors = r.get("anchors")
        if isinstance(anchors, dict):
            offset = anchors.get("offset")
        if offset is None:
            offset = r.get("anchor_offset")
        if offset is None:
            offset = _parse_offset_from_item_id(r.get("item_id", ""))
        if offset is not None:
            by_offset.setdefault(int(offset), []).append(r)

    return {
        offset: compute_unified_metrics(
            recs, epsilon=epsilon, baseline_condition=baseline_condition,
        )
        for offset, recs in sorted(by_offset.items())
    }


def compute_by_difficulty(
    records: list[dict],
    epsilon: float = EPSILON,
    baseline_condition: str = "control",
) -> dict[str, dict]:
    """Compute metrics grouped by item difficulty (easy/hard)."""
    by_diff: dict[str, list[dict]] = {}
    for r in records:
        diff = r.get("difficulty", "unknown")
        by_diff.setdefault(diff, []).append(r)

    return {
        diff: compute_unified_metrics(
            recs, epsilon=epsilon, baseline_condition=baseline_condition,
        )
        for diff, recs in sorted(by_diff.items())
    }
