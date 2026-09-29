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

from anchorbench.data.suites._shared import CONDITIONS as _CONDITION_TABLE

# The five core conditions, in the renderers' order: control, then
# irrelevant and plausible, each low and high. The table itself lives with
# the renderers in data/suites/_shared.py.
CONDITIONS: list[str] = [name for name, _, _ in _CONDITION_TABLE]

EPSILON = 3.0


def baseline_condition(records: list[dict], prefer: str = "control") -> str:
    """The control condition to score against: ``prefer`` when the records
    carry it, else the other control condition (``control`` /
    ``control_twostage``) when they carry that one, else ``prefer``.

    History runs before the matched-format re-run carry only ``control``;
    the re-run carries ``control_twostage`` as well (ledger D8).
    """
    present = {r["condition"] for r in records}
    if prefer in present:
        return prefer
    other = "control_twostage" if prefer == "control" else "control"
    return other if other in present else prefer


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


def item_uai(y_anchor: float, y_ctrl: float, anchor: float, epsilon: float = EPSILON) -> float | None:
    """UAI for one item, Sec. 3.4: the fraction of the control-to-anchor gap
    the anchored answer closes, (y_anchor - y_ctrl) / (a - y_ctrl).

    None when |a - y_ctrl| < epsilon: the ratio is unstable there and the
    paper excludes the item rather than smoothing the denominator.
    """
    denom = anchor - y_ctrl
    if abs(denom) < epsilon:
        return None
    return (y_anchor - y_ctrl) / denom


def toward_anchor(y_anchor: float, y_ctrl: float, anchor: float) -> int:
    """TAR indicator, Sec. 3.4: 1 when the shift has the sign of the gap.
    Defined for every item, including those UAI excludes."""
    return 1 if (y_anchor - y_ctrl) * (anchor - y_ctrl) > 0 else 0


def control_errors(items: dict[str, dict[str, dict]], baseline: str) -> list[float]:
    """|y_ctrl - y*| per item with a parsed baseline answer and a gold value."""
    errors = []
    for conds in items.values():
        ctrl = conds.get(baseline)
        if ctrl is None:
            continue
        y, y_star = ctrl.get("answer_int"), ctrl.get("y_star_evidence")
        if y is None or y_star is None:
            continue
        errors.append(abs(y - y_star))
    return errors


def control_accuracy(items: dict[str, dict[str, dict]], baseline: str) -> tuple[float | None, float | None]:
    """(MAE_c, Acc_10) on the control answers, Eq. (accuracy): mean absolute
    error and the share within 10 points of gold; None without any."""
    errors = control_errors(items, baseline)
    if not errors:
        return None, None
    return float(np.mean(errors)), sum(1 for e in errors if e <= 10) / len(errors)


def anchor_effects(
    items: dict[str, dict[str, dict]], baseline: str, epsilon: float,
) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    """Per-condition lists of item UAI (after the epsilon exclusion) and of
    TAR indicators (every item), for the four core anchored conditions."""
    uai: dict[str, list[float]] = {c: [] for c in CONDITIONS if c != "control"}
    tar: dict[str, list[int]] = {c: [] for c in CONDITIONS if c != "control"}
    for conds in items.values():
        ctrl = conds.get(baseline)
        if ctrl is None or ctrl.get("answer_int") is None:
            continue
        y_ctrl = ctrl["answer_int"]
        for cond in CONDITIONS:
            if cond == baseline:
                continue
            rec = conds.get(cond)
            if rec is None or rec.get("answer_int") is None or rec.get("anchor_value") is None:
                continue
            y_anchor, a = rec["answer_int"], rec["anchor_value"]
            tar[cond].append(toward_anchor(y_anchor, y_ctrl, a))
            u = item_uai(y_anchor, y_ctrl, a, epsilon)
            if u is not None:
                uai[cond].append(u)
    return uai, tar


def history_adjustment(items: dict[str, dict[str, dict]]) -> tuple[list[float], list[float]]:
    """History only: ACR = (s2 - s1) / (y* - s1) and RR = 1 - |s2 - y*| / |s1 - y*|
    over the plausible conditions, per item with a Stage-1 answer at least one
    point from gold."""
    acr: list[float] = []
    rr: list[float] = []
    for conds in items.values():
        for cond in ("plausible_low", "plausible_high"):
            rec = conds.get(cond)
            if rec is None:
                continue
            s1, s2, y_star = rec.get("stage1_answer"), rec.get("answer_int"), rec.get("y_star_evidence")
            if s1 is None or s2 is None or y_star is None:
                continue
            if abs(y_star - s1) >= 1:
                acr.append((s2 - s1) / (y_star - s1))
            if abs(s1 - y_star) >= 1:
                rr.append(1.0 - abs(s2 - y_star) / abs(s1 - y_star))
    return acr, rr


def _mean(values: list) -> float | None:
    return float(np.mean(values)) if values else None


def _r(v: float | None, d: int = 4) -> float | None:
    return round(v, d) if v is not None else None


def compute_unified_metrics(
    records: list[dict],
    epsilon: float = EPSILON,
    baseline_condition: str = "control",
) -> dict:
    """The per-cell metrics of Sec. 3.4 from one results.jsonl: parse rate,
    MAE_c and Acc_10, UAI and TAR per condition and pooled per relevance,
    Disc_delta, and for History ACR and RR.

    ``baseline_condition`` is the control answer UAI and MAE are measured
    against (``control_twostage`` for the matched-format History run).
    """
    n_total = len(records)
    n_parsed = sum(1 for r in records if r.get("parsed_ok") and r.get("answer_int") is not None)
    items = group_by_item(records)

    mae_control, acc10_control = control_accuracy(items, baseline_condition)
    uai, tar = anchor_effects(items, baseline_condition, epsilon)
    # Low and high are pooled within a relevance level: one UAI_irr and one
    # UAI_pls per cell (Sec. 3.4, "Reporting pipeline").
    all_irr = uai["irrelevant_low"] + uai["irrelevant_high"]
    all_plaus = uai["plausible_low"] + uai["plausible_high"]
    uai_irr, uai_plaus = _mean(all_irr), _mean(all_plaus)

    result = {
        "baseline_condition": baseline_condition,
        "n_items": len(items),
        "n_records": n_total,
        "parse_rate": round(n_parsed / n_total if n_total else 0.0, 4),
        "mae_control": _r(mae_control, 2),
        "acc10_control": _r(acc10_control),
        "uai_irr_low": _r(_mean(uai["irrelevant_low"])),
        "uai_irr_high": _r(_mean(uai["irrelevant_high"])),
        "uai_plaus_low": _r(_mean(uai["plausible_low"])),
        "uai_plaus_high": _r(_mean(uai["plausible_high"])),
        "uai_irr": _r(uai_irr),
        "uai_plaus": _r(uai_plaus),
        "tar_irr_low": _r(_mean(tar["irrelevant_low"])),
        "tar_irr_high": _r(_mean(tar["irrelevant_high"])),
        "tar_plaus_low": _r(_mean(tar["plausible_low"])),
        "tar_plaus_high": _r(_mean(tar["plausible_high"])),
        "tar_irr": _r(_mean(tar["irrelevant_low"] + tar["irrelevant_high"])),
        "tar_plaus": _r(_mean(tar["plausible_low"] + tar["plausible_high"])),
        "disc_delta": _r(uai_plaus - uai_irr) if uai_irr is not None and uai_plaus is not None else None,
        "epsilon": epsilon,
        "n_uai_irr": len(all_irr),
        "n_uai_plaus": len(all_plaus),
    }

    if any(r.get("stage1_answer") is not None for r in records):
        acr, rr = history_adjustment(items)
        result["acr_mean"] = _r(_mean(acr))
        result["rr_mean"] = _r(_mean(rr))
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
            uai = item_uai(y_anchor, y_ctrl, a, epsilon)
            if uai is None:
                continue

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


def _ci_summary(values: list[float], digits: int = 4) -> dict[str, float]:
    mean, lo, hi = bootstrap_ci(values)
    return {"mean": round(mean, digits), "lo": round(lo, digits), "hi": round(hi, digits)}


def _paired_tests(
    vectors: dict[str, dict[tuple[str, str], float]], paired_keys: list[tuple[str, str]],
) -> dict[str, float | None]:
    """Two-sided Wilcoxon signed-rank p-values: plausible vs irrelevant on the
    paired items, and each relevance group against zero; then
    Benjamini-Hochberg over the tests that ran (Appendix app:stats)."""
    p_values: dict[str, float | None] = {}
    labels: list[str] = []

    def record(label: str, p: float) -> None:
        p_values[label] = round(p, 6) if not np.isnan(p) else None
        labels.append(label)

    if paired_keys:
        record("p_plaus_vs_irr", paired_wilcoxon([vectors["plaus"][k] for k in paired_keys],
                                                 [vectors["irr"][k] for k in paired_keys]))
    for key in ("irr", "plaus", "placebo", "authority", "neutral"):
        if vectors[key]:
            vals = list(vectors[key].values())
            record(f"p_{key}_vs_zero", paired_wilcoxon(vals, [0.0] * len(vals)))

    valid = [k for k in labels if p_values[k] is not None]
    for k, adj in zip(valid, bh_correction([p_values[k] for k in valid])):
        p_values[f"{k}_bh"] = round(adj, 6)
    return p_values


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
            ci_results[f"uai_{key}_ci"] = _ci_summary(list(vectors[key].values()))
            ci_results[f"n_uai_{key}"] = len(vectors[key])
        else:
            ci_results[f"uai_{key}_ci"] = None
            ci_results[f"n_uai_{key}"] = 0

    if base.get("mae_control") is not None:
        mae_vals = control_errors(items, baseline_condition)
        if mae_vals:
            ci_results["mae_control_ci"] = _ci_summary(mae_vals, digits=2)

    # Pair on (item_id, direction). This used to slice both lists to the
    # shorter length in dict-iteration order, which is not a pairing at all:
    # the epsilon exclusion drops different items from each vector, so the
    # "paired" rows were unrelated. Neither output below is read by any paper
    # artifact, so this corrects the computation without changing a published
    # number.
    paired_keys = sorted(vectors["plaus"].keys() & vectors["irr"].keys())
    if base.get("disc_delta") is not None and paired_keys:
        ci_results["disc_delta_ci"] = _ci_summary(
            [vectors["plaus"][k] - vectors["irr"][k] for k in paired_keys])

    p_values = _paired_tests(vectors, paired_keys)
    parse_denom = _parse_rate_by_condition(records)

    has_twostage_ctrl = any(r.get("condition") == "control_twostage" for r in records)
    twostage_metrics = {}
    if has_twostage_ctrl and baseline_condition != "control_twostage":
        ts_vectors = _collect_item_uai_vectors(
            records, epsilon, control_key="control_twostage",
        )
        for key in ("irr", "plaus"):
            if ts_vectors[key]:
                ci = _ci_summary(list(ts_vectors[key].values()))
                twostage_metrics[f"uai_{key}_ts"] = ci["mean"]
                twostage_metrics[f"uai_{key}_ts_ci"] = ci

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
