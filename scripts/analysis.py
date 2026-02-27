"""Response parsing, metric computation, visualization, and CSV output."""

import csv
import json
import math
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

X1_TO_ANCHOR_PCT = {
    1: 14, 2: 22, 3: 28, 4: 38, 5: 47, 6: 58, 7: 74, 8: 82, 9: 86,
}

MODEL_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_regime_a(response):
    """Parse model response to an option index (1-11), or None on failure."""
    m = re.search(r"Option\s*(\d{1,2})", response, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= 11:
            return idx
    m = re.search(r"(\d{1,3})\s*%", response)
    if m:
        pct = int(m.group(1))
        return max(1, min(11, round(pct / 10) + 1))
    return None


def parse_regime_b(response, expected_n):
    """Parse model response to a list of {decision, confidence} dicts."""
    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list) and len(data) >= expected_n:
                out = []
                for item in data[:expected_n]:
                    out.append({
                        "decision": str(item.get("decision", "")).lower(),
                        "confidence": int(item.get("confidence", 50)),
                    })
                return out
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    decisions = re.findall(
        r'"decision"\s*:\s*"(admit|reject)"', response, re.IGNORECASE,
    )
    confidences = re.findall(r'"confidence"\s*:\s*(\d+)', response)

    if len(decisions) >= expected_n:
        out = []
        for i in range(expected_n):
            conf = int(confidences[i]) if i < len(confidences) else 50
            out.append({"decision": decisions[i].lower(), "confidence": conf})
        return out
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_regime_a_metrics(results):
    """Compute per-row shifts and aggregated stats."""
    valid = [
        r for r in results
        if r["control_idx"] is not None and r["treatment_idx"] is not None
    ]
    for r in valid:
        r["signed_shift"] = r["treatment_idx"] - r["control_idx"]
        r["abs_shift"] = abs(r["signed_shift"])

    if not valid:
        return valid, {"n_valid": 0, "n_total": len(results)}

    by_x1 = defaultdict(list)
    for r in valid:
        by_x1[r["x_1"]].append(r["signed_shift"])

    per_bin = {}
    for x1 in sorted(by_x1):
        s = by_x1[x1]
        per_bin[x1] = {
            "mean_signed_shift": float(np.mean(s)),
            "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
            "n": len(s),
        }

    anchor_pcts = [X1_TO_ANCHOR_PCT.get(r["x_1"], r["x_1"] * 10) for r in valid]
    signed_shifts = [r["signed_shift"] for r in valid]
    corr = (
        float(np.corrcoef(anchor_pcts, signed_shifts)[0, 1])
        if len(set(anchor_pcts)) > 1 else float("nan")
    )

    agg = {
        "mean_abs_shift": float(np.mean([r["abs_shift"] for r in valid])),
        "mean_signed_shift": float(np.mean(signed_shifts)),
        "pearson_r": corr,
        "parse_error_rate": 1 - len(valid) / len(results),
        "n_valid": len(valid),
        "n_total": len(results),
        "per_bin": per_bin,
    }
    return valid, agg


def compute_regime_b_metrics(results):
    """Compute flip-rate, confidence variance, and position effects."""
    all_flip_rates = []
    all_conf_vars = []
    scenario_stats = {}
    position_admits = defaultdict(list)
    position_confs = defaultdict(list)

    for sid, student_data in results.items():
        flip_rates = []
        conf_vars = []

        for _student_idx, perm_results in student_data.items():
            decisions = [r["decision"] for r in perm_results]
            confidences = [r["confidence"] for r in perm_results]
            n = len(decisions)
            if n == 0:
                continue
            n_admit = sum(1 for d in decisions if d == "admit")
            flip_rate = 1 - max(n_admit, n - n_admit) / n
            flip_rates.append(flip_rate)
            all_flip_rates.append(flip_rate)

            if len(confidences) > 1:
                v = float(np.var(confidences, ddof=1))
                conf_vars.append(v)
                all_conf_vars.append(v)

            for r in perm_results:
                position_admits[r["position"]].append(
                    1 if r["decision"] == "admit" else 0,
                )
                position_confs[r["position"]].append(r["confidence"])

        scenario_stats[sid] = {
            "mean_flip_rate": float(np.mean(flip_rates)) if flip_rates else 0,
            "flip_rates": flip_rates,
            "mean_conf_var": float(np.mean(conf_vars)) if conf_vars else 0,
        }

    position_effect = {}
    for pos in sorted(position_admits):
        position_effect[pos] = {
            "mean_admit_rate": float(np.mean(position_admits[pos])),
            "mean_confidence": float(np.mean(position_confs[pos])),
            "n": len(position_admits[pos]),
        }

    return {
        "overall_flip_rate": float(np.mean(all_flip_rates)) if all_flip_rates else 0,
        "overall_conf_var": float(np.mean(all_conf_vars)) if all_conf_vars else 0,
        "scenario_stats": scenario_stats,
        "position_effect": position_effect,
        "n_students_total": len(all_flip_rates),
    }


# ---------------------------------------------------------------------------
# Per-model visualization
# ---------------------------------------------------------------------------

def plot_regime_a(valid_results, agg, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    per_bin = agg["per_bin"]
    x1_bins = sorted(per_bin)
    x_labels = [f"{X1_TO_ANCHOR_PCT.get(x, x * 10)}%" for x in x1_bins]
    means = [per_bin[x]["mean_signed_shift"] for x in x1_bins]
    stds = [per_bin[x]["std"] for x in x1_bins]
    ns = [per_bin[x]["n"] for x in x1_bins]
    ci95 = [1.96 * s / math.sqrt(n) if n > 1 else 0 for s, n in zip(stds, ns)]

    ax1.bar(
        range(len(x1_bins)), means, yerr=ci95, capsize=4,
        color="#4C72B0", edgecolor="black", alpha=0.85,
    )
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xticks(range(len(x1_bins)))
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("Anchor Value")
    ax1.set_ylabel("Mean Signed Shift (option units)")
    ax1.set_title("Regime A: Mean Shift by Anchor Position")

    anchor_pcts = [X1_TO_ANCHOR_PCT.get(r["x_1"], r["x_1"] * 10) for r in valid_results]
    signed_shifts = [r["signed_shift"] for r in valid_results]
    ax2.scatter(anchor_pcts, signed_shifts, alpha=0.3, color="#DD8452",
                edgecolors="black", s=20, linewidths=0.3)

    if len(set(anchor_pcts)) > 1:
        z = np.polyfit(anchor_pcts, signed_shifts, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(anchor_pcts), max(anchor_pcts), 100)
        ax2.plot(x_line, p(x_line), "r--", linewidth=2)
        ax2.annotate(
            f"r = {agg['pearson_r']:.3f}", xy=(0.05, 0.95),
            xycoords="axes fraction", fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Anchor Value (%)")
    ax2.set_ylabel("Signed Shift (option units)")
    ax2.set_title("Regime A: Shift vs Anchor Value")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regime_b(agg, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scenario_stats = agg["scenario_stats"]
    sids = sorted(scenario_stats)
    flip_data = [scenario_stats[s]["flip_rates"] for s in sids]

    if len(sids) <= 20:
        tick_labels = [str(s) for s in sids]
    else:
        tick_labels = ["" for _ in sids]

    bp = ax1.boxplot(flip_data, tick_labels=tick_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#55A868")
        patch.set_alpha(0.7)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Scenario ID")
    ax1.set_ylabel("Flip Rate")
    ax1.set_title("Regime B: Decision Inconsistency by Scenario")
    if len(sids) > 20:
        ax1.tick_params(axis="x", labelsize=6, rotation=90)

    pos_effect = agg["position_effect"]
    positions = sorted(pos_effect)
    if positions:
        admit_rates = [pos_effect[p]["mean_admit_rate"] for p in positions]
        ax2.bar(positions, admit_rates, color="#4C72B0", edgecolor="black", alpha=0.85)
        ax2.set_xlabel("Position in Permutation")
        ax2.set_ylabel("Admit Rate")
        ax2.set_title("Regime B: Admit Rate by Presentation Position")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-model comparison visualization
# ---------------------------------------------------------------------------

def plot_comparison_regime_a(model_aggs, output_path):
    """Grouped bar chart: mean signed shift per x_1 bin, one colour per model."""
    models = list(model_aggs.keys())
    all_x1 = sorted({
        x1 for agg in model_aggs.values()
        for x1 in agg.get("per_bin", {})
    })
    if not all_x1:
        return
    x_labels = [f"{X1_TO_ANCHOR_PCT.get(x, x * 10)}%" for x in all_x1]
    n_models = len(models)
    bar_width = 0.8 / n_models
    x_pos = np.arange(len(all_x1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, model in enumerate(models):
        per_bin = model_aggs[model].get("per_bin", {})
        means = [per_bin.get(x1, {}).get("mean_signed_shift", 0) for x1 in all_x1]
        stds = [per_bin.get(x1, {}).get("std", 0) for x1 in all_x1]
        ns = [per_bin.get(x1, {}).get("n", 1) for x1 in all_x1]
        ci = [1.96 * s / math.sqrt(n) if n > 1 else 0 for s, n in zip(stds, ns)]
        ax1.bar(
            x_pos + i * bar_width, means, bar_width, yerr=ci, capsize=2,
            color=MODEL_COLORS[i % len(MODEL_COLORS)], label=model, alpha=0.85,
        )

    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xticks(x_pos + bar_width * (n_models - 1) / 2)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("Anchor Value")
    ax1.set_ylabel("Mean Signed Shift")
    ax1.set_title("Regime A: Shift by Anchor (all models)")
    ax1.legend(fontsize=8)

    metrics = ["mean_abs_shift", "mean_signed_shift", "pearson_r"]
    metric_labels = ["Mean |Shift|", "Mean Shift", "Pearson r"]
    x_m = np.arange(len(metrics))
    for i, model in enumerate(models):
        vals = [model_aggs[model].get(m, 0) for m in metrics]
        ax2.bar(
            x_m + i * bar_width, vals, bar_width,
            color=MODEL_COLORS[i % len(MODEL_COLORS)], label=model, alpha=0.85,
        )
    ax2.set_xticks(x_m + bar_width * (n_models - 1) / 2)
    ax2.set_xticklabels(metric_labels)
    ax2.set_title("Regime A: Summary Metrics")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_regime_b(model_aggs, output_path):
    """Side-by-side comparison of Regime B metrics across models."""
    models = list(model_aggs.keys())
    n_models = len(models)
    bar_width = 0.8 / n_models

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    flip_rates = [model_aggs[m].get("overall_flip_rate", 0) for m in models]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(n_models)]
    ax1.bar(range(n_models), flip_rates, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xticks(range(n_models))
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Overall Flip Rate")
    ax1.set_title("Regime B: Order Inconsistency by Model")

    all_positions = sorted({
        p for agg in model_aggs.values()
        for p in agg.get("position_effect", {})
    })
    if all_positions:
        x_pos = np.arange(len(all_positions))
        for i, model in enumerate(models):
            pe = model_aggs[model].get("position_effect", {})
            rates = [pe.get(p, {}).get("mean_admit_rate", 0) for p in all_positions]
            ax2.bar(
                x_pos + i * bar_width, rates, bar_width,
                color=MODEL_COLORS[i % len(MODEL_COLORS)], label=model, alpha=0.85,
            )
        ax2.set_xticks(x_pos + bar_width * (n_models - 1) / 2)
        ax2.set_xticklabels([str(p) for p in all_positions])
        ax2.set_xlabel("Position in Permutation")
        ax2.set_ylabel("Admit Rate")
        ax2.set_title("Regime B: Position Effect by Model")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_regime_a_csv(results, path):
    fieldnames = [
        "idx", "scenario", "x_1", "anchor_pct",
        "control_idx", "treatment_idx", "signed_shift", "abs_shift",
        "control_raw", "treatment_raw",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            r["anchor_pct"] = X1_TO_ANCHOR_PCT.get(r["x_1"], r["x_1"] * 10)
            w.writerow(r)


def save_regime_b_csv(results, path):
    fieldnames = [
        "scenario_id", "student_idx", "perm_idx", "position",
        "decision", "confidence",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sid, student_data in sorted(results.items()):
            for student_idx, perm_results in sorted(student_data.items()):
                for r in perm_results:
                    w.writerow({
                        "scenario_id": sid,
                        "student_idx": student_idx,
                        **r,
                    })
