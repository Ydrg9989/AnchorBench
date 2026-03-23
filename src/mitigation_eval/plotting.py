"""Paper-ready figures for mitigation evaluation results."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

MITIGATION_COLORS = {
    "B0": "#333333", "B1": "#1f77b4", "B2": "#ff7f0e", "B3": "#2ca02c",
    "SELFHELP": "#d62728", "COUNTERFACTUAL_ENSEMBLE": "#9467bd",
    "LATE_BINDING": "#8c564b",
}


def plot_rr_vs_utility(csv_path: Path, out_path: Path) -> None:
    """Scatter: anchoring reduction rate vs MAE."""
    rows = _load_csv(csv_path)
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        rr = r.get("rr")
        mae = r.get("mae")
        if rr in (None, "", "None") or mae in (None, "", "None"):
            continue
        rr, mae = float(rr), float(mae)
        mit = r.get("mitigation", "?")
        color = MITIGATION_COLORS.get(mit, "#888888")
        ax.scatter(rr, mae, color=color, s=40, alpha=0.7, edgecolors="white", linewidth=0.5)

    handles = []
    for mit, color in MITIGATION_COLORS.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=color, label=mit, markersize=8))
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_xlabel("Reduction Rate (RR)")
    ax.set_ylabel("MAE vs gold")
    ax.set_title("Anchoring reduction vs utility tradeoff")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved RR vs utility -> %s", out_path)


def plot_nai_heatmap(csv_path: Path, out_path: Path) -> None:
    """Heatmap: rows = mitigations, columns = suites, cells = NAI."""
    rows = _load_csv(csv_path)
    if not rows:
        return

    suites = sorted(set(r["suite"] for r in rows))
    mits = sorted(set(r["mitigation"] for r in rows))

    data: dict[tuple, list[float]] = {}
    for r in rows:
        nai = r.get("nai_mean")
        if nai in (None, "", "None"):
            continue
        key = (r["mitigation"], r["suite"])
        data.setdefault(key, []).append(float(nai))

    matrix = np.full((len(mits), len(suites)), np.nan)
    for i, mit in enumerate(mits):
        for j, suite in enumerate(suites):
            vals = data.get((mit, suite), [])
            if vals:
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=-0.1, vmax=0.5)
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels([s.capitalize() for s in suites], fontsize=8)
    ax.set_yticks(range(len(mits)))
    ax.set_yticklabels(mits, fontsize=8)

    for i in range(len(mits)):
        for j in range(len(suites)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(matrix[i, j]) < 0.3 else "white")

    plt.colorbar(im, ax=ax, label="NAI")
    ax.set_title("NAI by mitigation and suite")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved NAI heatmap -> %s", out_path)


def _load_csv(path: Path) -> list[dict]:
    """Read a CSV file and return rows as list of dicts."""
    if not path.exists():
        log.warning("CSV not found: %s", path)
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
