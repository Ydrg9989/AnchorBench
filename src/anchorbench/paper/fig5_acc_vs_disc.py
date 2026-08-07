#!/usr/bin/env python3
"""Figure 5: scatter of task accuracy vs anchoring discrimination.

Reads ``results/{full_benchmark,api_benchmark}/unified_all_suites.json``
and plots one point per (model, suite) cell with circles for open-weight
models and diamonds for API models, colour-coded by suite.

Reports the Pearson correlation r(Acc10, Disc_delta) with a
nonparametric bootstrap 95% CI in the figure caption text printed to
stdout.

Outputs:
    outputs/figures/fig5_acc_vs_disc.pdf
    outputs/figures/fig5_acc_vs_disc.png

Usage:
    python anchorbench.paper.fig5_acc_vs_disc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Embed real TrueType outlines instead of matplotlib's default Type 3
# fonts, which are not embedded as a FontFile and which arXiv and most
# venues flag. Affects the PDF only; the golden PNGs are unchanged.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
import numpy as np

from ._common import (
    API_MODELS_ORDER,
    DEFAULT_API_RESULTS,
    DEFAULT_FIG_DIR,
    DEFAULT_OW_RESULTS,
    SUITES,
    rel_to_root,
)

SUITE_COLORS = {
    "External": "#1f77b4",
    "History":  "#d62728",
    "Icl":      "#7f7f7f",
    "Rag":      "#2ca02c",
    "Tool":     "#9467bd",
}

SUITE_LABELS = {
    "External": "External",
    "History":  "History",
    "Icl":      "ICL",
    "Rag":      "RAG",
    "Tool":     "Tool",
}


def load_unified(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def collect_points(unified: list[dict], api_set: set[str]) -> list[dict]:
    pts = []
    for r in unified:
        acc = r.get("acc10_control")
        disc = r.get("disc_delta")
        if acc is None or disc is None:
            continue
        pts.append({
            "model": r["model"],
            "suite": r["suite"],
            "acc": acc,
            "disc": disc,
            "is_api": r["model"] in api_set,
        })
    return pts


def pearson_with_bootstrap_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        rs[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    lo, hi = float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))
    return r, lo, hi


def plot(points: list[dict], out_pdf: Path, r: float, lo: float, hi: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    seen = set()
    for suite in SUITES:
        for tier_label, marker, size in (("Open-weight", "o", 55),
                                         ("API", "D", 65)):
            is_api = tier_label == "API"
            xs = [p["acc"] * 100 for p in points
                  if p["suite"] == suite and p["is_api"] == is_api]
            ys = [p["disc"] for p in points
                  if p["suite"] == suite and p["is_api"] == is_api]
            if not xs:
                continue
            label = SUITE_LABELS[suite] if suite not in seen else None
            seen.add(suite)
            ax.scatter(xs, ys, s=size, marker=marker,
                       facecolors=SUITE_COLORS[suite],
                       edgecolors="white", linewidths=0.6, alpha=0.85,
                       label=label)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"Acc$_{10}$ (control, \%)")
    ax.set_ylabel(r"Disc$_{\Delta}$")
    ax.grid(True, alpha=0.3)

    suite_handles = []
    for suite in SUITES:
        suite_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                        markerfacecolor=SUITE_COLORS[suite],
                                        markersize=8, label=SUITE_LABELS[suite]))
    tier_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="gray", markersize=8, label="Open-weight"),
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="gray", markersize=8, label="API"),
    ]
    leg1 = ax.legend(handles=suite_handles, loc="upper left",
                     fontsize=9, frameon=False, title="Suite")
    ax.add_artist(leg1)
    ax.legend(handles=tier_handles, loc="lower left",
              fontsize=9, frameon=False, title="Tier")

    sign_lo = "$-$" if lo < 0 else ""
    sign_hi = "$-$" if hi < 0 else ""
    sign_r = "$-$" if r < 0 else ""
    txt = (rf"$r = {sign_r}{abs(r):.2f}$  "
           rf"95\% CI [{sign_lo}{abs(lo):.2f}, {sign_hi}{abs(hi):.2f}]")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"  wrote {rel_to_root(out_pdf)}")
    print(f"  wrote {rel_to_root(out_pdf.with_suffix('.png'))}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ow_unified", type=Path,
                   default=DEFAULT_OW_RESULTS / "unified_all_suites.json")
    p.add_argument("--api_unified", type=Path,
                   default=DEFAULT_API_RESULTS / "unified_all_suites.json")
    p.add_argument("--out_pdf", type=Path,
                   default=DEFAULT_FIG_DIR / "fig5_acc_vs_disc.pdf")
    args = p.parse_args()

    api_set = set(API_MODELS_ORDER)

    pts = []
    pts.extend(collect_points(load_unified(args.ow_unified), api_set))
    pts.extend(collect_points(load_unified(args.api_unified), api_set))

    print(f"Collected {len(pts)} (model, suite) cells with computable Disc_delta")

    x = np.array([p["acc"] for p in pts])
    y = np.array([p["disc"] for p in pts])
    r, lo, hi = pearson_with_bootstrap_ci(x, y)
    print(f"Pearson r(Acc10, Disc_delta) = {r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    plot(pts, args.out_pdf, r, lo, hi)


if __name__ == "__main__":
    main()
