#!/usr/bin/env python3
"""Figure 4: dose-response of UAI_pls vs anchor offset.

Reads per-record ``results.jsonl`` files for the External, RAG, and ICL
suites under ``results/full_benchmark/`` (open-weight) and
``results/api_benchmark/`` (API), computes per-offset UAI_plaus per
model via :func:`compute_by_offset`, then plots suite-mean curves with
95% CIs (mean ± 1.96 * SE across models) for each tier.

Outputs:
    outputs/figures/fig4_dose_response.pdf
    outputs/figures/fig4_dose_response.png

Usage:
    python anchorbench.paper.fig4_dose_response
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anchorbench import registry
from anchorbench.eval.io import load_records
from anchorbench.eval.metrics import compute_by_offset

from ._common import (
    DEFAULT_API_RESULTS,
    DEFAULT_FIG_DIR,
    DEFAULT_OW_RESULTS,
    discover_jsonl,
    rel_to_root,
)

# Embed real TrueType outlines instead of matplotlib's default Type 3
# fonts, which are not embedded as a FontFile and which arXiv and most
# venues flag. Affects the PDF only; the golden PNGs are unchanged.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Suites with designer-controlled offsets; History and Tool are excluded (Figure 3 caption).
SUITES = [(k, registry.suite(k).label) for k in ("external", "rag", "icl")]
OFFSETS = [15, 25, 40]


def collect_per_offset_uai(
    base: Path, suite_dir: str, key: str = "uai_plaus",
) -> dict[str, dict[int, float]]:
    """For ``base/suite_dir/<model>/results.jsonl``, return {short: {offset: val}}."""
    out: dict[str, dict[int, float]] = {}
    for short, _slug, path in discover_jsonl(base, suite_dir):
        records = load_records(str(path))
        if not records:
            continue
        by_off = compute_by_offset(records)
        out[short] = {
            int(off): m.get(key) for off, m in by_off.items() if m.get(key) is not None
        }
    return out


def aggregate(model_to_offset: dict[str, dict[int, float]]) -> dict[int, dict[str, float]]:
    """Aggregate across models -> {offset: {mean, ci_lo, ci_hi, n}}."""
    out: dict[int, dict[str, float]] = {}
    for off in OFFSETS:
        vals = [m[off] for m in model_to_offset.values() if off in m]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        n = len(arr)
        if n > 1:
            se = float(arr.std(ddof=1)) / np.sqrt(n)
        else:
            se = 0.0
        out[off] = {
            "mean": mean,
            "ci_lo": mean - 1.96 * se,
            "ci_hi": mean + 1.96 * se,
            "n": n,
        }
    return out


def plot(
    ow: dict[str, dict[int, dict[str, float]]],
    api: dict[str, dict[int, dict[str, float]]],
    out_pdf: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    suite_titles = ["External", "RAG", "ICL"]
    suite_keys = ["external", "rag", "icl"]

    for ax, key, title in zip(axes, suite_keys, suite_titles):
        ow_agg = ow.get(key, {})
        api_agg = api.get(key, {})

        x = OFFSETS
        ow_means = [ow_agg.get(o, {}).get("mean") for o in x]
        ow_lo = [ow_agg.get(o, {}).get("ci_lo") for o in x]
        ow_hi = [ow_agg.get(o, {}).get("ci_hi") for o in x]
        api_means = [api_agg.get(o, {}).get("mean") for o in x]
        api_lo = [api_agg.get(o, {}).get("ci_lo") for o in x]
        api_hi = [api_agg.get(o, {}).get("ci_hi") for o in x]

        is_icl = key == "icl"
        ow_style = dict(linestyle=":" if is_icl else "-", linewidth=1.8,
                        marker="o", markersize=5, color="#1f77b4")
        api_style = dict(linestyle=":" if is_icl else "--", linewidth=1.8,
                         marker="s", markersize=5, color="#d62728")

        if any(v is not None for v in ow_means):
            ax.plot(x, ow_means, label="Open-weight", **ow_style)
            if all(v is not None for v in ow_lo):
                ax.fill_between(x, ow_lo, ow_hi, alpha=0.18, color=ow_style["color"])
        if any(v is not None for v in api_means):
            ax.plot(x, api_means, label="API", **api_style)
            if all(v is not None for v in api_lo):
                ax.fill_between(x, api_lo, api_hi, alpha=0.18, color=api_style["color"])

        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_xlabel(r"Anchor offset $\delta$")
        ax.set_xticks(x)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(r"UAI$_{\mathrm{pls}}$")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"  wrote {rel_to_root(out_pdf)}")
    print(f"  wrote {rel_to_root(out_pdf.with_suffix('.png'))}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ow_results", type=Path, default=DEFAULT_OW_RESULTS)
    p.add_argument("--api_results", type=Path, default=DEFAULT_API_RESULTS)
    p.add_argument("--out_pdf", type=Path,
                   default=DEFAULT_FIG_DIR / "fig4_dose_response.pdf")
    args = p.parse_args()

    print("Computing per-offset UAI_pls (open-weight)...")
    ow_models = {key: collect_per_offset_uai(args.ow_results, key)
                 for key, _ in SUITES}
    print("Computing per-offset UAI_pls (API)...")
    api_models = {key: collect_per_offset_uai(args.api_results, key)
                  for key, _ in SUITES}

    ow_agg = {key: aggregate(ow_models[key]) for key, _ in SUITES}
    api_agg = {key: aggregate(api_models[key]) for key, _ in SUITES}

    print("\nSuite-mean UAI_pls by offset:")
    for key, label in SUITES:
        print(f"  {label}:")
        for off in OFFSETS:
            ow = ow_agg[key].get(off, {})
            api = api_agg[key].get(off, {})
            ow_s = f"{ow['mean']:.3f} (n={ow['n']})" if ow else "---"
            api_s = f"{api['mean']:.3f} (n={api['n']})" if api else "---"
            print(f"    offset={off:>2}: OW {ow_s:<18} API {api_s}")

    plot(ow_agg, api_agg, args.out_pdf)


if __name__ == "__main__":
    main()
