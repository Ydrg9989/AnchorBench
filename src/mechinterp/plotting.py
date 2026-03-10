"""Paper-ready figures for mechanistic interpretability results."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

SUITE_COLORS = {
    "external": "#1f77b4", "icl": "#ff7f0e", "rag": "#2ca02c",
    "history": "#d62728", "tool": "#9467bd", "overall": "#333333",
}


def plot_cae_vs_layer(csv_path: Path, out_path: Path, model_name: str = "") -> None:
    """Line plot: CAE vs relative depth, grouped by suite."""
    rows = _load_csv(csv_path)
    if not rows:
        return

    suites: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        suite = r.get("suite", "overall")
        layer = int(r["layer"])
        cae = float(r["cae"])
        suites.setdefault(suite, {}).setdefault(layer, []).append(cae)

    n_layers = max(max(d.keys()) for d in suites.values()) + 1

    fig, ax = plt.subplots(figsize=(7, 4))
    for suite, layer_data in sorted(suites.items()):
        xs, ys, yerr = [], [], []
        for l in range(n_layers):
            vals = layer_data.get(l, [0.0])
            xs.append(l / n_layers)
            ys.append(np.mean(vals))
            yerr.append(np.std(vals) / max(1, np.sqrt(len(vals))))
        color = SUITE_COLORS.get(suite, "#888888")
        ax.plot(xs, ys, label=suite, color=color, linewidth=1.5)
        ax.fill_between(xs, np.array(ys) - np.array(yerr),
                        np.array(ys) + np.array(yerr), alpha=0.15, color=color)

    ax.set_xlabel("Relative depth (layer / L)")
    ax.set_ylabel("CAE (causal attribution effect)")
    ax.set_title(f"Layer-wise CAE{f' — {model_name}' if model_name else ''}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved CAE plot -> %s", out_path)


def plot_top_heads(csv_path: Path, out_path: Path, top_k: int = 20) -> None:
    """Bar plot: top heads ranked by effect size."""
    rows = _load_csv(csv_path)
    if not rows:
        return

    rows.sort(key=lambda r: float(r["effect_size"]), reverse=True)
    rows = rows[:top_k]

    labels = [f"L{r['layer']}H{r['head']}" for r in rows]
    effects = [float(r["effect_size"]) for r in rows]
    layers = [int(r["layer"]) for r in rows]
    max_layer = max(layers) if layers else 1

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [plt.cm.viridis(l / max_layer) for l in layers]
    ax.barh(range(len(labels)), effects, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Effect size (CAE)")
    ax.set_title(f"Top-{top_k} attention heads by causal effect")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved top heads plot -> %s", out_path)


def plot_logit_lens(csv_path: Path, out_path: Path, model_name: str = "") -> None:
    """Line plot: JS divergence vs layer from logit lens."""
    rows = _load_csv(csv_path)
    if not rows:
        return

    layers_data: dict[int, list[float]] = {}
    for r in rows:
        layer = int(r["layer"])
        js = float(r["js_divergence"])
        layers_data.setdefault(layer, []).append(js)

    n_layers = max(layers_data.keys()) + 1
    xs, ys, yerr = [], [], []
    for l in range(n_layers):
        vals = layers_data.get(l, [0.0])
        xs.append(l / n_layers)
        ys.append(np.mean(vals))
        yerr.append(np.std(vals) / max(1, np.sqrt(len(vals))))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, color="#1f77b4", linewidth=1.5)
    ax.fill_between(xs, np.array(ys) - np.array(yerr),
                    np.array(ys) + np.array(yerr), alpha=0.2, color="#1f77b4")
    ax.set_xlabel("Relative depth (layer / L)")
    ax.set_ylabel("JS divergence (anchor vs control)")
    ax.set_title(f"Logit lens{f' — {model_name}' if model_name else ''}")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved logit lens plot -> %s", out_path)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        log.warning("CSV not found: %s", path)
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
