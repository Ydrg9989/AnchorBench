"""Generate LaTeX table snippets from mechinterp CSV results."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def generate_imprint_table(csv_path: Path, out_path: Path) -> None:
    """LaTeX table of anchor imprint metrics grouped by suite."""
    if not csv_path.exists():
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    suites: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        suite = r.get("suite", "all")
        for cond_suffix, cond_label in [("_low", "low_anchor"), ("_high", "high_anchor")]:
            key = f"{suite}_{cond_label}"
            for base_metric in ("kl", "tvd", "delta_ev"):
                col = f"{base_metric}{cond_suffix}"
                if col in r and r[col]:
                    suites.setdefault(key, {}).setdefault(base_metric, []).append(float(r[col]))

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{KL}} & \multicolumn{2}{c}{\textbf{TVD}} & \multicolumn{2}{c}{\textbf{$\Delta$EV}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"\textbf{Suite} & Low & High & Low & High & Low & High \\",
        r"\midrule",
    ]

    for suite_name in ("external", "icl", "rag", "tool", "history"):
        vals = []
        for cond in ("low_anchor", "high_anchor"):
            key = f"{suite_name}_{cond}"
            for metric in ("kl", "tvd", "delta_ev"):
                data = suites.get(key, {}).get(metric, [])
                vals.append(f"{np.mean(data):.2f}" if data else "--")
        lines.append(f"{suite_name.capitalize()} & {' & '.join(vals)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Anchor imprint metrics by suite.}", r"\label{tab:imprint}", r"\end{table}"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("LaTeX table -> %s", out_path)
