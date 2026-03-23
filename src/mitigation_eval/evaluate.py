"""Compute mitigation evaluation metrics from result JSONL files."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

N_BOOT = 2000
BOOT_SEED = 42


def _bootstrap_ci(values: list[float], n_boot: int = N_BOOT,
                   seed: int = BOOT_SEED) -> tuple[float, float]:
    """Return (lo, hi) 95% bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    means = np.sort([float(arr[rng.randint(0, n, n)].mean()) for _ in range(n_boot)])
    return float(means[int(n_boot * 0.025)]), float(means[int(n_boot * 0.975)])


def load_results(paths: list[Path]) -> list[dict]:
    """Load and concatenate JSONL result records from multiple files."""
    records = []
    for p in paths:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    log.info("Loaded %d result records from %d files", len(records), len(paths))
    return records


def compute_metrics(records: list[dict]) -> list[dict]:
    """Compute per-(model, mitigation, suite, domain) metrics."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r["model_id"], r["mitigation"], r["suite"], r["domain"])
        groups[key].append(r)

    results = []
    for (model_id, mitigation, suite, domain), recs in sorted(groups.items()):
        by_item: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for r in recs:
            if r["parsed_ok"] and r["answer_int"] is not None:
                by_item[r["item_id"]][r["condition"]].append(r["answer_int"])

        nais, maes = [], []
        n_parsed = sum(1 for r in recs if r["parsed_ok"])
        n_total = len(recs)

        for item_id, conds in by_item.items():
            ev_low = np.mean(conds.get("low_anchor", [])) if conds.get("low_anchor") else None
            ev_high = np.mean(conds.get("high_anchor", [])) if conds.get("high_anchor") else None
            if ev_low is not None and ev_high is not None:
                nais.append((ev_high - ev_low) / 60.0)

            y_star = None
            for r in recs:
                if r["item_id"] == item_id and r.get("y_star") is not None:
                    y_star = r["y_star"]
                    break
            if y_star is not None:
                for cond_vals in conds.values():
                    for v in cond_vals:
                        maes.append(abs(v - y_star))

        nai_mean = float(np.mean(nais)) if nais else None
        nai_ci = _bootstrap_ci(nais) if nais else (None, None)
        mae_mean = float(np.mean(maes)) if maes else None
        rmse = float(np.sqrt(np.mean(np.array(maes) ** 2))) if maes else None
        parse_rate = n_parsed / n_total if n_total else 0.0

        results.append({
            "model_id": model_id,
            "mitigation": mitigation,
            "suite": suite,
            "domain": domain,
            "n_items": len(by_item),
            "nai_mean": round(nai_mean, 4) if nai_mean is not None else None,
            "nai_ci_lo": round(nai_ci[0], 4) if nai_ci[0] is not None else None,
            "nai_ci_hi": round(nai_ci[1], 4) if nai_ci[1] is not None else None,
            "mae": round(mae_mean, 2) if mae_mean is not None else None,
            "rmse": round(rmse, 2) if rmse is not None else None,
            "parse_rate": round(parse_rate, 3),
            "n_records": n_total,
        })

    return results


def compute_reduction_rates(
    metrics: list[dict], baseline_mitigation: str = "B0"
) -> list[dict]:
    """Add reduction rate (RR) relative to baseline."""
    baseline_nai: dict[tuple, float] = {}
    for m in metrics:
        if m["mitigation"] == baseline_mitigation and m["nai_mean"] is not None:
            baseline_nai[(m["model_id"], m["suite"], m["domain"])] = m["nai_mean"]

    for m in metrics:
        key = (m["model_id"], m["suite"], m["domain"])
        b_nai = baseline_nai.get(key)
        if b_nai is not None and abs(b_nai) > 0.01 and m["nai_mean"] is not None:
            m["rr"] = round(1.0 - abs(m["nai_mean"]) / abs(b_nai), 3)
        else:
            m["rr"] = None
    return metrics


def write_summary_csv(metrics: list[dict], out_path: Path) -> None:
    """Write metrics list to a CSV summary file."""
    if not metrics:
        return
    fields = list(metrics[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(metrics)
    log.info("Summary CSV -> %s", out_path)


def generate_latex_table(metrics: list[dict], out_path: Path) -> None:
    """Generate LaTeX table grouped by model x mitigation, columns = suites."""
    suites = sorted(set(m["suite"] for m in metrics))
    models = sorted(set(m["model_id"] for m in metrics))
    mitigations = sorted(set(m["mitigation"] for m in metrics))

    lookup: dict[tuple, float | None] = {}
    for m in metrics:
        key = (m["model_id"], m["mitigation"], m["suite"])
        if key not in lookup or m["nai_mean"] is not None:
            lookup[key] = m["nai_mean"]

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{@{}ll" + "r" * len(suites) + r"@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Strategy} & " +
        " & ".join(f"\\textbf{{{s.capitalize()}}}" for s in suites) + r" \\",
        r"\midrule",
    ]

    for model_id in models:
        model_short = model_id.split("/")[-1]
        for i, mit in enumerate(mitigations):
            model_label = model_short if i == 0 else ""
            vals = []
            for s in suites:
                v = lookup.get((model_id, mit, s))
                vals.append(f"{v:.2f}" if v is not None else "--")
            lines.append(f"{model_label} & {mit} & " + " & ".join(vals) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\caption{NAI by model, mitigation, and suite.}",
        r"\label{tab:mitigation_main}",
        r"\end{table}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("LaTeX table -> %s", out_path)
