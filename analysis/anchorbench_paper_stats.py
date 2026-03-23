#!/usr/bin/env python3
"""Paper-facing statistical analysis for AnchorBench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "full_benchmark"
UNIFIED_PATH = RESULTS_ROOT / "unified_all_suites.json"
OUT_STATS = ROOT / "outputs" / "stats"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_SUMMARY = OUT_STATS / "STATISTICAL_ANALYSIS_SUMMARY.md"

SUITE_ORDER = ["External", "History", "Icl", "Rag", "Tool"]
EPSILON = 3.0
N_BOOT = 2000
SEED = 42


@dataclass
class ClaimResult:
    claim: str
    comparison: str
    effect_size: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_raw: float | None
    p_adj_bh: float | None
    n_units: int | None
    notes: str


def bootstrap_ci(values: Iterable[float], n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    n = arr.size
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[i] = arr[idx].mean()
    return (float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def bh_adjust(p_values: list[float | None]) -> list[float | None]:
    vals = [(i, p) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    if not vals:
        return [None] * len(p_values)
    vals.sort(key=lambda x: x[1])
    m = len(vals)
    adj = [None] * len(p_values)
    prev = 1.0
    for rank, (idx, p) in enumerate(reversed(vals), start=1):
        k = m - rank + 1
        q = min(prev, p * m / k)
        prev = q
        adj[idx] = q
    return adj


def load_unified() -> pd.DataFrame:
    with open(UNIFIED_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["suite"] = pd.Categorical(df["suite"], categories=SUITE_ORDER, ordered=True)
    return df.sort_values(["model", "suite"]).reset_index(drop=True)


def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_results_paths() -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for suite in ["external", "history", "icl", "rag", "tool"]:
        suite_dir = RESULTS_ROOT / suite
        if not suite_dir.exists():
            continue
        for p in sorted(suite_dir.glob("*/results.jsonl")):
            rows.append((suite.capitalize() if suite != "icl" else "Icl", p.parent.name, p))
    return rows


def item_level_metrics(records: list[dict], epsilon: float = EPSILON) -> dict[str, np.ndarray]:
    parse_vec = np.array([1.0 if (r.get("parsed_ok") and r.get("answer_int") is not None) else 0.0 for r in records], dtype=float)

    by_item: dict[str, dict[str, dict]] = {}
    for r in records:
        by_item.setdefault(r["item_id"], {})[r["condition"]] = r

    acc10_vals: list[float] = []
    uai_irr_vals: list[float] = []
    uai_pls_vals: list[float] = []
    disc_vals: list[float] = []
    paired_pls_minus_irr: list[float] = []

    for conds in by_item.values():
        ctrl = conds.get("control")
        if ctrl is None:
            continue
        y_ctrl = ctrl.get("answer_int")
        y_star = ctrl.get("y_star_evidence")
        if y_ctrl is not None and y_star is not None:
            acc10_vals.append(1.0 if abs(y_ctrl - y_star) <= 10 else 0.0)

        irr_item: list[float] = []
        pls_item: list[float] = []
        if y_ctrl is not None:
            for c, bucket in [("irrelevant_low", irr_item), ("irrelevant_high", irr_item), ("plausible_low", pls_item), ("plausible_high", pls_item)]:
                rec = conds.get(c)
                if rec is None:
                    continue
                y = rec.get("answer_int")
                a = rec.get("anchor_value")
                if y is None or a is None:
                    continue
                denom = a - y_ctrl
                if abs(denom) < epsilon:
                    continue
                bucket.append((y - y_ctrl) / denom)

        irr_mean = float(np.mean(irr_item)) if irr_item else None
        pls_mean = float(np.mean(pls_item)) if pls_item else None
        if irr_mean is not None:
            uai_irr_vals.append(irr_mean)
        if pls_mean is not None:
            uai_pls_vals.append(pls_mean)
        if irr_mean is not None and pls_mean is not None:
            disc_vals.append(pls_mean - irr_mean)
            paired_pls_minus_irr.append(pls_mean - irr_mean)

    return {
        "parse": parse_vec,
        "acc10": np.asarray(acc10_vals, dtype=float),
        "uai_irr": np.asarray(uai_irr_vals, dtype=float),
        "uai_pls": np.asarray(uai_pls_vals, dtype=float),
        "disc": np.asarray(disc_vals, dtype=float),
        "paired_pls_minus_irr": np.asarray(paired_pls_minus_irr, dtype=float),
    }


def bootstrap_metric_table() -> pd.DataFrame:
    rows: list[dict] = []
    for suite, model_slug, p in get_results_paths():
        recs = load_records(p)
        m = item_level_metrics(recs)

        for key, label in [
            ("acc10", "Acc10_c"),
            ("uai_irr", "UAI_irr"),
            ("uai_pls", "UAI_pls"),
            ("disc", "DiscDelta"),
            ("parse", "Parse"),
        ]:
            vals = m[key]
            mean, lo, hi = bootstrap_ci(vals)
            rows.append(
                {
                    "suite": suite,
                    "model_slug": model_slug,
                    "metric": label,
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_used": int(vals.size),
                }
            )
    return pd.DataFrame(rows).sort_values(["suite", "model_slug", "metric"]).reset_index(drop=True)


def safe_wilcoxon(x: np.ndarray, y: np.ndarray | None = None) -> float | None:
    try:
        if y is None:
            if x.size == 0 or np.allclose(x, 0):
                return None
            return float(wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)
        n = min(x.size, y.size)
        if n == 0:
            return None
        d = x[:n] - y[:n]
        if np.allclose(d, 0):
            return None
        return float(wilcoxon(x[:n], y[:n], zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return None


def claim_tests(unified: pd.DataFrame, boot_tbl: pd.DataFrame) -> pd.DataFrame:
    claims: list[ClaimResult] = []

    # Claim 1: interface dependence (suite spread in DiscDelta across interfaces).
    suite_means = unified.groupby("suite", observed=True)["disc_delta"].mean()
    effect = float(suite_means.max() - suite_means.min())
    # Conservative CI via bootstrap over model-suite rows.
    disc = unified["disc_delta"].to_numpy(float)
    rng = np.random.RandomState(SEED)
    b = []
    for _ in range(N_BOOT):
        s = disc[rng.randint(0, disc.size, size=disc.size)]
        # approximate by grouping resampled rows by suite label from a resampled dataframe
        idx = rng.randint(0, unified.shape[0], size=unified.shape[0])
        d2 = unified.iloc[idx]
        means = d2.groupby("suite", observed=True)["disc_delta"].mean()
        b.append(float(means.max() - means.min()))
    ci_lo, ci_hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    # p-value from paired nonparametric Friedman not requested and not available in scipy without assumptions here.
    claims.append(
        ClaimResult(
            claim="Interface-dependent anchoring",
            comparison="Range of suite mean DiscDelta",
            effect_size=effect,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            p_raw=None,
            p_adj_bh=None,
            n_units=int(unified.shape[0]),
            notes="Effect-size CI from bootstrap over model-suite rows; no single omnibus p-value reported.",
        )
    )

    # Claim 2: plausible > irrelevant, paired by model within each suite.
    for suite in SUITE_ORDER:
        d = unified[unified["suite"] == suite]
        diff_raw = d["uai_plaus"] - d["uai_irr"]
        diff = diff_raw[np.isfinite(diff_raw)].to_numpy(float)
        mean, lo, hi = bootstrap_ci(diff)
        p = safe_wilcoxon(diff)
        claims.append(
            ClaimResult(
                claim="Plausible anchors usually > irrelevant",
                comparison=f"{suite}: UAI_plaus - UAI_irr (paired by model)",
                effect_size=mean,
                ci_lo=lo,
                ci_hi=hi,
                p_raw=p,
                p_adj_bh=None,
                n_units=int(diff.size),
                notes="Wilcoxon signed-rank across models within suite.",
            )
        )

    # Claim 3: accuracy vs robustness are different abilities (across model-suite pairs).
    sub = unified[np.isfinite(unified["disc_delta"])].reset_index(drop=True)
    x = sub["acc10_control"].to_numpy(float)
    y = sub["disc_delta"].to_numpy(float)
    corr = float(np.corrcoef(x, y)[0, 1])
    # bootstrap CI for correlation
    rng = np.random.RandomState(SEED)
    boots = []
    n = x.size
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        boots.append(float(np.corrcoef(xb, yb)[0, 1]))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    claims.append(
        ClaimResult(
            claim="Accuracy and robustness dissociate",
            comparison="Correlation: Acc10_c vs DiscDelta (all model-suite pairs)",
            effect_size=corr,
            ci_lo=lo,
            ci_hi=hi,
            p_raw=None,
            p_adj_bh=None,
            n_units=int(n),
            notes="Bootstrap CI for Pearson correlation; inferential p-value omitted for conservative reporting.",
        )
    )

    # Claim 4a: Tool protocol sensitivity via parse-rate drop vs non-Tool, paired by model.
    piv_parse = unified.pivot(index="model", columns="suite", values="parse_rate")
    if "Tool" in piv_parse.columns:
        non_tool_cols = [c for c in piv_parse.columns if c != "Tool"]
        tool = piv_parse["Tool"].to_numpy(float)
        non_tool_mean = piv_parse[non_tool_cols].mean(axis=1).to_numpy(float)
        diff = tool - non_tool_mean
        mean, lo, hi = bootstrap_ci(diff)
        p = safe_wilcoxon(diff)
        claims.append(
            ClaimResult(
                claim="Tool partly reflects protocol compliance",
                comparison="Tool Parse - mean(Parse of External/History/ICL/RAG), paired by model",
                effect_size=mean,
                ci_lo=lo,
                ci_hi=hi,
                p_raw=p,
                p_adj_bh=None,
                n_units=int(diff.size),
                notes="Negative effect implies lower parse/protocol compliance in Tool.",
            )
        )

    # Additional requested paired comparisons.
    # History vs ICL by model (DiscDelta).
    piv_disc = unified.pivot(index="model", columns="suite", values="disc_delta")
    if "History" in piv_disc.columns and "Icl" in piv_disc.columns:
        diff = (piv_disc["History"] - piv_disc["Icl"]).to_numpy(float)
        mean, lo, hi = bootstrap_ci(diff)
        p = safe_wilcoxon(diff)
        claims.append(
            ClaimResult(
                claim="Requested paired comparison",
                comparison="History - ICL DiscDelta (paired by model)",
                effect_size=mean,
                ci_lo=lo,
                ci_hi=hi,
                p_raw=p,
                p_adj_bh=None,
                n_units=int(diff.size),
                notes="Item-level pairing across suites not available; model-level pairing used.",
            )
        )

    # Tool vs other suites by model (DiscDelta).
    if "Tool" in piv_disc.columns:
        non_tool_cols = [c for c in piv_disc.columns if c != "Tool"]
        diff_raw = piv_disc["Tool"] - piv_disc[non_tool_cols].mean(axis=1)
        diff = diff_raw[np.isfinite(diff_raw)].to_numpy(float)
        mean, lo, hi = bootstrap_ci(diff)
        p = safe_wilcoxon(diff)
        claims.append(
            ClaimResult(
                claim="Requested paired comparison",
                comparison="Tool - mean(other suites) DiscDelta (paired by model)",
                effect_size=mean,
                ci_lo=lo,
                ci_hi=hi,
                p_raw=p,
                p_adj_bh=None,
                n_units=int(diff.size),
                notes="Model-level pairing used.",
            )
        )

    # Tool-Read vs Tool-Agentic availability check.
    tr = (RESULTS_ROOT / "tool_read").exists()
    ta = (RESULTS_ROOT / "tool_agentic").exists()
    if not (tr and ta):
        claims.append(
            ClaimResult(
                claim="Requested paired comparison",
                comparison="Tool-Read vs Tool-Agentic",
                effect_size=None,
                ci_lo=None,
                ci_hi=None,
                p_raw=None,
                p_adj_bh=None,
                n_units=None,
                notes="Blocked: tool_read/tool_agentic source-of-truth runs not found under results/full_benchmark.",
            )
        )

    df = pd.DataFrame([c.__dict__ for c in claims])
    df["p_adj_bh"] = bh_adjust(df["p_raw"].tolist())
    return df


def save_df(df: pd.DataFrame, stem: str) -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_TABLES / f"{stem}.csv"
    tex_path = OUT_TABLES / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))


def write_summary(unified: pd.DataFrame, claim_df: pd.DataFrame) -> None:
    OUT_STATS.mkdir(parents=True, exist_ok=True)
    n_runs = unified.shape[0]
    n_models = unified["model"].nunique()
    n_suites = unified["suite"].nunique()
    parse_min = float(unified["parse_rate"].min())
    parse_max = float(unified["parse_rate"].max())

    lines: list[str] = []
    lines.append("# AnchorBench Statistical Analysis Summary")
    lines.append("")
    lines.append("## Source files used")
    lines.append(f"- `{UNIFIED_PATH.relative_to(ROOT)}` (machine-readable model-suite summaries)")
    lines.append(f"- `{RESULTS_ROOT.relative_to(ROOT)}/**/results.jsonl` (item-level records)")
    lines.append("")
    lines.append("## Feasible analyses performed")
    lines.append("- Bootstrap CIs for `Acc10_c`, `UAI_irr`, `UAI_pls`, `DiscDelta`, `Parse` at model-suite level from item-level vectors.")
    lines.append("- Paired nonparametric tests (Wilcoxon signed-rank) for requested comparisons where pairing is valid.")
    lines.append("- BH multiple-comparison correction across reported raw p-values.")
    lines.append("")
    lines.append("## Bootstrap method")
    lines.append(f"- Nonparametric bootstrap over analysis units with replacement, `n_boot={N_BOOT}`, `seed={SEED}`, two-sided 95% percentile CI.")
    lines.append("- Item-level vectors are suite/metric-specific; parse failures are preserved and explicitly reflected in parse-rate vectors.")
    lines.append("")
    lines.append("## Parse and missing-data handling")
    lines.append("- `Parse` uses all records (parsed vs not parsed).")
    lines.append("- `Acc10_c` uses control items with parsed control answers.")
    lines.append("- `UAI`/`DiscDelta` use items with required parsed control+anchor answers and denominator filter `|a - y_ctrl| >= 3.0`.")
    lines.append("")
    lines.append("## Limitations / blockers")
    lines.append("- Tool-Read vs Tool-Agentic is blocked in `results/full_benchmark`: those suites are not present as source-of-truth runs.")
    lines.append("- History vs ICL item-level pairing is not valid because suites are different task interfaces; model-level paired tests were used conservatively.")
    lines.append("- For some broad claims (e.g., interface dependence omnibus), we report bootstrap effect CIs without forcing potentially invalid omnibus p-values.")
    lines.append("")
    lines.append("## Dataset scope")
    lines.append(f"- Model-suite runs: {n_runs} ({n_models} models x {n_suites} suites)")
    lines.append(f"- Parse-rate range across runs: [{parse_min:.4f}, {parse_max:.4f}]")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("- `outputs/stats/bootstrap_model_suite_metrics.csv` and `.json`")
    lines.append("- `outputs/tables/main_claims_statistical_table.csv` and `.tex`")
    lines.append("- `outputs/tables/appendix_bootstrap_metrics_table.csv` and `.tex`")
    lines.append("- `outputs/tables/appendix_paired_comparisons_table.csv` and `.tex`")
    lines.append("- `outputs/stats/STATISTICAL_ANALYSIS_SUMMARY.md`")

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_STATS.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    unified = load_unified()
    boot_tbl = bootstrap_metric_table()
    boot_tbl.to_csv(OUT_STATS / "bootstrap_model_suite_metrics.csv", index=False)
    boot_tbl.to_json(OUT_STATS / "bootstrap_model_suite_metrics.json", orient="records", indent=2)

    claims = claim_tests(unified, boot_tbl)

    # Main claims table: concise and paper-facing.
    main_claims = claims[
        [
            "claim",
            "comparison",
            "effect_size",
            "ci_lo",
            "ci_hi",
            "p_raw",
            "p_adj_bh",
            "n_units",
            "notes",
        ]
    ].copy()
    save_df(main_claims, "main_claims_statistical_table")

    # Appendix tables.
    save_df(boot_tbl, "appendix_bootstrap_metrics_table")
    paired = claims[claims["comparison"].str.contains("paired", case=False, na=False)].copy()
    save_df(paired, "appendix_paired_comparisons_table")

    write_summary(unified, claims)
    print("Wrote statistical outputs to outputs/stats and outputs/tables.")


if __name__ == "__main__":
    main()

