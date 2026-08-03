"""Rational Bayesian-bound analysis for AnchorBench (COLM 2026 rebuttal).

Addresses reviewer Cluster A (REVIEWER, REVIEWER, REVIEWER partial): "Is following a
plausible anchor actually bias, or rational Bayesian updating?"

Model. The model sees ``n`` evidence ratings ``e_1, ..., e_n`` and an anchor
``a``. Treat the model's anchor-free control response ``mu_0`` as its prior
mean (encoding the n evidence observations). Treat the anchor as one
additional noisy observation with reliability weight ``w >= 0`` relative to
the per-evidence weight (so ``w = 1`` means "anchor is worth one evidence
item", ``w = n`` means "anchor is worth all the evidence combined").

Under a Gaussian-conjugate update, the posterior mean is::

    mu_post = (n * mu_0 + w * a) / (n + w)

so the rational shift toward the anchor is ``w / (n + w) * (a - mu_0)`` and
the rational UAI (fraction of the anchor-control gap closed) is bounded by::

    UAI_rational_max(w, n) = w / (n + w)

For a single anonymous anchor ("a recent industry report suggested ~23")
treated as a noisy observation, ``w ~= 1`` is generous; ``w > n`` would mean
the model treats the anchor as more credible than all five evidence items
combined, which is hard to defend.

Output. For each (model, suite) cell we report:

* ``uai_plaus_observed``: the measured UAI on plausible anchors
* ``w_implied``: the weight that would make the observed UAI rational,
  ``w_imp = UAI_obs / (1 - UAI_obs) * n``
* ``excess_uai``: ``UAI_obs - UAI_rational_max(w_ref, n)`` for a chosen
  reference weight ``w_ref`` (default 1)
* irrelevant analogue: the rational weight for irrelevant anchors is zero
  (they carry no information), so any positive UAI on irrelevant anchors is
  unambiguously bias.

Outputs are written to ``results/rebuttal/bayesian_bound/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Default n: number of visible evidence items. Easy items: 5; hard: 3.
# Use 5 (full visible evidence) as the conservative default -- it gives the
# largest prior precision and therefore the smallest rational UAI bound, i.e.,
# the strictest test of "is the shift above rational."
DEFAULT_N_EVIDENCE = 5
DEFAULT_REFERENCE_W = 1.0
DEFAULT_W_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


@dataclass
class CellBound:
    """Bayesian-bound analysis for a single (model, suite) cell."""

    model: str
    suite: str
    n_evidence: int
    uai_irr_observed: float | None
    uai_plaus_observed: float | None
    # Rational max for several reference w values
    rational_max_at: dict[str, float]
    # Implied weights
    w_implied_plaus: float | None
    w_implied_irr: float | None
    # Excess vs reference weight
    reference_w: float
    excess_plaus_vs_ref: float | None
    excess_irr_vs_zero: float | None  # irrelevant rational max == 0
    # 95% CI on UAI_plaus from the unified file, if available
    uai_plaus_ci_lo: float | None
    uai_plaus_ci_hi: float | None
    # Significance flag: True if the entire 95% CI for UAI_plaus lies above
    # the reference rational bound
    plaus_above_rational_ci: bool | None
    # Implied weight if we use the LOWER 95% CI bound (most conservative test
    # of "above rational"). None if no CI was available.
    w_implied_plaus_ci_lo: float | None


def rational_uai_max(w: float, n: int) -> float:
    """Rational UAI ceiling under one extra Gaussian observation of weight w."""
    if n <= 0:
        return 1.0
    if w < 0:
        return 0.0
    return w / (n + w)


def implied_weight(uai_obs: float, n: int) -> float | None:
    """Back-solve for the weight w that would make ``uai_obs`` rational.

    Solves ``uai_obs = w / (n + w)`` => ``w = uai_obs * n / (1 - uai_obs)``.
    Returns ``None`` for ``uai_obs >= 1`` (no finite weight) and ``0`` for
    ``uai_obs <= 0`` (anchor pulled the answer the wrong way; not interpretable
    as rational weight).
    """
    if uai_obs is None or not math.isfinite(uai_obs):
        return None
    if uai_obs <= 0:
        return 0.0
    if uai_obs >= 1.0:
        return None
    return uai_obs * n / (1.0 - uai_obs)


def analyze_cell(
    entry: dict,
    n_evidence: int = DEFAULT_N_EVIDENCE,
    reference_w: float = DEFAULT_REFERENCE_W,
    w_grid: Iterable[float] = DEFAULT_W_GRID,
) -> CellBound:
    """Compute Bayesian-bound metrics for a single unified-summary entry."""
    uai_irr = entry.get("uai_irr")
    uai_plaus = entry.get("uai_plaus")
    plaus_ci = entry.get("uai_plaus_ci") or {}
    ci_lo = plaus_ci.get("lo")
    ci_hi = plaus_ci.get("hi")

    rational_grid = {
        f"w={w:g}": rational_uai_max(w, n_evidence) for w in w_grid
    }
    ref_rational = rational_uai_max(reference_w, n_evidence)

    excess_plaus = (
        (uai_plaus - ref_rational)
        if uai_plaus is not None and math.isfinite(uai_plaus)
        else None
    )
    excess_irr = (
        (uai_irr - 0.0)
        if uai_irr is not None and math.isfinite(uai_irr)
        else None
    )

    plaus_above_ci: bool | None = None
    w_imp_ci_lo: float | None = None
    if ci_lo is not None and math.isfinite(ci_lo):
        plaus_above_ci = ci_lo > ref_rational
        w_imp_ci_lo = implied_weight(ci_lo, n_evidence)

    return CellBound(
        model=entry["model"],
        suite=entry["suite"],
        n_evidence=n_evidence,
        uai_irr_observed=uai_irr,
        uai_plaus_observed=uai_plaus,
        rational_max_at=rational_grid,
        w_implied_plaus=implied_weight(uai_plaus, n_evidence),
        w_implied_irr=implied_weight(uai_irr, n_evidence),
        reference_w=reference_w,
        excess_plaus_vs_ref=excess_plaus,
        excess_irr_vs_zero=excess_irr,
        uai_plaus_ci_lo=ci_lo,
        uai_plaus_ci_hi=ci_hi,
        plaus_above_rational_ci=plaus_above_ci,
        w_implied_plaus_ci_lo=w_imp_ci_lo,
    )


def analyze_unified(
    entries: list[dict],
    n_evidence: int = DEFAULT_N_EVIDENCE,
    reference_w: float = DEFAULT_REFERENCE_W,
    w_grid: Iterable[float] = DEFAULT_W_GRID,
) -> list[CellBound]:
    return [
        analyze_cell(e, n_evidence=n_evidence, reference_w=reference_w, w_grid=w_grid)
        for e in entries
    ]


# --- Reporters --------------------------------------------------------------

_PRIMARY_SUITES = ("External", "History", "Rag", "Tool", "Icl")


def _fmt(v: float | None, prec: int = 2, dash: str = "---") -> str:
    if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
        return dash
    return f"{v:.{prec}f}"


def write_json(bounds: list[CellBound], path: Path) -> None:
    payload = [asdict(b) for b in bounds]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s (%d cells)", path, len(payload))


def write_csv(bounds: list[CellBound], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "model", "suite", "n_evidence",
        "uai_irr_observed", "uai_plaus_observed",
        "uai_plaus_ci_lo", "uai_plaus_ci_hi",
        "w_implied_irr", "w_implied_plaus",
        "reference_w", "excess_irr_vs_zero", "excess_plaus_vs_ref",
        "plaus_above_rational_ci",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for b in bounds:
            row = [getattr(b, k) for k in keys]
            w.writerow(row)
    log.info("Wrote %s", path)


def write_latex_table(
    bounds: list[CellBound],
    path: Path,
    reference_w: float = DEFAULT_REFERENCE_W,
    n_evidence: int = DEFAULT_N_EVIDENCE,
    suites: tuple[str, ...] = _PRIMARY_SUITES,
) -> None:
    """Wide table: rows = models, columns = suites; reports w_implied (plaus)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rational_ref = rational_uai_max(reference_w, n_evidence)
    by_model: dict[str, dict[str, CellBound]] = {}
    for b in bounds:
        by_model.setdefault(b.model, {})[b.suite] = b
    models = sorted(by_model.keys())

    lines = [
        r"% Auto-generated by anchorbench.analysis.bayesian_bound",
        r"% Cluster A response: implied anchor weight for plausible anchors.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l " + " ".join(["r"] * len(suites)) + r"}",
        r"\toprule",
        "Model & " + " & ".join(suites) + r" \\",
        r"\midrule",
    ]
    for m in models:
        cells = by_model[m]
        row = [m]
        for s in suites:
            b = cells.get(s)
            if b is None or b.w_implied_plaus is None:
                row.append("---")
            else:
                marker = ""
                if b.plaus_above_rational_ci:
                    marker = r"$^{*}$"
                row.append(f"{b.w_implied_plaus:.2f}{marker}")
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Implied anchor weight $w_{\mathrm{imp}}$ on \emph{plausible}"
            r" anchors, defined as the smallest weight under a Gaussian-conjugate"
            r" Bayesian model that would make the observed mean UAI rational"
            r" (with $n=" + str(n_evidence) + r"$ evidence ratings). Under the"
            r" reference assumption that an anonymous anchor is worth one"
            r" evidence item ($w=" + f"{reference_w:g}" + r"$), the rational"
            r" UAI ceiling is " + f"{rational_ref:.3f}" + r"; $w_{\mathrm{imp}}>"
            + f"{reference_w:g}" + r"$ means the model treats the anchor as more"
            r" credible than a single evidence item. $w_{\mathrm{imp}}>n$ means"
            r" the model treats the anchor as more credible than all visible"
            r" evidence combined. $^{*}$ marks cells where the 95\% CI for"
            r" UAI$_{\mathrm{pls}}$ lies entirely above the rational ceiling.}"
        ),
        r"\label{tab:rebuttal_implied_w}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", path)


def write_excess_table(
    bounds: list[CellBound],
    path: Path,
    suites: tuple[str, ...] = _PRIMARY_SUITES,
) -> None:
    """Wide table: rows = models, columns = suites; reports excess_plaus_vs_ref."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, dict[str, CellBound]] = {}
    for b in bounds:
        by_model.setdefault(b.model, {})[b.suite] = b
    models = sorted(by_model.keys())
    ref_w = bounds[0].reference_w if bounds else DEFAULT_REFERENCE_W
    n_ev = bounds[0].n_evidence if bounds else DEFAULT_N_EVIDENCE
    ceiling = rational_uai_max(ref_w, n_ev)

    lines = [
        r"% Auto-generated by anchorbench.analysis.bayesian_bound",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l " + " ".join(["r"] * len(suites)) + r"}",
        r"\toprule",
        "Model & " + " & ".join(suites) + r" \\",
        r"\midrule",
    ]
    for m in models:
        cells = by_model[m]
        row = [m]
        for s in suites:
            b = cells.get(s)
            if b is None or b.excess_plaus_vs_ref is None:
                row.append("---")
            else:
                v = b.excess_plaus_vs_ref
                marker = r"$^{*}$" if b.plaus_above_rational_ci else ""
                row.append(f"{v:+.3f}{marker}")
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Excess UAI on plausible anchors above the rational"
            r" Bayesian ceiling " + f"{ceiling:.3f}" + r" (assuming the anchor"
            r" is worth one evidence item, $w=" + f"{ref_w:g}" + r"$, $n="
            + str(n_ev) + r"$ evidence ratings). Positive values indicate the"
            r" model shifted further toward the anchor than a rational"
            r" Bayesian update could justify. $^{*}$ marks cells where the"
            r" 95\% CI for UAI$_{\mathrm{pls}}$ lies entirely above the"
            r" ceiling.}"
        ),
        r"\label{tab:rebuttal_excess_uai}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", path)


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    if n % 2 == 1:
        return xs_sorted[n // 2]
    return 0.5 * (xs_sorted[n // 2 - 1] + xs_sorted[n // 2])


def write_markdown_report(
    bounds: list[CellBound],
    path: Path,
    reference_w: float = DEFAULT_REFERENCE_W,
    n_evidence: int = DEFAULT_N_EVIDENCE,
) -> None:
    """Human-readable interpretation aimed at the rebuttal author."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ceiling = rational_uai_max(reference_w, n_evidence)
    by_suite: dict[str, list[CellBound]] = {}
    for b in bounds:
        by_suite.setdefault(b.suite, []).append(b)

    lines = [
        "# Bayesian-Bound Analysis (Cluster A response)\n",
        "## Setup\n",
        (
            f"- n_evidence = {n_evidence} (visible ratings on easy items)\n"
            f"- Reference w = {reference_w} (anchor worth one evidence item)\n"
            f"- Rational UAI ceiling = w/(n+w) = {ceiling:.3f}\n"
        ),
        "## Interpretation rules\n",
        (
            "- **Irrelevant anchors:** the rational weight is 0 (they carry no\n"
            "  information about the answer). Any positive UAI on irrelevant\n"
            "  anchors is unambiguously bias above rational updating.\n"
            "- **Plausible anchors:** UAI_plaus > w/(n+w) implies the model\n"
            "  weights the anchor more than the reference assumption. The\n"
            "  implied weight w_imp = UAI/(1-UAI) * n is the smallest weight\n"
            "  consistent with rationality.\n"
            "- **w_imp > n** means the model treats a single anonymous anchor\n"
            "  as more credible than ALL visible evidence combined --- a\n"
            "  pattern that is hard to defend on rational grounds.\n"
        ),
        "## Per-suite summary\n",
    ]

    for suite in _PRIMARY_SUITES:
        cells = by_suite.get(suite)
        if not cells:
            continue
        lines.append(f"### {suite}\n")
        # Average across models
        plaus_vals = [c.uai_plaus_observed for c in cells if c.uai_plaus_observed is not None]
        irr_vals = [c.uai_irr_observed for c in cells if c.uai_irr_observed is not None]
        w_vals = [c.w_implied_plaus for c in cells if c.w_implied_plaus is not None]
        above = sum(1 for c in cells if c.plaus_above_rational_ci)
        n_cells = len(cells)
        if plaus_vals:
            lines.append(
                f"- mean UAI_pls = {sum(plaus_vals)/len(plaus_vals):.3f}, "
                f"mean UAI_irr = {sum(irr_vals)/len(irr_vals):.3f} "
                f"({n_cells} models)\n"
            )
        if w_vals:
            med_w = _median(w_vals)
            n_above_n = sum(1 for w in w_vals if w > n_evidence)
            n_above_1 = sum(1 for w in w_vals if w > reference_w)
            ci_lo_w = [c.w_implied_plaus_ci_lo for c in cells
                       if c.w_implied_plaus_ci_lo is not None]
            med_w_ci_lo = _median(ci_lo_w) if ci_lo_w else None
            lines.append(
                f"- median implied w (plausible) = {med_w:.2f} "
                f"(reference w = {reference_w:g}); "
                f"{n_above_1}/{len(w_vals)} cells exceed reference; "
                f"{n_above_n}/{len(w_vals)} cells imply w > n = {n_evidence} "
                f"(anchor weighted more than all visible evidence combined)\n"
            )
            if med_w_ci_lo is not None:
                lines.append(
                    f"- median implied w using the 95% CI LOWER bound for "
                    f"UAI_pls (conservative) = {med_w_ci_lo:.2f}\n"
                )
            lines.append(
                f"- {above}/{n_cells} cells have UAI_pls 95% CI entirely above "
                f"the rational ceiling {ceiling:.3f} (i.e., the anchor effect "
                f"is significantly above rational at p<0.05)\n"
            )

        # Worst offenders
        sorted_cells = sorted(
            (c for c in cells if c.w_implied_plaus is not None),
            key=lambda c: -(c.w_implied_plaus or 0),
        )[:3]
        if sorted_cells:
            lines.append("- top 3 implied weights:\n")
            for c in sorted_cells:
                lines.append(
                    f"  - {c.model}: UAI_pls={_fmt(c.uai_plaus_observed,3)} "
                    f"=> w_imp={c.w_implied_plaus:.2f}\n"
                )
        lines.append("\n")

    lines.append("## Headline numbers for the rebuttal\n")
    plaus_all = [b for b in bounds if b.uai_plaus_observed is not None and b.suite != "Icl"]
    if plaus_all:
        w_vals = [b.w_implied_plaus for b in plaus_all if b.w_implied_plaus is not None]
        ci_lo_vals = [b.w_implied_plaus_ci_lo for b in plaus_all
                      if b.w_implied_plaus_ci_lo is not None]
        n_above = sum(1 for b in plaus_all if b.plaus_above_rational_ci)
        n_above_n = sum(1 for w in w_vals if w > n_evidence)
        if w_vals:
            lines.append(
                f"Across {len(plaus_all)} non-ICL (model, suite) cells:\n"
                f"- **{n_above}/{len(plaus_all)} cells "
                f"({100 * n_above / len(plaus_all):.0f}%) have the 95% CI of "
                f"UAI_pls entirely above the rational ceiling "
                f"({ceiling:.3f}, w=1).** This is the primary headline.\n"
                f"- median implied w (point estimate) = {_median(w_vals):.2f} "
                f"(reference = 1; mean is skewed by tail outliers).\n"
                f"- median implied w using 95% CI lower bound = "
                f"{(_median(ci_lo_vals) if ci_lo_vals else 0):.2f} "
                f"(the most conservative estimate).\n"
                f"- {n_above_n}/{len(w_vals)} cells imply w > n = {n_evidence}, "
                f"i.e., the anchor is treated as more credible than all visible "
                f"evidence combined --- a regime that is hard to defend on "
                f"rational grounds.\n"
                f"- For irrelevant anchors, the rational weight is 0, so ANY "
                f"positive UAI_irr is bias above rational updating "
                f"(see Table 1 / Table 2 of the paper).\n"
            )

    path.write_text("".join(lines))
    log.info("Wrote %s", path)


# --- CLI --------------------------------------------------------------------


def _load_unified(unified_paths: Iterable[Path]) -> list[dict]:
    out: list[dict] = []
    for p in unified_paths:
        if not p.exists():
            log.warning("Unified file not found: %s", p)
            continue
        entries = json.loads(p.read_text())
        if isinstance(entries, list):
            out.extend(entries)
        else:
            out.append(entries)
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Bayesian-bound analysis for AnchorBench")
    p.add_argument(
        "--unified", nargs="+", type=Path,
        default=[
            Path("results/full_benchmark/unified_all_suites.json"),
            Path("results/api_benchmark/unified_all_suites.json"),
        ],
        help="Unified summary JSON files to analyze (default: full + api benchmarks)",
    )
    p.add_argument(
        "--out_dir", type=Path,
        default=Path("results/rebuttal/bayesian_bound"),
    )
    p.add_argument("--n_evidence", type=int, default=DEFAULT_N_EVIDENCE)
    p.add_argument("--reference_w", type=float, default=DEFAULT_REFERENCE_W)
    args = p.parse_args(argv)

    entries = _load_unified(args.unified)
    if not entries:
        log.error("No unified entries loaded; aborting")
        raise SystemExit(1)
    log.info("Loaded %d entries from %d files", len(entries), len(args.unified))

    bounds = analyze_unified(
        entries,
        n_evidence=args.n_evidence,
        reference_w=args.reference_w,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(bounds, out_dir / "bayesian_bound.json")
    write_csv(bounds, out_dir / "bayesian_bound.csv")
    write_latex_table(
        bounds, out_dir / "implied_weight_table.tex",
        reference_w=args.reference_w, n_evidence=args.n_evidence,
    )
    write_excess_table(bounds, out_dir / "excess_uai_table.tex")
    write_markdown_report(
        bounds, out_dir / "interpretation.md",
        reference_w=args.reference_w, n_evidence=args.n_evidence,
    )

    # Console summary
    print("\n=== Bayesian-bound summary (plausible anchors, non-ICL) ===")
    non_icl = [b for b in bounds if b.suite != "Icl"]
    print(f"  cells: {len(non_icl)}")
    w_vals = [b.w_implied_plaus for b in non_icl if b.w_implied_plaus is not None]
    if w_vals:
        print(f"  mean implied w: {sum(w_vals)/len(w_vals):.3f}")
        print(f"  max implied w:  {max(w_vals):.3f}")
        print(f"  cells with w > n ({args.n_evidence}): "
              f"{sum(1 for w in w_vals if w > args.n_evidence)}/{len(w_vals)}")
    above = sum(1 for b in non_icl if b.plaus_above_rational_ci)
    print(f"  cells with UAI_pls 95% CI entirely above ceiling: "
          f"{above}/{len(non_icl)}")
    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()
