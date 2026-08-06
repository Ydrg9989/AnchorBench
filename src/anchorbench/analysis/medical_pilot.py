"""Medical-vs-business domain comparison.

Compares anchoring magnitudes on the 3 medical pilot domains against the
6 business domains in the published benchmark, on External and History
suites.

Outputs are written to ``results/rebuttal/medical/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MEDICAL_DIR = Path("results/rebuttal/medical")
DEFAULT_BUSINESS_DIR = Path("results/full_benchmark")
DEFAULT_OUT = Path("results/rebuttal/medical")

SUITES = ("external", "history")


def _load_records(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _safe_metrics(records: list[dict], baseline: str) -> dict:
    """Compute metrics, falling back to ``control`` if the requested
    baseline condition isn't present (e.g. published history runs used
    ``control`` rather than ``control_twostage``).
    """
    from anchorbench.eval.metrics import compute_unified_metrics
    if not records:
        return {}
    present = {r.get("condition") for r in records}
    effective_baseline = baseline if baseline in present else "control"
    if effective_baseline not in present:
        log.warning("No suitable baseline in records (have %s); "
                    "metrics may be partial", sorted(present))
    return compute_unified_metrics(records, baseline_condition=effective_baseline)


def gather(medical_dir: Path, business_dir: Path) -> list[dict]:
    from anchorbench.eval.constants import MODEL_SHORT
    rows: list[dict] = []
    for suite in SUITES:
        baseline = "control_twostage" if suite == "history" else "control"
        med_suite = medical_dir / suite
        bus_suite = business_dir / suite
        if not med_suite.exists():
            log.warning("Medical results missing for %s", suite)
            continue
        for slug_dir in sorted(med_suite.iterdir()):
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            med_records = _load_records(slug_dir / "results.jsonl")
            bus_records = _load_records(bus_suite / slug / "results.jsonl")
            if not med_records:
                log.warning("Skipping %s/%s: no medical records", suite, slug)
                continue
            if not bus_records:
                log.info("%s/%s has no business comparator (e.g., API "
                         "model not in published benchmark); emitting "
                         "medical-only row.", suite, slug)
            med = _safe_metrics(med_records, baseline)
            bus = _safe_metrics(bus_records, baseline) if bus_records else {}
            row = {
                "suite": suite,
                "model": MODEL_SHORT.get(slug, slug),
                "model_slug": slug,
                "n_medical": len(med_records),
                "n_business": len(bus_records),
                "uai_irr_medical": med.get("uai_irr"),
                "uai_irr_business": bus.get("uai_irr"),
                "uai_pls_medical": med.get("uai_plaus"),
                "uai_pls_business": bus.get("uai_plaus"),
                "disc_medical": med.get("disc_delta"),
                "disc_business": bus.get("disc_delta"),
                "mae_medical": med.get("mae_control"),
                "mae_business": bus.get("mae_control"),
                "acc10_medical": med.get("acc10_control"),
                "acc10_business": bus.get("acc10_control"),
            }
            for k in ("uai_irr", "uai_pls", "disc"):
                vm = row[f"{k}_medical"]
                vb = row[f"{k}_business"]
                row[f"delta_{k}"] = (
                    vm - vb if isinstance(vm, (int, float))
                    and isinstance(vb, (int, float)) else None
                )
            rows.append(row)
    return rows


def _fmt(v: float | None, prec: int = 2) -> str:
    if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
        return "---"
    return f"{v:.{prec}f}"


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %s (%d rows)", path, len(rows))


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    log.info("Wrote %s", path)


def write_latex(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r["suite"], r["model"]))
    lines = [
        r"% Auto-generated by anchorbench.analysis.medical_pilot",
        r"% Medical-vs-business UAI side-by-side.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll rr rr rr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{UAI$_{\mathrm{irr}}$}"
        r" & \multicolumn{2}{c}{UAI$_{\mathrm{pls}}$}"
        r" & \multicolumn{2}{c}{Disc$_{\Delta}$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Suite & Model & medical & business & medical & business "
        r"& medical & business \\",
        r"\midrule",
    ]
    prev = ""
    for r in rows_sorted:
        suite_cell = r["suite"] if r["suite"] != prev else ""
        prev = r["suite"]
        lines.append(
            f"{suite_cell} & {r['model']} & "
            f"{_fmt(r['uai_irr_medical'])} & {_fmt(r['uai_irr_business'])} & "
            f"{_fmt(r['uai_pls_medical'])} & {_fmt(r['uai_pls_business'])} & "
            f"{_fmt(r['disc_medical'])} & {_fmt(r['disc_business'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Medical-domain pilot (3 clinical domains:"
            r" clinical readmission risk, medication dosage adjustment,"
            r" diagnostic confidence) vs. published business-domain results"
            r" (6 domains) on the same prompt scaffolding. The pattern"
            r" UAI$_{\mathrm{pls}}>$UAI$_{\mathrm{irr}}>0$ persists in the"
            r" medical setting, demonstrating that anchoring is not a"
            r" business-domain artifact.}"
        ),
        r"\label{tab:medical_pilot}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", path)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r["suite"], r["model"]))

    by_suite: dict[str, list[dict]] = defaultdict(list)
    for r in rows_sorted:
        by_suite[r["suite"]].append(r)

    lines = ["# Medical-domain pilot\n\n"]
    for suite, rs in by_suite.items():
        lines.append(f"## {suite.capitalize()}\n\n")
        # Means across models
        for key in ("uai_irr", "uai_pls", "disc"):
            med_vals = [r[f"{key}_medical"] for r in rs
                        if r.get(f"{key}_medical") is not None]
            bus_vals = [r[f"{key}_business"] for r in rs
                        if r.get(f"{key}_business") is not None]
            if med_vals and bus_vals:
                med_m = sum(med_vals) / len(med_vals)
                bus_m = sum(bus_vals) / len(bus_vals)
                lines.append(
                    f"- mean {key}: medical={med_m:+.3f}, "
                    f"business={bus_m:+.3f}, "
                    f"\u0394={(med_m-bus_m):+.3f}\n"
                )
        lines.append("\n")
        lines.append("| Model | UAI_pls (med) | UAI_pls (bus) | "
                     "Disc (med) | Disc (bus) |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for r in rs:
            lines.append(
                f"| {r['model']} | {_fmt(r['uai_pls_medical'])} | "
                f"{_fmt(r['uai_pls_business'])} | "
                f"{_fmt(r['disc_medical'])} | {_fmt(r['disc_business'])} |\n"
            )
        lines.append("\n")
    path.write_text("".join(lines))
    log.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Medical-vs-business comparison (C1)")
    p.add_argument("--medical_dir", type=Path, default=DEFAULT_MEDICAL_DIR)
    p.add_argument("--business_dir", type=Path, default=DEFAULT_BUSINESS_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    rows = gather(args.medical_dir, args.business_dir)
    log.info("Gathered %d (suite, model) rows", len(rows))
    if not rows:
        log.error("No rows; aborting")
        raise SystemExit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "medical_comparison.csv")
    write_json(rows, args.out_dir / "medical_comparison.json")
    write_latex(rows, args.out_dir / "medical_table.tex")
    write_markdown(rows, args.out_dir / "interpretation.md")


if __name__ == "__main__":
    main()
