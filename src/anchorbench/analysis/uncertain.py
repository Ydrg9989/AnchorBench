"""Uncertain-judgment analysis.

Reads ``results/rebuttal/uncertain/<slug>/results.jsonl`` for every model
and per-k summary, then computes a *Bayesian-rational shift baseline*:

    expected_rational_shift(k, w) = w * (anchor - visible_mean_k) / (k + w)

where ``visible_mean_k`` is the mean of the k visible ratings and ``w`` is
the implied weight of the anchor as an additional evidence point. For an
ideal Bayesian updater treating the anchor as one extra rating of equal
credibility (``w = 1``), the rational UAI ceiling shrinks as k grows
(more visible evidence -> anchor counts for less).

Outputs:
  uncertain.csv / .json
  uncertain_table.tex
  uncertain.pdf / .png  (UAI vs k, with rational w=1 baseline overlay)
  interpretation.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_IN = Path("results/rebuttal/uncertain")
DEFAULT_OUT = Path("results/rebuttal/uncertain")
DEFAULT_CORE = Path("datasets/anchorbench_external_core")

K_LEVELS = (1, 2, 3)


def _load_itemspecs(core_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    specs_path = core_dir / "itemspecs.jsonl"
    if not specs_path.exists():
        return out
    with open(specs_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("suite") != "external":
                continue
            out[d["item_id"]] = d
    return out


def _visible_mean(spec: dict, k: int) -> float | None:
    from anchorbench.data.suites.external_uncertain import visible_mean
    return visible_mean(spec, k)


def _rational_ceiling(k: int, w: float = 1.0) -> float:
    """For UAI normalised by |anchor - control|, the rational Bayesian
    shift toward the anchor when treating it as w additional ratings of
    equal credibility is w / (k + w). Here ``control`` corresponds to the
    visible-mean baseline."""
    return w / (k + w)


def gather(in_dir: Path, core_dir: Path) -> dict:
    from anchorbench.eval.constants import MODEL_SHORT
    specs = _load_itemspecs(core_dir)

    rows: list[dict] = []
    for model_dir in sorted(in_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        rj = model_dir / "results.jsonl"
        if not rj.exists():
            continue
        records = [json.loads(line) for line in open(rj) if line.strip()]
        # Group by item -> condition
        by_ic: dict[tuple[str, str], dict] = {}
        for r in records:
            if not r.get("parsed_ok"):
                continue
            by_ic[(r["item_id"], r["condition"])] = r

        # Per-k: compute (acc10_control, mean UAI_irr, mean UAI_pls).
        per_k: dict[int, dict] = {}
        for k in K_LEVELS:
            control_cond = f"uncertain_p{k}_control"
            uai_pls: list[float] = []
            uai_irr: list[float] = []
            acc10: list[int] = []
            n_items_seen = 0

            for iid, spec in specs.items():
                ctrl_rec = by_ic.get((iid, control_cond))
                if ctrl_rec is None:
                    continue
                ctrl_int = ctrl_rec.get("answer_int")
                if ctrl_int is None:
                    continue
                # Accuracy: |ctrl_int - y_star| <= 10 (where y_star = full-5 mean).
                y_star = spec.get("y_star_evidence", spec.get("y_star"))
                if y_star is not None:
                    acc10.append(1 if abs(ctrl_int - y_star) <= 10 else 0)
                n_items_seen += 1

                for cond_suffix, bucket in (
                    ("plausible_low", uai_pls), ("plausible_high", uai_pls),
                    ("irrelevant_low", uai_irr), ("irrelevant_high", uai_irr),
                ):
                    rec = by_ic.get((iid, f"uncertain_p{k}_{cond_suffix}"))
                    if rec is None:
                        continue
                    ai = rec.get("answer_int")
                    anchor = rec.get("anchor_value")
                    if ai is None or anchor is None:
                        continue
                    denom = abs(anchor - ctrl_int)
                    if denom < 1e-6:
                        continue
                    uai = (float(ai) - ctrl_int) / denom * (
                        1 if anchor > ctrl_int else -1
                    )
                    bucket.append(uai)

            per_k[k] = {
                "k": k,
                "n_items": n_items_seen,
                "acc10": (sum(acc10) / len(acc10)) if acc10 else None,
                "uai_pls": (sum(uai_pls) / len(uai_pls)) if uai_pls else None,
                "uai_irr": (sum(uai_irr) / len(uai_irr)) if uai_irr else None,
                "rational_ceiling_w1": _rational_ceiling(k, 1.0),
            }

        rows.append({
            "model": MODEL_SHORT.get(model_dir.name, model_dir.name),
            "model_slug": model_dir.name,
            "per_k": per_k,
        })

    return {
        "rows": rows,
        "rational_baseline": {
            str(k): _rational_ceiling(k, 1.0) for k in K_LEVELS
        },
    }


def _fmt(v: float | None) -> str:
    if v is None:
        return "---"
    s = f"{v:+.2f}"
    if s.startswith("-"):
        return f"$-${s[1:]}"
    if s.startswith("+"):
        return s[1:]
    return s


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "---"
    return f"{round(v * 100):d}\\%"


def write_csv(out: dict, path: Path) -> None:
    import csv
    flat = []
    for r in out["rows"]:
        for k in K_LEVELS:
            d = r["per_k"].get(k, {})
            flat.append({
                "model": r["model"], "model_slug": r["model_slug"],
                "k": k, "acc10": d.get("acc10"),
                "uai_pls": d.get("uai_pls"), "uai_irr": d.get("uai_irr"),
                "rational_ceiling_w1": d.get("rational_ceiling_w1"),
                "n_items": d.get("n_items"),
            })
    if not flat:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)


def write_json(out: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


def write_latex(out: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = out["rows"]
    lines = [
        r"% Auto-generated by anchorbench.analysis.uncertain",
        r"% Uncertain judgment: UAI vs visible-evidence count k.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l rrr rrr rrr}",
        r"\toprule",
        r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{Acc$_{10}$} "
        r"& \multicolumn{3}{c}{UAI$_{\mathrm{pls}}$} "
        r"& \multicolumn{3}{c}{UAI$_{\mathrm{irr}}$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r" & k=1 & k=2 & k=3 & k=1 & k=2 & k=3 & k=1 & k=2 & k=3 \\",
        r"\midrule",
    ]
    for r in sorted(rows, key=lambda r: r["model"]):
        cells = [r["model"]]
        for metric, fmt in (("acc10", _fmt_pct),
                            ("uai_pls", _fmt), ("uai_irr", _fmt)):
            for k in K_LEVELS:
                cells.append(fmt(r["per_k"].get(k, {}).get(metric)))
        lines.append("  " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    # Rational ceiling row
    rb = out["rational_baseline"]
    rb_cells = [r"Rational ceiling ($w{=}1$)", "---", "---", "---"]
    for k in K_LEVELS:
        rb_cells.append(_fmt(rb[str(k)]))
    rb_cells += ["---", "---", "---"]
    lines.append("  " + " & ".join(rb_cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Uncertain-judgment results. Only k of 5 ratings are"
         r" shown to the model; the gold answer remains the full-5 mean."
         r" An ideal Bayesian updater treating the plausible anchor as one"
         r" additional rating of equal credibility ($w{=}1$) would produce"
         r" UAI$_{\mathrm{pls}} = w / (k + w)$ (Rational ceiling row)."
         r" Measured UAI$_{\mathrm{pls}}$ above the ceiling indicates"
         r" classical (super-rational) anchoring; below indicates the model"
         r" weights the anchor less than one full additional evidence"
         r" point. UAI$_{\mathrm{irr}}$ should stay near zero for capable"
         r" models at every $k$.}"),
        r"\label{tab:uncertain_k}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_figure(out: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log.warning("matplotlib unavailable; skipping figure")
        return
    rows = out["rows"]
    rb = out["rational_baseline"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = list(K_LEVELS)
    for r in sorted(rows, key=lambda r: r["model"]):
        y = [r["per_k"].get(k, {}).get("uai_pls") for k in K_LEVELS]
        if all(v is None for v in y):
            continue
        ax.plot(x, y, marker="o", linewidth=1.3, markersize=4,
                label=r["model"])
    yb = [rb[str(k)] for k in K_LEVELS]
    ax.plot(x, yb, marker="x", linestyle="--", color="black",
            linewidth=1.0, label="Rational ceiling ($w{=}1$)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in K_LEVELS])
    ax.set_xlabel("Visible-evidence count")
    ax.set_ylabel("UAI$_{plausible}$ (mean)")
    ax.set_title("Anchoring under epistemic uncertainty")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_markdown(out: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = out["rows"]
    rb = out["rational_baseline"]

    lines = [
        "# Uncertain-judgment\n\n",
        "Tests whether anchoring requires genuine "
        "judgment under uncertainty; the published task is arithmetic "
        "mean of visible numbers). We render External items with only "
        "k of 5 ratings visible, but score against the original full-5 "
        "mean. Now there is true epistemic uncertainty, and we can "
        "compare measured anchoring against the rational Bayesian "
        "ceiling.\n\n",
        "## Rational Bayesian baseline (anchor weighted as 1 extra rating)\n\n",
    ]
    for k in K_LEVELS:
        lines.append(f"- k={k}: rational UAI_pls ceiling = {rb[str(k)]:+.3f}\n")
    lines.append("\n## Per-model curves (mean UAI_pls)\n\n")
    lines.append("| Model | k=1 | k=2 | k=3 |\n|---|---:|---:|---:|\n")
    for r in sorted(rows, key=lambda r: r["model"]):
        per_k = r["per_k"]
        lines.append(
            f"| {r['model']} | "
            f"{_fmt(per_k.get(1, {}).get('uai_pls'))} | "
            f"{_fmt(per_k.get(2, {}).get('uai_pls'))} | "
            f"{_fmt(per_k.get(3, {}).get('uai_pls'))} |\n"
        )
    lines.append(
        "\n## Interpretation\n\n"
        "- **Measured UAI_pls > rational ceiling at the same k**: "
        "the model exceeds what a Bayesian updater treating the anchor "
        "as a single extra rating would predict — clear classical "
        "anchoring bias on top of legitimate updating.\n"
        "- **Measured UAI_pls ≈ rational ceiling**: behaviour is "
        "indistinguishable from rational Bayesian updating; the "
        "published anchoring effect on standard External (k=5 visible) "
        "would then be defensible as evidence integration rather than "
        "bias.\n"
        "- **Decreasing UAI_pls with k**: the model treats the anchor "
        "as one piece of evidence among many — exactly the rational "
        "behaviour. A flat curve indicates the anchor has fixed "
        "salience regardless of how much evidence is visible.\n"
        "- **UAI_irr near zero at every k**: irrelevant anchors should "
        "stay ignored regardless of uncertainty.\n"
    )
    path.write_text("".join(lines))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Uncertain-judgment analysis")
    p.add_argument("--in_dir", type=Path, default=DEFAULT_IN)
    p.add_argument("--core_dir", type=Path, default=DEFAULT_CORE)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    out = gather(args.in_dir, args.core_dir)
    log.info("Gathered %d models", len(out["rows"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out, args.out_dir / "uncertain.csv")
    write_json(out, args.out_dir / "uncertain.json")
    write_latex(out, args.out_dir / "uncertain_table.tex")
    write_figure(out, args.out_dir / "uncertain.png")
    write_markdown(out, args.out_dir / "interpretation.md")


if __name__ == "__main__":
    main()
