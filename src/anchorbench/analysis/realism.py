"""RAG and Tool realism tables (Appendix tab:rag_realism, tab:tool_realism).

Reads ``results/rebuttal/<suite>_realism/<slug>/realism_curve.json`` for
every model (written by ``runners.realism``) and emits, per suite:
  <suite>_realism.csv / .json
  <suite>_realism_table.tex
  interpretation.md
  rag_realism.pdf / .png (RAG only: rank-degradation bar chart)

    python -m anchorbench.analysis.realism --suite rag
    python -m anchorbench.analysis.realism --suite tool
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from anchorbench.analysis._io import fmt_latex, mean_or_none, write_csv, write_json
from anchorbench.paths import RESULTS_DIR

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Suite:
    tags: tuple[str, ...]            # "baseline" (the published run) first
    tag_label: dict[str, str]
    column_headers: str              # the LaTeX header row after "Model"
    caption: str
    header_comment: str
    markdown: tuple[str, ...]        # title, intro, table header, interpretation


SUITES: dict[str, Suite] = {
    "rag": Suite(
        tags=("baseline", "rank1", "rank5", "rank5_distract"),
        tag_label={
            "baseline":         "Baseline (rank 2)",
            "rank1":            "Rank 1",
            "rank5":            "Rank 5 + 2 distract.",
            "rank5_distract":   "Rank 5 + 2 distract. + rel scores",
        },
        column_headers=r" & Base & R1 & R5+D & R5+D+Sc & Base & R1 & R5+D & R5+D+Sc \\",
        caption=(r"\caption{RAG realism ablation. Base = anchored doc at position 2"
                 r" (published RAG layout); R1 = anchor at top of retrieval (rank 1);"
                 r" R5+D = anchor at bottom of a 5-doc list with 2 plausible"
                 r" distractor documents adjacent to the anchor; R5+D+Sc adds"
                 r" synthetic per-doc relevance scores. A drop from R1 to R5+D"
                 r" indicates models down-weight low-ranked retrieved"
                 r" anchors.}"),
        header_comment=r"% RAG realism ablation: anchor-document rank x distractors.",
        markdown=(
            "# RAG realism ablation\n\n",
            "The RAG suite varies document "
            "rank (1 vs 5), adds plausible distractors, and optionally "
            "exposes synthetic relevance scores.\n\n",
            "| Relevance | Base (rank 2) | Rank 1 | Rank 5 + 2 distract. "
            "| Rank 5 + distract. + scores |\n"
            "|---|---:|---:|---:|---:|\n",
            "\n## Interpretation\n\n"
            "If UAI is higher at Rank 1 than Rank 5, models privilege "
            "top-ranked retrieved documents (a property of realistic "
            "RAG systems) and the published RAG number is an "
            "intermediate estimate. If UAI drops when distractors are "
            "added, the anchored document's salience is reduced by "
            "surrounding context — meaning real-world RAG pipelines with "
            "diverse retrieval would observe weaker anchoring than the "
            "controlled benchmark. The R5+D+Sc column tests whether "
            "explicit relevance scores let models down-weight the "
            "anchor when its retrieval score is mid-pack.\n",
        ),
    ),
    "tool": Suite(
        tags=("baseline", "elicited", "noisy"),
        tag_label={
            "baseline":  "Baseline (externally injected)",
            "elicited":  "Model-elicited call",
            "noisy":     "Noisy tool envelope",
        },
        column_headers=r" & Base & Elicited & Noisy & Base & Elicited & Noisy \\",
        caption=(r"\caption{Tool realism ablation. Base reproduces the published"
                 r" Tool suite (externally injected tool output); Elicited prepends a"
                 r" synthetic model-planned tool-call turn (provenance signal);"
                 r" Noisy wraps the anchor value in a realistic JSON envelope"
                 r" containing additional metadata fields. Closing the gap between"
                 r" Base and Elicited indicates the anchor effect is not driven by"
                 r" the externally-injected framing; reductions under Noisy"
                 r" indicate the anchor's salience competes with surrounding"
                 r" metadata.}"),
        header_comment=(r"% Tool realism: model-elicited vs externally injected tool calls;"
                        r" plus noisy tool envelope."),
        markdown=(
            "# Tool realism ablation\n\n",
            "Varies two tool-realism axes:\n\n"
            "- **Elicited**: a synthetic assistant turn announces the lookup "
            "before the tool call, framing the response as model-initiated.\n"
            "- **Noisy**: the tool response wraps the anchor value in a "
            "realistic envelope with extra metadata fields (`confidence`, "
            "`freshness_days`, `sample_size_n`, `request_timestamp_ms`, "
            "etc.) to test salience-of-number.\n\n",
            "| Relevance | Baseline | Elicited | Noisy |\n"
            "|---|---:|---:|---:|\n",
            "\n## Interpretation\n\n"
            "- If `Elicited` ≈ `Baseline`, the published Tool number is "
            "robust to the provenance signal (and the controlled "
            "externally-injected setup is a fair test).\n"
            "- If `Noisy` < `Baseline` plausibly, the published Tool "
            "number slightly overstates the in-practice anchoring effect "
            "because realistic tool responses dilute the anchor's "
            "salience.\n"
            "- If `Noisy` ≈ `Baseline`, anchoring is driven by the "
            "numeric value regardless of surrounding metadata — "
            "the bias is real even in production-style tool envelopes.\n",
        ),
    ),
}

RELEVANCES = ("plausible", "irrelevant")


def gather(in_dir: Path, tags: tuple[str, ...]) -> list[dict]:
    from anchorbench.eval.constants import MODEL_SHORT
    rows: list[dict] = []
    for model_dir in sorted(in_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        curve_path = model_dir / "realism_curve.json"
        if not curve_path.exists():
            continue
        curve = json.loads(curve_path.read_text())
        row = {
            "model": MODEL_SHORT.get(model_dir.name, model_dir.name),
            "model_slug": model_dir.name,
        }
        for rel in RELEVANCES:
            for tag in tags:
                row[f"{rel}_{tag}"] = curve.get(f"{rel}__{tag}")
        rows.append(row)
    return rows


def write_latex(rows: list[dict], path: Path, suite: str, cfg: Suite) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cfg.tags)
    lines = [
        f"% Auto-generated by anchorbench.analysis.realism --suite {suite}",
        cfg.header_comment,
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l " + " ".join(["r" * n] * 2) + "}",
        r"\toprule",
        r"\multirow{2}{*}{Model} & \multicolumn{" + str(n) + r"}{c}{Plausible (UAI)} "
        r"& \multicolumn{" + str(n) + r"}{c}{Irrelevant (UAI)} \\",
        r"\cmidrule(lr){2-" + str(n + 1) + r"}\cmidrule(lr){" + str(n + 2) + "-" + str(2 * n + 1) + "}",
        cfg.column_headers,
        r"\midrule",
    ]
    for r in sorted(rows, key=lambda r: r["model"]):
        cells = [r["model"]] + [fmt_latex(r.get(f"{rel}_{tag}")) for rel in RELEVANCES for tag in cfg.tags]
        lines.append("  " + " & ".join(cells) + r" \\")
    mean_cells = ["Mean"] + [
        f"\\textbf{{{fmt_latex(mean_or_none(r.get(f'{rel}_{tag}') for r in rows))}}}"
        for rel in RELEVANCES for tag in cfg.tags
    ]
    lines.append(r"\midrule")
    lines.append("  " + " & ".join(mean_cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        cfg.caption,
        f"\\label{{tab:{suite}_realism}}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_figure(rows: list[dict], path: Path, cfg: Suite) -> None:
    """Rank-degradation bar chart of the plausible UAI (RAG only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(cfg.tags))
    width = 0.7 / max(1, len(rows))
    for i, r in enumerate(sorted(rows, key=lambda r: r["model"])):
        y = [r.get(f"plausible_{t}") for t in cfg.tags]
        y = [v if v is not None else 0.0 for v in y]
        ax.bar(x + (i - len(rows) / 2 + 0.5) * width, y, width,
               label=r["model"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg.tag_label[t] for t in cfg.tags], rotation=15,
                       ha="right", fontsize=8)
    ax.set_ylabel("Plausible-anchor UAI (mean)")
    ax.set_title("RAG realism: anchor rank + distractors (P2)")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_markdown(rows: list[dict], path: Path, cfg: Suite) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title, intro, table_header, interpretation = cfg.markdown
    lines = [title, intro, "## Mean UAI across the panel\n\n", table_header]
    for rel in RELEVANCES:
        cells = [fmt_latex(mean_or_none(r.get(f"{rel}_{tag}") for r in rows)) for tag in cfg.tags]
        lines.append(f"| {rel.capitalize()} | " + " | ".join(cells) + " |\n")
    lines.append(interpretation)
    path.write_text("".join(lines))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="RAG / Tool realism analysis")
    p.add_argument("--suite", choices=sorted(SUITES), required=True)
    p.add_argument("--in_dir", type=Path, default=None,
                   help="default: results/rebuttal/<suite>_realism")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="default: same as --in_dir")
    args = p.parse_args(argv)
    suite, cfg = args.suite, SUITES[args.suite]
    in_dir = args.in_dir or RESULTS_DIR / f"rebuttal/{suite}_realism"
    out_dir = args.out_dir or in_dir

    rows = gather(in_dir, cfg.tags)
    if not rows:
        raise SystemExit(f"no realism_curve.json under {in_dir}")
    log.info("Gathered %d rows", len(rows))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / f"{suite}_realism.csv")
    write_json(rows, out_dir / f"{suite}_realism.json")
    write_latex(rows, out_dir / f"{suite}_realism_table.tex", suite, cfg)
    if suite == "rag":
        write_figure(rows, out_dir / "rag_realism.png", cfg)
    write_markdown(rows, out_dir / "interpretation.md", cfg)


if __name__ == "__main__":
    main()
