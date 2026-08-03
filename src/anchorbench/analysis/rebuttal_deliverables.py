"""Collect every rebuttal artifact (tables + figures + interpretation notes)
into a single handoff directory for the writing track.

Produces ``COLM/review/rebuttal_artifacts/`` populated with:

  tables/
    tab_main_results_revised.tex            (A2)
    tab_uai_pathway.tex                     (existing)
    implied_weight_table.tex                (A1)
    excess_uai_table.tex                    (A1)
    cot_vs_baseline.tex                     (A3)
    cot_extended_table.tex                  (B1, after runs)
    spectrum_table.tex                      (B2, after runs)
    wmean_table.tex                         (B3, after runs)
    medical_table.tex                       (C1, after runs)
  figures/
    spectrum.pdf                            (B2)
  notes/
    00_day1_lockdown.md
    bayesian_bound_interpretation.md
    cot_existing_interpretation.md
    cot_extended_interpretation.md
    spectrum_interpretation.md
    wmean_interpretation.md
    medical_interpretation.md
  INDEX.md                                 (top-level inventory)
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("COLM/review/rebuttal_artifacts")

# (src_path, dst_subdir/filename). All paths are relative to repo root.
ARTIFACTS = [
    # --- A1 / Cluster A: Bayesian bound -----------------------------------
    ("results/rebuttal/bayesian_bound/implied_weight_table.tex",
     "tables/implied_weight_table.tex"),
    ("results/rebuttal/bayesian_bound/excess_uai_table.tex",
     "tables/excess_uai_table.tex"),
    ("results/rebuttal/bayesian_bound/interpretation.md",
     "notes/bayesian_bound_interpretation.md"),

    # --- A2 / Cluster C: revised Table 1 ----------------------------------
    ("outputs/tables/tab_main_results_revised.tex",
     "tables/tab_main_results_revised.tex"),
    ("outputs/tables/tab_uai_pathway.tex",
     "tables/tab_uai_pathway.tex"),

    # --- A3 / Cluster F: existing CoT meta-analysis -----------------------
    ("results/rebuttal/cot_reasoning/cot_vs_baseline.tex",
     "tables/cot_vs_baseline_existing.tex"),
    ("results/rebuttal/cot_reasoning/interpretation.md",
     "notes/cot_existing_interpretation.md"),

    # --- B1 / Cluster F: CoT extension (after runs) -----------------------
    ("results/rebuttal/cot_reasoning_extended/cot_vs_baseline.tex",
     "tables/cot_vs_baseline_extended.tex"),
    ("results/rebuttal/cot_reasoning_extended/interpretation.md",
     "notes/cot_extended_interpretation.md"),

    # --- B2 / Cluster H: plausibility spectrum (after runs) ---------------
    ("results/rebuttal/spectrum/spectrum_table.tex",
     "tables/spectrum_table.tex"),
    ("results/rebuttal/spectrum/spectrum.pdf",
     "figures/spectrum.pdf"),
    ("results/rebuttal/spectrum/spectrum.png",
     "figures/spectrum.png"),
    ("results/rebuttal/spectrum/interpretation.md",
     "notes/spectrum_interpretation.md"),

    # --- B3 / Cluster J: weighted-mean (after runs) -----------------------
    ("results/rebuttal/weighted_mean/wmean_table.tex",
     "tables/wmean_table.tex"),
    ("results/rebuttal/weighted_mean/interpretation.md",
     "notes/wmean_interpretation.md"),

    # --- C1 / Cluster B: medical pilot (after runs) -----------------------
    ("results/rebuttal/medical/medical_table.tex",
     "tables/medical_table.tex"),
    ("results/rebuttal/medical/interpretation.md",
     "notes/medical_interpretation.md"),

    # --- D1 / Cluster H: plausibility-intensity probe ---------------------
    ("results/rebuttal/intensity/intensity_table.tex",
     "tables/intensity_table.tex"),
    ("results/rebuttal/intensity/intensity.pdf",
     "figures/intensity.pdf"),
    ("results/rebuttal/intensity/intensity.png",
     "figures/intensity.png"),
    ("results/rebuttal/intensity/interpretation.md",
     "notes/intensity_interpretation.md"),

    # --- Extended LARGE-model panel (3 frontier API + 2 70B OW) -----------
    ("results/rebuttal/large_panel/large_panel_table.tex",
     "tables/large_panel_table.tex"),
    ("results/rebuttal/large_panel/interpretation.md",
     "notes/large_panel_interpretation.md"),
    ("results/rebuttal/large_api/cost_estimate.json",
     "notes/large_api_actual_cost.json"),

    # --- P1 Cross-pathway intensity --------------------------------------
    ("results/rebuttal/intensity_pathway/intensity_pathway_table.tex",
     "tables/intensity_pathway_table.tex"),
    ("results/rebuttal/intensity_pathway/intensity_pathway.pdf",
     "figures/intensity_pathway.pdf"),
    ("results/rebuttal/intensity_pathway/intensity_pathway.png",
     "figures/intensity_pathway.png"),
    ("results/rebuttal/intensity_pathway/interpretation.md",
     "notes/intensity_pathway_interpretation.md"),

    # --- P2 RAG realism --------------------------------------------------
    ("results/rebuttal/rag_realism/rag_realism_table.tex",
     "tables/rag_realism_table.tex"),
    ("results/rebuttal/rag_realism/rag_realism.pdf",
     "figures/rag_realism.pdf"),
    ("results/rebuttal/rag_realism/rag_realism.png",
     "figures/rag_realism.png"),
    ("results/rebuttal/rag_realism/interpretation.md",
     "notes/rag_realism_interpretation.md"),

    # --- P3 Tool realism -------------------------------------------------
    ("results/rebuttal/tool_realism/tool_realism_table.tex",
     "tables/tool_realism_table.tex"),
    ("results/rebuttal/tool_realism/interpretation.md",
     "notes/tool_realism_interpretation.md"),

    # --- P4 Task-specification ablation ---------------------------------
    ("results/rebuttal/task_spec/task_spec_table.tex",
     "tables/task_spec_table.tex"),
    ("results/rebuttal/task_spec/task_spec.pdf",
     "figures/task_spec.pdf"),
    ("results/rebuttal/task_spec/task_spec.png",
     "figures/task_spec.png"),
    ("results/rebuttal/task_spec/interpretation.md",
     "notes/task_spec_interpretation.md"),

    # --- P5 Uncertain-judgment ------------------------------------------
    ("results/rebuttal/uncertain/uncertain_table.tex",
     "tables/uncertain_table.tex"),
    ("results/rebuttal/uncertain/uncertain.pdf",
     "figures/uncertain.pdf"),
    ("results/rebuttal/uncertain/uncertain.png",
     "figures/uncertain.png"),
    ("results/rebuttal/uncertain/interpretation.md",
     "notes/uncertain_interpretation.md"),

    # --- REVIEWER-2 Extension-domain pilot (law + consumer) -----------------
    ("results/rebuttal/extension_pilot/extension_pilot_table.tex",
     "tables/extension_pilot_table.tex"),
    ("results/rebuttal/extension_pilot/interpretation.md",
     "notes/extension_pilot_interpretation.md"),

    # --- REVIEWER-4 Cohen's d translation -----------------------------------
    ("results/rebuttal/cohens_d/cohens_d_table.tex",
     "tables/cohens_d_table.tex"),
    ("results/rebuttal/cohens_d/interpretation.md",
     "notes/cohens_d_interpretation.md"),

    # --- REVIEWER-5 Per-item case studies -----------------------------------
    ("results/rebuttal/case_studies/case_studies_table.tex",
     "tables/case_studies_table.tex"),
    ("results/rebuttal/case_studies/case_studies.md",
     "notes/case_studies.md"),

    # --- Reviewer reply documents ---------------------------------------
    ("COLM/review/rebuttal_notes/11_pointbypoint_replies.md",
     "notes/11_pointbypoint_replies.md"),

    # --- Day 1 lockdown notes ---------------------------------------------
    ("COLM/review/rebuttal_notes/00_day1_lockdown.md",
     "notes/00_day1_lockdown.md"),
]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Aggregate rebuttal deliverables")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--strict", action="store_true",
                   help="Fail if any expected artifact is missing")
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "notes").mkdir(exist_ok=True)

    copied: list[tuple[str, str]] = []
    missing: list[str] = []
    for src_rel, dst_rel in ARTIFACTS:
        src = repo_root / src_rel
        dst = out_dir / dst_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append((src_rel, dst_rel))

    # Build INDEX.md
    idx = ["# AnchorBench COLM 2026 Rebuttal — Artifact Index\n\n"]
    idx.append("Generated by `anchorbench.analysis.rebuttal_deliverables`.\n\n")
    idx.append("## Tables\n\n")
    for src, dst in copied:
        if dst.startswith("tables/"):
            idx.append(f"- `{dst}` (from `{src}`)\n")
    idx.append("\n## Figures\n\n")
    for src, dst in copied:
        if dst.startswith("figures/"):
            idx.append(f"- `{dst}` (from `{src}`)\n")
    idx.append("\n## Notes\n\n")
    for src, dst in copied:
        if dst.startswith("notes/"):
            idx.append(f"- `{dst}` (from `{src}`)\n")
    if missing:
        idx.append("\n## Missing artifacts (not yet generated)\n\n")
        for m in missing:
            idx.append(f"- `{m}`\n")
    idx.append("\n## Cluster -> artifact map\n\n")
    idx.append("""\
- Cluster A (Bayesian updating): `tables/implied_weight_table.tex`,
  `tables/excess_uai_table.tex`, `notes/bayesian_bound_interpretation.md`
- Cluster B (Task realism): `tables/medical_table.tex`,
  `notes/medical_interpretation.md`
- Cluster C (Metric presentation): `tables/tab_main_results_revised.tex`
- Cluster F (Reasoning-allowed): `tables/cot_vs_baseline_existing.tex`,
  `tables/cot_vs_baseline_extended.tex`, `notes/cot_*_interpretation.md`
- Cluster H (Binary relevance axis): `tables/spectrum_table.tex`,
  `figures/spectrum.pdf`, `notes/spectrum_interpretation.md`;
  plus intensity probe at `tables/intensity_table.tex`,
  `figures/intensity.pdf`, `notes/intensity_interpretation.md`
- Cluster J (Mean gold standard): `tables/wmean_table.tex`,
  `notes/wmean_interpretation.md`
""")
    (out_dir / "INDEX.md").write_text("".join(idx))

    print(f"\nCopied {len(copied)}/{len(ARTIFACTS)} artifacts to {out_dir}")
    if missing:
        print(f"Missing ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
