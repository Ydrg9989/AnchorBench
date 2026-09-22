"""Per-item case studies.

Qualitative case-study evidence complementing
the aggregate metrics. We mechanically extract paired triples (control,
plausible_high, irrelevant_high; and optionally plausible_low,
irrelevant_low) from the published External-suite results.jsonl,
together with the corresponding evidence summary and anchor value, so
that a reader can inspect the model's actual response under each
condition for the same underlying item.

For each model in the standard 4-OW panel we pick the K items with the
largest plausible-anchor shift (|y_plaus_high - y_ctrl|) that also
*discriminate* (plausible-anchor shift > irrelevant-anchor shift), so
the examples illustrate the headline pattern. Selection is deterministic
given the input results files.

Outputs are written to ``results/rebuttal/case_studies/``:

* ``case_studies.csv``: long-form table for spreadsheet inspection
* ``case_studies.json``: structured records
* ``case_studies.md``: appendix-style narrative

This is pure mechanical extraction; no new inference is run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

from anchorbench.analysis._io import load_records, write_csv, write_json

log = logging.getLogger(__name__)

DEFAULT_BUSINESS_DIR = Path("results/full_benchmark")
DEFAULT_DATASET_DIR = Path("datasets/anchorbench_external_core")
DEFAULT_OUT = Path("results/rebuttal/case_studies")

PANEL_SLUGS = [
    "Qwen_Qwen2.5-7B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "google_gemma-3-4b-it",
    "allenai_OLMo-2-1124-13B-Instruct",
]
SUITE = "external"

# How many case studies to surface per model.
TOP_K_PER_MODEL = 3


def _load_promptviews(p: Path) -> dict[tuple[str, str], dict]:
    """Index promptviews by (item_id, condition)."""
    out: dict[tuple[str, str], dict] = {}
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[(d["item_id"], d["condition"])] = d
    return out


def _group_by_item(records: list[dict]) -> dict[str, dict[str, dict]]:
    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_item[r["item_id"]][r["condition"]] = r
    return by_item


def _is_parsed(r: dict | None) -> bool:
    return bool(r and r.get("parsed_ok") and r.get("answer_int") is not None)


def select_case_studies(
    records: list[dict],
    promptviews: dict[tuple[str, str], dict],
    k: int = TOP_K_PER_MODEL,
) -> list[dict]:
    """Pick K discriminative items per model."""
    by_item = _group_by_item(records)
    candidates: list[tuple[float, dict]] = []
    for iid, conds in by_item.items():
        ctrl = conds.get("control")
        p_hi = conds.get("plausible_high")
        i_hi = conds.get("irrelevant_high")
        p_lo = conds.get("plausible_low")
        i_lo = conds.get("irrelevant_low")
        if not all(_is_parsed(r) for r in (ctrl, p_hi, i_hi)):
            continue
        y_c = float(ctrl["answer_int"])
        y_ph = float(p_hi["answer_int"])
        y_ih = float(i_hi["answer_int"])
        shift_p = y_ph - y_c
        shift_i = y_ih - y_c
        # Discriminative: plausible pulls more than irrelevant (in absolute terms)
        discrim = abs(shift_p) - abs(shift_i)
        # Anchor toward direction
        if discrim <= 0:
            continue
        # Score by absolute plausible shift x discriminative margin
        score = abs(shift_p) * (1.0 + discrim / 30.0)
        anchor = p_hi.get("anchor_value")
        pv = promptviews.get((iid, "control"))
        scenario = ""
        evidence = ""
        question = ""
        if pv and "prompt_components" in pv:
            pc = pv["prompt_components"] or {}
            scenario = pc.get("scenario", "")
            evidence = pc.get("evidence", "")
            question = pc.get("question", "")
        rec = {
            "item_id": iid,
            "domain": ctrl.get("domain"),
            "difficulty": ctrl.get("difficulty"),
            "scenario": scenario.strip(),
            "evidence": evidence.strip(),
            "question": question.strip(),
            "anchor_value_high": anchor,
            "anchor_value_low": p_lo.get("anchor_value") if p_lo else None,
            "y_star_evidence": ctrl.get("y_star_evidence"),
            "y_control": int(y_c),
            "y_plausible_high": int(y_ph),
            "y_irrelevant_high": int(y_ih),
            "y_plausible_low": int(p_lo["answer_int"]) if _is_parsed(p_lo) else None,
            "y_irrelevant_low": int(i_lo["answer_int"]) if _is_parsed(i_lo) else None,
            "shift_plausible_high": int(shift_p),
            "shift_irrelevant_high": int(shift_i),
            "discrim_margin": int(round(discrim)),
            "raw_control": (ctrl.get("raw_text") or "")[:200],
            "raw_plausible_high": (p_hi.get("raw_text") or "")[:200],
            "raw_irrelevant_high": (i_hi.get("raw_text") or "")[:200],
        }
        candidates.append((score, rec))
    candidates.sort(key=lambda t: -t[0])
    # Diversify across domains
    chosen: list[dict] = []
    seen_domains: set = set()
    for _, rec in candidates:
        if rec["domain"] in seen_domains and len(chosen) < k:
            continue
        chosen.append(rec)
        seen_domains.add(rec["domain"])
        if len(chosen) >= k:
            break
    # Fill if we hit fewer than k from distinct domains
    if len(chosen) < k:
        for _, rec in candidates:
            if rec in chosen:
                continue
            chosen.append(rec)
            if len(chosen) >= k:
                break
    return chosen


def gather(
    business_dir: Path,
    dataset_dir: Path,
    k_per_model: int = TOP_K_PER_MODEL,
) -> list[dict]:
    from anchorbench.eval.constants import MODEL_SHORT
    promptviews = _load_promptviews(dataset_dir / "promptviews_core.jsonl")
    if not promptviews:
        # Try alternate name (some datasets use 'promptviews.jsonl')
        promptviews = _load_promptviews(dataset_dir / "promptviews.jsonl")
    out: list[dict] = []
    for slug in PANEL_SLUGS:
        res_path = business_dir / SUITE / slug / "results.jsonl"
        records = load_records(res_path)
        if not records:
            log.warning("No records for %s; skipping", slug)
            continue
        cases = select_case_studies(records, promptviews, k=k_per_model)
        for c in cases:
            c["model_slug"] = slug
            c["model"] = MODEL_SHORT.get(slug, slug)
            out.append(c)
    return out


def _fmt(v) -> str:
    if v is None:
        return "---"
    if isinstance(v, float) and not math.isfinite(v):
        return "---"
    return str(v)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    lines = [
        "# Per-item case studies\n\n",
        "Mechanically-extracted paired triples from the published External"
        " suite. For each model in the standard 4-OW panel we surface the"
        " top discriminative items where the plausible-anchor shift"
        " clearly exceeds the irrelevant-anchor shift, so a reader can"
        " inspect the actual model behaviour under each condition for the"
        " same underlying item.\n\n",
        "Notation:\n"
        "- `y_c` = control answer (no anchor)\n"
        "- `y_p^hi` = plausible-high answer\n"
        "- `y_i^hi` = irrelevant-high answer\n"
        "- `a^hi` = plausible-high anchor value\n"
        "- shift columns = answer - control\n\n",
    ]
    for model, recs in by_model.items():
        lines.append(f"## {model}\n\n")
        for i, r in enumerate(recs, 1):
            lines.append(
                f"### Case {i}: `{r['item_id']}` "
                f"(domain: {r['domain']}, difficulty: {r['difficulty']})\n\n"
            )
            lines.append(f"**Scenario.** {r['scenario']}\n\n")
            if r["evidence"]:
                lines.append(
                    "**Evidence.**\n```\n" + r["evidence"] + "\n```\n\n"
                )
            if r["question"]:
                lines.append(f"**Question.** {r['question']}\n\n")
            lines.append(
                "| Condition | Anchor `a` | Answer `y` | Shift y-y_c |\n"
                "|---|---:|---:|---:|\n"
            )
            lines.append(
                f"| control            | --- | {r['y_control']} | 0 |\n"
            )
            lines.append(
                f"| plausible_high     | {_fmt(r['anchor_value_high'])} | "
                f"{r['y_plausible_high']} | "
                f"{r['shift_plausible_high']:+d} |\n"
            )
            lines.append(
                f"| irrelevant_high    | {_fmt(r['anchor_value_high'])} | "
                f"{r['y_irrelevant_high']} | "
                f"{r['shift_irrelevant_high']:+d} |\n"
            )
            if r["y_plausible_low"] is not None:
                lines.append(
                    f"| plausible_low      | {_fmt(r['anchor_value_low'])} | "
                    f"{r['y_plausible_low']} | "
                    f"{r['y_plausible_low'] - r['y_control']:+d} |\n"
                )
            if r["y_irrelevant_low"] is not None:
                lines.append(
                    f"| irrelevant_low     | {_fmt(r['anchor_value_low'])} | "
                    f"{r['y_irrelevant_low']} | "
                    f"{r['y_irrelevant_low'] - r['y_control']:+d} |\n"
                )
            lines.append("\n")
            lines.append(
                f"**Discrimination margin** "
                f"|y_p-y_c| - |y_i-y_c| = "
                f"{r['discrim_margin']:+d}\n\n"
            )
        lines.append("\n")

    path.write_text("".join(lines))
    log.info("Wrote %s", path)


def write_latex(rows: list[dict], path: Path) -> None:
    """Compact LaTeX appendix table: one row per case study."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"% Auto-generated by anchorbench.analysis.case_studies",
        r"% Per-item case studies illustrating that the",
        r"% aggregate UAI pattern is driven by interpretable per-item shifts.",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        r"Model & Item & $a^{hi}$ & $y_c$ & $y_p^{hi}$ & $y_i^{hi}$ & $\Delta_p-\Delta_i$ \\",
        r"\midrule",
    ]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    for model, recs in by_model.items():
        for r in recs:
            iid = r["item_id"].replace("_", r"\_")
            lines.append(
                f"{model} & {iid} & "
                f"{_fmt(r['anchor_value_high'])} & {r['y_control']} & "
                f"{r['y_plausible_high']} & {r['y_irrelevant_high']} & "
                f"{r['discrim_margin']:+d} \\\\"
            )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Per-item case studies (External): for each model"
            r" we surface the top discriminative items where the"
            r" plausible-anchor shift cleanly exceeds the irrelevant-"
            r" anchor shift, illustrating that the aggregate UAI"
            r" pattern is driven by interpretable per-item behaviour"
            r" rather than averaging artifacts."
            r" Full scenarios and per-condition answers are tabulated"
            r" in the supplementary material.}"
        ),
        r"\label{tab:case_studies}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(
        description="Per-item case study extraction"
    )
    p.add_argument("--business_dir", type=Path, default=DEFAULT_BUSINESS_DIR)
    p.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--k_per_model", type=int, default=TOP_K_PER_MODEL)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    rows = gather(args.business_dir, args.dataset_dir, args.k_per_model)
    log.info("Selected %d case studies across %d models",
             len(rows), len({r['model_slug'] for r in rows}))
    if not rows:
        log.error("No case studies selected; aborting")
        raise SystemExit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "case_studies.csv")
    write_json(rows, args.out_dir / "case_studies.json")
    write_markdown(rows, args.out_dir / "case_studies.md")
    write_latex(rows, args.out_dir / "case_studies_table.tex")


if __name__ == "__main__":
    main()
