#!/usr/bin/env python3
"""Verify every numeric claim in the COLM 2026 paper against the
unified_all_suites.json files and the auxiliary CSV summaries.

This is a single, data-driven verifier that consolidates and replaces the
older `verify_paper_tables.py` (v1) and `verify_paper_tables_v2.py` (v2)
scripts. Both now alias to this one.

Each `Claim` is `(suite_or_key, metric, paper_value, tolerance)`. The
script re-reads the canonical JSON outputs and checks values within
tolerance; mismatches are summarized at the end and the script returns
a non-zero exit code if any are found.

Usage:
    python -m anchorbench.paper.verify            # check all claims
    python -m anchorbench.paper.verify --quick    # main + uai-pathway only
    python -m anchorbench.paper.verify --strict   # tighten tolerances
    python -m anchorbench.paper.verify --json OUT # machine-readable result

Claims known to disagree with the data are listed in KNOWN_DIVERGENCES with
the size of the disagreement, and each has a row in docs/RECONCILIATION.md.
They are reported but do not fail the run; a claim that diverges by a
different amount, or that is listed there and no longer diverges, does.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"

SUITE_NAME_MAP = {"ICL": "Icl", "RAG": "Rag"}

MODEL_ORDER = [
    ("Qwen-1.5B", "Qwen_Qwen2.5-1.5B-Instruct"),
    ("Qwen-3B", "Qwen_Qwen2.5-3B-Instruct"),
    ("Qwen-7B", "Qwen_Qwen2.5-7B-Instruct"),
    ("Llama-1B", "meta-llama_Llama-3.2-1B-Instruct"),
    ("Llama-3B", "meta-llama_Llama-3.2-3B-Instruct"),
    ("Llama-8B", "meta-llama_Llama-3.1-8B-Instruct"),
    ("Gemma-1B", "google_gemma-3-1b-it"),
    ("Gemma-4B", "google_gemma-3-4b-it"),
    ("OLMo-13B", "allenai_OLMo-2-1124-13B-Instruct"),
    ("OLMo-32B", "allenai_OLMo-2-0325-32B-Instruct"),
    ("GPT-5.4-mini", "openai_gpt-5.4-mini"),
    ("Claude-Haiku", "anthropic_claude-haiku-4.5"),
    ("Gemini-Flash", "google_gemini-2.5-flash"),
    ("Grok-3-mini", "x-ai_grok-3-mini-beta"),
]
OW_SLUGS = {s for _, s in MODEL_ORDER[:10]}
API_SLUGS = {s for _, s in MODEL_ORDER[10:]}
ALL_SLUGS = OW_SLUGS | API_SLUGS
SUITES = ["External", "History", "ICL", "RAG", "Tool"]


@dataclass
class Mismatch:
    section: str
    label: str
    paper: float
    data: float
    tol: float
    # Listed in KNOWN_DIVERGENCES and failing by the recorded amount, so it is
    # tracked in docs/RECONCILIATION.md rather than being a new problem.
    known: bool = False

    def __str__(self) -> str:
        tag = " [known]" if self.known else ""
        return (f"  [{self.section}] {self.label}: "
                f"paper={self.paper:+.4f} data={self.data:+.4f} "
                f"diff={abs(self.paper - self.data):.4f} (tol {self.tol:.3f}){tag}")


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def find_entry(data, slug, suite):
    mapped = SUITE_NAME_MAP.get(suite, suite)
    for d in data:
        if d["model_slug"] == slug and d["suite"] == mapped:
            return d
    return None


def check(section: str, label: str, paper: float, data: float | None,
          tol: float, mismatches: list[Mismatch]) -> str:
    if data is None:
        mismatches.append(Mismatch(section, label, paper, float("nan"), tol))
        return "MISSING"
    delta = abs(paper - data)
    if delta <= tol:
        return "OK"
    expected = KNOWN_DIVERGENCES.get((section, label))
    known = expected is not None and abs(delta - expected) <= KNOWN_SLACK
    mismatches.append(Mismatch(section, label, paper, data, tol, known=known))
    return "KNOWN" if known else "FAIL"


def header(s: str) -> None:
    print("\n" + "=" * 90)
    print(s)
    print("=" * 90)


# ─── Claim tables (from COLM/sections/findings.tex + appendix.tex) ───

# (slug, suite) → (acc10_control, disc_delta)
PAPER_MAIN: dict[tuple[str, str], tuple[float, float]] = {
    ("Qwen_Qwen2.5-1.5B-Instruct", "External"): (0.342, 0.18),
    ("Qwen_Qwen2.5-1.5B-Instruct", "History"): (0.325, 0.70),
    ("Qwen_Qwen2.5-1.5B-Instruct", "ICL"): (0.594, -0.01),
    ("Qwen_Qwen2.5-1.5B-Instruct", "RAG"): (0.219, 0.21),
    ("Qwen_Qwen2.5-1.5B-Instruct", "Tool"): (0.459, 0.57),
    ("Qwen_Qwen2.5-3B-Instruct", "External"): (0.637, 0.08),
    ("Qwen_Qwen2.5-3B-Instruct", "History"): (0.606, 0.57),
    ("Qwen_Qwen2.5-3B-Instruct", "ICL"): (0.775, 0.00),
    ("Qwen_Qwen2.5-3B-Instruct", "RAG"): (0.741, 0.10),
    ("Qwen_Qwen2.5-3B-Instruct", "Tool"): (0.892, 0.10),
    ("Qwen_Qwen2.5-7B-Instruct", "External"): (0.725, 0.26),
    ("Qwen_Qwen2.5-7B-Instruct", "History"): (0.786, 0.43),
    ("Qwen_Qwen2.5-7B-Instruct", "ICL"): (0.875, 0.01),
    ("Qwen_Qwen2.5-7B-Instruct", "RAG"): (0.639, 0.20),
    ("Qwen_Qwen2.5-7B-Instruct", "Tool"): (0.989, 0.17),
    ("meta-llama_Llama-3.2-1B-Instruct", "External"): (0.472, 0.03),
    ("meta-llama_Llama-3.2-1B-Instruct", "History"): (0.488, -0.35),
    ("meta-llama_Llama-3.2-1B-Instruct", "ICL"): (0.545, -0.09),
    ("meta-llama_Llama-3.2-1B-Instruct", "RAG"): (0.605, -0.01),
    ("meta-llama_Llama-3.2-3B-Instruct", "External"): (0.694, 0.15),
    ("meta-llama_Llama-3.2-3B-Instruct", "History"): (0.722, 0.58),
    ("meta-llama_Llama-3.2-3B-Instruct", "ICL"): (0.808, 0.06),
    ("meta-llama_Llama-3.2-3B-Instruct", "RAG"): (0.715, 0.08),
    ("meta-llama_Llama-3.2-3B-Instruct", "Tool"): (0.945, 0.19),
    ("meta-llama_Llama-3.1-8B-Instruct", "External"): (0.826, 0.30),
    ("meta-llama_Llama-3.1-8B-Instruct", "History"): (0.829, 0.10),
    ("meta-llama_Llama-3.1-8B-Instruct", "ICL"): (0.767, 0.12),
    ("meta-llama_Llama-3.1-8B-Instruct", "RAG"): (0.851, 0.14),
    ("meta-llama_Llama-3.1-8B-Instruct", "Tool"): (0.733, 0.17),
    ("google_gemma-3-1b-it", "External"): (0.368, 0.16),
    ("google_gemma-3-1b-it", "History"): (0.392, 0.31),
    ("google_gemma-3-1b-it", "ICL"): (0.464, -0.02),
    ("google_gemma-3-1b-it", "RAG"): (0.372, 0.04),
    ("google_gemma-3-1b-it", "Tool"): (0.258, 0.02),
    ("google_gemma-3-4b-it", "External"): (0.551, 0.23),
    ("google_gemma-3-4b-it", "History"): (0.562, 0.27),
    ("google_gemma-3-4b-it", "ICL"): (0.386, -0.07),
    ("google_gemma-3-4b-it", "RAG"): (0.475, -0.05),
    ("google_gemma-3-4b-it", "Tool"): (0.628, 0.04),
    ("allenai_OLMo-2-1124-13B-Instruct", "External"): (0.894, 0.17),
    ("allenai_OLMo-2-1124-13B-Instruct", "History"): (0.858, 0.32),
    ("allenai_OLMo-2-1124-13B-Instruct", "ICL"): (0.569, -0.01),
    ("allenai_OLMo-2-1124-13B-Instruct", "RAG"): (0.872, 0.05),
    ("allenai_OLMo-2-1124-13B-Instruct", "Tool"): (0.709, -0.02),
    ("allenai_OLMo-2-0325-32B-Instruct", "External"): (0.528, 0.35),
    ("allenai_OLMo-2-0325-32B-Instruct", "History"): (0.536, 1.05),
    ("allenai_OLMo-2-0325-32B-Instruct", "ICL"): (0.697, 0.02),
    ("allenai_OLMo-2-0325-32B-Instruct", "RAG"): (0.581, 0.44),
    ("allenai_OLMo-2-0325-32B-Instruct", "Tool"): (0.725, 0.01),
    ("openai_gpt-5.4-mini", "External"): (0.989, 0.14),
    ("openai_gpt-5.4-mini", "History"): (0.944, 0.06),
    ("openai_gpt-5.4-mini", "ICL"): (0.994, -0.01),
    ("openai_gpt-5.4-mini", "RAG"): (0.986, 0.08),
    ("openai_gpt-5.4-mini", "Tool"): (0.994, 0.06),
    ("anthropic_claude-haiku-4.5", "External"): (1.000, 0.12),
    ("anthropic_claude-haiku-4.5", "History"): (0.978, -0.04),
    ("anthropic_claude-haiku-4.5", "ICL"): (0.969, -0.01),
    ("anthropic_claude-haiku-4.5", "RAG"): (0.983, 0.05),
    ("anthropic_claude-haiku-4.5", "Tool"): (0.983, 0.02),
    ("google_gemini-2.5-flash", "External"): (0.958, 0.05),
    ("google_gemini-2.5-flash", "History"): (0.947, -0.03),
    ("google_gemini-2.5-flash", "ICL"): (0.994, 0.00),
    ("google_gemini-2.5-flash", "RAG"): (0.842, 0.09),
    ("google_gemini-2.5-flash", "Tool"): (0.983, 0.03),
    ("x-ai_grok-3-mini-beta", "External"): (0.992, 0.16),
    ("x-ai_grok-3-mini-beta", "History"): (0.978, -0.01),
    ("x-ai_grok-3-mini-beta", "ICL"): (0.992, 0.01),
    ("x-ai_grok-3-mini-beta", "RAG"): (0.992, 0.03),
    ("x-ai_grok-3-mini-beta", "Tool"): (0.989, 0.02),
}

# Suite-mean (irr, plaus, disc_delta) over all 14 models (Tool: parseable).
#
# These came from Table 2, which was cut from the paper, and the generator
# for it is gone. The check stays because it is the only guard on the
# absolute UAI levels: Table 1 prints U_irr and U_pls per cell, but
# PAPER_MAIN only carries acc10 and disc_delta, so without this the levels
# are covered solely through their difference.
PAPER_PATHWAY: dict[str, tuple[float, float, float]] = {
    "External": (0.05, 0.22, 0.17),
    "History": (0.05, 0.34, 0.28),
    "RAG": (0.03, 0.13, 0.10),
    "Tool": (0.07, 0.18, 0.11),
    "ICL": (0.02, 0.03, 0.00),
}

PAPER_POS_N: dict[str, str] = {
    "External": "14/14", "History": "10/14", "ICL": "7/14",
    "RAG": "12/14", "Tool": "12/13",
}

PAPER_AMAE: dict[str, tuple[float, float]] = {
    "External": (0.08, 3.61),
    "History": (-1.60, 0.12),
    "ICL": (0.14, -0.04),
    "RAG": (-0.04, 0.59),
    "Tool": (-0.67, 3.39),
}

# tab:stats_inference, transcribed from COLM_camera_ready/sections/appendix.tex.
# suite -> (mean dUAI, CI lo, CI hi, p_BH); "range" and "pearson" carry
# (estimate, CI lo, CI hi).
PAPER_STATS_INFERENCE: dict[str, tuple[float, ...]] = {
    "External": (0.17, 0.13, 0.22, 0.0),   # p printed as "<0.01"
    "History": (0.28, 0.12, 0.47, 0.02),
    "Icl": (0.00, -0.02, 0.03, 0.81),
    "Rag": (0.10, 0.05, 0.17, 0.0),
    "Tool": (0.11, 0.04, 0.20, 0.0),
}
PAPER_STATS_RANGE = (0.40, 0.20, 0.60)
PAPER_STATS_PEARSON = (-0.24, -0.43, -0.00)

# Claims that are known to disagree with the data, with the size of the
# disagreement. Every entry must have a row in docs/RECONCILIATION.md.
#
# This exists so tolerances can be tightened without hiding anything: a
# divergence recorded here is tracked, whereas a tolerance wide enough to
# swallow it is not. A claim listed here that starts *passing*, or that
# fails by a different amount, is itself an error -- the ledger is then
# stale and must be updated deliberately.
KNOWN_DIVERGENCES: dict[tuple[str, str], float] = {
    # Empty by design. The four tables that used to diverge (D6) are now
    # input-ed from generated bodies via scripts/sync_paper_tables.py, so
    # the paper prints exactly what the code computes. Entries belong here
    # only when a claim genuinely disagrees and the disagreement is
    # recorded in docs/RECONCILIATION.md -- never to silence a tolerance.
}

# How far a known divergence may move before it counts as a new problem.
# The values above are measured, so this only absorbs numeric wobble.
KNOWN_SLACK = 0.002

# Half of the last printed digit for a value shown to 2 d.p. Tolerances on
# such tables cannot go below this: a smaller one flags ordinary rounding as
# disagreement, which is what --strict used to do here.
ROUNDING_2DP = 0.005

# Findings 3: paper text values.
# NOTE: these are the *open-weight* tier means, matching the "Open-weight"
# curve in Figure 4 (fig4_dose_response.py keeps the two tiers separate) and
# the numbers quoted in findings.tex Sec. "Finding 3". Do not pool with API.
PAPER_DOSE: dict[str, dict[str, float]] = {
    "External": {"15": 0.32, "25": 0.26, "40": 0.18},
    "Rag": {"15": 0.23, "25": 0.15, "40": 0.06},
}

# Same claim for the API tier ("API models show the same monotonic decrease on
# External at roughly 2x smaller magnitudes; on RAG they peak at delta=25").
PAPER_DOSE_API: dict[str, dict[str, float]] = {
    "External": {"15": 0.16, "25": 0.13, "40": 0.08},
    "Rag": {"15": 0.10, "25": 0.11, "40": 0.05},
}


def verify_main(all_data, mismatches, strict: bool):
    header("1. Main results table  (acc10_control, disc_delta)")
    acc_tol = 0.005 if strict else 0.011  # paper rounds Acc to nearest 1%
    disc_tol = 0.01 if strict else 0.015
    n_ok = n_fail = 0
    for (slug, suite), (p_acc, p_disc) in PAPER_MAIN.items():
        e = find_entry(all_data, slug, suite)
        if e is None:
            mismatches.append(Mismatch("main", f"{slug}/{suite}",
                                       p_acc, float("nan"), acc_tol))
            n_fail += 1
            continue
        s_acc = check("main", f"{slug}/{suite} Acc",
                      p_acc, e["acc10_control"], acc_tol, mismatches)
        s_disc = check("main", f"{slug}/{suite} Disc",
                       p_disc, e["disc_delta"], disc_tol, mismatches)
        if s_acc == "OK" and s_disc == "OK":
            n_ok += 1
        else:
            n_fail += 1
    print(f"  {n_ok}/{n_ok + n_fail} cells passed.")


def verify_pathway(all_data, mismatches, strict: bool):
    header("2. UAI by pathway  (suite-mean over 14 models)")
    tol = 0.01 if strict else 0.015
    for suite, (p_irr, p_pls, p_disc) in PAPER_PATHWAY.items():
        mapped = SUITE_NAME_MAP.get(suite, suite)
        entries = [d for d in all_data if d["suite"] == mapped]
        if suite == "Tool":
            entries = [d for d in entries if d.get("parse_rate", 0) > 0.05]
        if not entries:
            print(f"  {suite}: NO DATA")
            continue
        d_irr = float(np.mean([d["uai_irr"] for d in entries]))
        d_pls = float(np.mean([d["uai_plaus"] for d in entries]))
        d_disc = float(np.mean([d["disc_delta"] for d in entries]))
        s1 = check("pathway", f"{suite} UAI_irr", p_irr, d_irr, tol, mismatches)
        s2 = check("pathway", f"{suite} UAI_pls", p_pls, d_pls, tol, mismatches)
        s3 = check("pathway", f"{suite} Disc",   p_disc, d_disc, tol, mismatches)
        ok = all(s == "OK" for s in (s1, s2, s3))
        print(f"  {suite:10s}: irr={d_irr:.4f} pls={d_pls:.4f} disc={d_disc:.4f}  "
              f"[{'OK' if ok else 'FAIL'}]")


def verify_pos_n(all_data, mismatches):
    header("3. UAI summary  (#models with positive Disc)")
    for suite, paper in PAPER_POS_N.items():
        mapped = SUITE_NAME_MAP.get(suite, suite)
        entries = [d for d in all_data if d["suite"] == mapped]
        if suite == "Tool":
            entries = [d for d in entries if d.get("parse_rate", 0) > 0.05]
        n_pos = sum(1 for d in entries if d["disc_delta"] > 0)
        computed = f"{n_pos}/{len(entries)}"
        ok = computed == paper
        if not ok:
            # Store as a numeric mismatch on "pos count" only
            paper_pos = int(paper.split("/")[0])
            mismatches.append(Mismatch("pos_n", suite, paper_pos, n_pos, 0))
        print(f"  {suite:10s}: +/n = {computed}  (paper: {paper})  "
              f"[{'OK' if ok else 'FAIL'}]")


def verify_anchored_mae(all_data, mismatches, unchecked: list[str]):
    header("4. Anchored MAE table")
    # The paper prints these to 2 d.p., so a legitimate rounding difference is
    # at most 0.005. The tolerance used to be 0.20 -- twenty times the largest
    # actual disagreement.
    #
    # This check previously read `mae_irr` / `mae_plaus` from the unified
    # summaries. Those keys have never existed there (only `mae_control` does),
    # so every suite hit the "skipped" branch and PAPER_AMAE verified nothing
    # at any tolerance. The deltas come from the per-record generations, which
    # is where the table itself gets them.
    tol = ROUNDING_2DP
    records_dirs = [d for d in (RESULTS / "full_benchmark", RESULTS / "api_benchmark")
                    if d.is_dir() and any(d.glob("*/*/results.jsonl"))]
    if not records_dirs:
        print("  NOT CHECKED: needs results/**/results.jsonl (results bundles).")
        unchecked.append("anchored MAE (tab:anchored_mae)")
        return

    from anchorbench.paper.tables_appendix import compute_delta_mae_by_suite
    delta = compute_delta_mae_by_suite(records_dirs)
    for suite, (p_irr, p_pls) in PAPER_AMAE.items():
        key = SUITE_NAME_MAP.get(suite, suite).lower()
        d = delta.get(key, {})
        d_irr, d_pls = d.get("irr"), d.get("pls")
        if d_irr is None or d_pls is None:
            print(f"  {suite:10s}: NOT CHECKED (no delta computed)")
            unchecked.append(f"anchored MAE / {suite}")
            continue
        s = [check("amae", f"{suite} dMAE_irr", p_irr, d_irr, tol, mismatches),
             check("amae", f"{suite} dMAE_pls", p_pls, d_pls, tol, mismatches)]
        print(f"  {suite:10s}: dMAE_irr={d_irr:+.2f} dMAE_pls={d_pls:+.2f}  "
              f"[{_worst(s)}]")


def verify_stats_inference(all_data, mismatches):
    """tab:stats_inference, checked against the same values the table prints."""
    header("4b. Statistical summary table (tab:stats_inference)")
    # Same reasoning as above: values are printed to 2 d.p.
    tol = ROUNDING_2DP
    # Import here: tables_appendix pulls in matplotlib-free but heavier deps,
    # and --quick should not pay for them.
    from anchorbench.paper.tables_appendix import stats_inference_values

    v = stats_inference_values(all_data)
    for suite, (p_mean, p_lo, p_hi, p_p) in PAPER_STATS_INFERENCE.items():
        row = v["suites"].get(suite)
        if row is None:
            print(f"  {suite:10s}: skipped (not in data)")
            continue
        s = [
            check("stats", f"{suite} mean", p_mean, row["mean"], tol, mismatches),
            check("stats", f"{suite} CI lo", p_lo, row["lo"], tol, mismatches),
            check("stats", f"{suite} CI hi", p_hi, row["hi"], tol, mismatches),
        ]
        # The paper prints "<0.01" rather than a value, so only check the
        # printed p when the paper gives a number.
        if p_p > 0:
            s.append(check("stats", f"{suite} p_BH", p_p, row["p_bh"], tol, mismatches))
        elif row["p_bh"] >= 0.01:
            s.append(check("stats", f"{suite} p_BH<0.01", 0.0, row["p_bh"], 0.01, mismatches))
        print(f"  {suite:10s}: mean={row['mean']:+.2f} "
              f"CI=[{row['lo']:+.2f}, {row['hi']:+.2f}] p={row['p_bh']:.3f}  "
              f"[{_worst(s)}]")

    for name, paper, got in (
        ("range", PAPER_STATS_RANGE, v["range"]),
        ("Pearson", PAPER_STATS_PEARSON, v["pearson"]),
    ):
        s = [
            check("stats", f"{name} est", paper[0], got["est"], tol, mismatches),
            check("stats", f"{name} CI lo", paper[1], got["lo"], tol, mismatches),
            check("stats", f"{name} CI hi", paper[2], got["hi"], tol, mismatches),
        ]
        print(f"  {name:10s}: est={got['est']:+.2f} "
              f"CI=[{got['lo']:+.2f}, {got['hi']:+.2f}]  [{_worst(s)}]")


def _worst(statuses: list[str]) -> str:
    for level in ("MISSING", "FAIL", "KNOWN"):
        if level in statuses:
            return level
    return "OK"


def verify_dose(all_data, mismatches, strict: bool):
    header("5. Dose-response (suite mean of UAI_pls, per model tier)")
    tol = 0.02 if strict else 0.04  # paper text rounds to 2 d.p., we recompute
    tiers = (
        ("open-weight", OW_SLUGS, PAPER_DOSE, "dose"),
        ("API", API_SLUGS, PAPER_DOSE_API, "dose_api"),
    )
    for tier_name, slugs, paper_vals, kind in tiers:
        print(f"  -- {tier_name} tier ({len(slugs)} models) --")
        for suite, by_off in paper_vals.items():
            entries = [d for d in all_data
                       if d["suite"] == suite
                       and d.get("model_slug") in slugs
                       and "by_offset" in d]
            for off, p_val in by_off.items():
                vals = [d["by_offset"][off]["uai_plaus"]
                        for d in entries if off in d.get("by_offset", {})]
                if not vals:
                    print(f"  {suite}@{off}: no data")
                    continue
                d_val = float(np.mean(vals))
                s = check(kind, f"{suite}@offset={off} ({tier_name})",
                          p_val, d_val, tol, mismatches)
                print(f"  {suite}@offset={off:>2s}: data={d_val:.3f} "
                      f"(paper {p_val:.2f}) [{s}]")


def verify_intext(all_data, mismatches, strict: bool):
    header("6. Key in-text claims")
    cells = [d for d in all_data if d.get("parse_rate", 0) > 0.05]
    n_cells = len(cells)
    n_pos = sum(1 for d in cells if d["disc_delta"] > 0)
    # Paper: "55/69 cells (80%) show positive discrimination"
    print(f"  Computable cells: {n_cells} (paper 69)")
    print(f"  Positive Disc cells: {n_pos} (paper 55, {55/69*100:.0f}%)")
    if n_cells != 69:
        mismatches.append(Mismatch("intext", "n_cells", 69, n_cells, 0))
    if n_pos != 55:
        mismatches.append(Mismatch("intext", "n_pos_disc", 55, n_pos, 0))

    accs = np.array([d["acc10_control"] for d in cells])
    discs = np.array([d["disc_delta"] for d in cells])
    r = float(np.corrcoef(accs, discs)[0, 1])
    print(f"  Pearson r(Acc, Disc): {r:.3f} (paper -0.24)")
    check("intext", "Pearson_r", -0.24, r, 0.02 if strict else 0.04, mismatches)


def verify_extension_csvs(mismatches, strict: bool):
    header("7. Extension CSVs (gold-shift, sampling, mitigation, icl-dist)")
    tol_pct = 0.5 if strict else 1.0  # percentage points
    tol_disc = 0.015
    tol_mae = 0.20

    # 7a. Gold-shift decomposition
    PAPER_GS = {
        "external": (33.9, 20.0, 19.9, 48.0),
        "history":  (39.2, 44.0, 29.5, 48.7),
        "icl":      (22.9, 20.0, 20.3, 25.4),
        "rag":      (22.2, 20.2, 16.4, 27.9),
        "tool":     (28.6, 17.5, 17.6, 39.9),
    }
    gs = RESULTS / "revision/gold_shift_decomposition/gold_shift_aggregated.csv"
    if gs.exists():
        with open(gs) as f:
            for row in csv.DictReader(f):
                if row["suite"] not in PAPER_GS:
                    continue
                p = PAPER_GS[row["suite"]]
                d = (float(row["harmful_pct"]), float(row["helpful_pct"]),
                     float(row["irrelevant_harmful_pct"]),
                     float(row["plausible_harmful_pct"]))
                for name, pv, dv in zip(("harm", "help", "irr_h", "pls_h"), p, d):
                    check("goldshift", f"{row['suite']}/{name}",
                          pv, dv, tol_pct, mismatches)
                print(f"  goldshift  {row['suite']:8s}: harm={d[0]:.1f}({p[0]:.1f}) "
                      f"help={d[1]:.1f}({p[1]:.1f})")
    else:
        print(f"  gold_shift CSV missing: {gs}")

    # 7b. Sampling robustness
    # Transcribed from Table "tab:sampling_robustness" in
    # COLM/sections/appendix.tex (Sec. "Sampling robustness"):
    # (Disc_delta greedy, sampled mean, sampled SD).
    PAPER_SAMP = {
        ("external", "Qwen-7B"): (0.26, 0.29, 0.02),
        ("external", "Llama-8B"): (0.30, 0.24, 0.07),
        ("rag", "Qwen-7B"): (0.19, 0.18, 0.02),
        ("rag", "Llama-8B"): (0.16, 0.06, 0.04),
        ("icl_dist", "Qwen-7B"): (0.07, 0.08, 0.02),
        ("icl_dist", "Llama-8B"): (0.03, 0.00, 0.02),
    }
    samp = RESULTS / "revision/sampling_robustness/sampling_robustness_summary.csv"
    if samp.exists():
        with open(samp) as f:
            for row in csv.DictReader(f):
                key = (row["suite"], row["model"])
                if key not in PAPER_SAMP:
                    continue
                p_g, p_m, p_s = PAPER_SAMP[key]
                d_g = float(row["disc_delta_greedy"])
                d_m = float(row["disc_delta_sample_mean"])
                d_s = float(row["disc_delta_sample_std"])
                check("sampling", f"{key[0]}/{key[1]} greedy",
                      p_g, d_g, tol_disc, mismatches)
                check("sampling", f"{key[0]}/{key[1]} mean",
                      p_m, d_m, tol_disc, mismatches)
                check("sampling", f"{key[0]}/{key[1]} sd",
                      p_s, d_s, tol_disc, mismatches)
                print(f"  sampling   {key[0]:9s}/{key[1]:8s}: greedy={d_g:.2f} "
                      f"mean={d_m:.2f} sd={d_s:.2f}")
    else:
        print(f"  sampling CSV missing: {samp}")

    # 7c. Mitigation headroom
    PAPER_MIT = {
        ("external", "Qwen_Qwen2.5-7B-Instruct"): {
            "baseline": (0.26, 7.7), "ignore": (0.16, 5.0),
            "self_check": (0.45, 8.8), "cot": (0.14, 1.1)},
        ("external", "meta-llama_Llama-3.1-8B-Instruct"): {
            "baseline": (0.30, 5.5), "ignore": (0.10, 3.4),
            "self_check": (0.28, 7.5), "cot": (0.31, 6.1)},
        ("rag", "Qwen_Qwen2.5-7B-Instruct"): {
            "baseline": (0.20, 9.8), "ignore": (0.14, 6.8),
            "self_check": (0.30, 12.2), "cot": (0.03, 1.3)},
        ("rag", "meta-llama_Llama-3.1-8B-Instruct"): {
            "baseline": (0.16, 4.8), "ignore": (0.04, 4.4),
            "self_check": (0.11, 6.0), "cot": (0.08, 4.5)},
    }
    mit_base = RESULTS / "revision/mitigation_headroom"
    for (suite, model), strats in PAPER_MIT.items():
        for strat, (p_d, p_m) in strats.items():
            sf = mit_base / suite / model / strat / "summary.json"
            if not sf.exists():
                mismatches.append(Mismatch("mitigation",
                                           f"{suite}/{model}/{strat}",
                                           p_d, float("nan"), tol_disc))
                continue
            d = load_json(sf)
            d_disc = d.get("disc_delta")
            d_mae = d.get("mae_control")
            check("mitigation", f"{suite}/{strat}/{model.split('_')[-1]} disc",
                  p_d, d_disc, tol_disc, mismatches)
            check("mitigation", f"{suite}/{strat}/{model.split('_')[-1]} mae",
                  p_m, d_mae, tol_mae, mismatches)


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="Only check main + uai-pathway (no extension CSVs).")
    p.add_argument("--strict", action="store_true",
                   help="Tighten tolerances by ~2x (catch rounding drift).")
    p.add_argument("--json", type=Path, default=None, metavar="PATH",
                   help="Also write the full result as JSON, for regenerating "
                        "the docs/RECONCILIATION.md rows mechanically.")
    args = p.parse_args(list(argv) if argv is not None else None)

    ow = load_json(RESULTS / "full_benchmark/unified_all_suites.json")
    api = load_json(RESULTS / "api_benchmark/unified_all_suites.json")
    all_data = ow + api
    print(f"Loaded {len(all_data)} (model, suite) entries "
          f"({len(ow)} OW + {len(api)} API).")

    mismatches: list[Mismatch] = []
    unchecked: list[str] = []
    verify_main(all_data, mismatches, args.strict)
    verify_pathway(all_data, mismatches, args.strict)
    verify_pos_n(all_data, mismatches)
    verify_anchored_mae(all_data, mismatches, unchecked)
    verify_dose(all_data, mismatches, args.strict)
    verify_intext(all_data, mismatches, args.strict)
    ran_sections = {"main", "pathway", "posn", "amae", "dose", "intext"}
    if not args.quick:
        verify_stats_inference(all_data, mismatches)
        verify_extension_csvs(mismatches, args.strict)
        ran_sections |= {"stats", "gs", "samp", "mit"}

    new = [m for m in mismatches if not m.known]
    known = [m for m in mismatches if m.known]

    # A claim listed in KNOWN_DIVERGENCES that no longer diverges means the
    # ledger is stale. That is a failure too: the record must track reality.
    # Only sections that actually ran can declare a ledger row stale;
    # --quick skips several, and a skipped check is not evidence of anything.
    still_diverging = {(m.section, m.label) for m in known}
    stale = sorted({k for k in KNOWN_DIVERGENCES if k[0] in ran_sections}
                   - still_diverging)

    header("SUMMARY")
    print(f"  New mismatches:     {len(new)}")
    print(f"  Known divergences:  {len(known)}  (recorded in docs/RECONCILIATION.md)")
    if unchecked:
        print(f"  Not checked:        {len(unchecked)}")
    if stale:
        print(f"  Stale ledger rows:  {len(stale)}")

    if known:
        print("\n  Known, tracked:")
        for m in known:
            print(m)
    if new:
        print("\n  New:")
        for m in new[:30]:
            print(m)
        if len(new) > 30:
            print(f"  ... and {len(new) - 30} more")
    if stale:
        print("\n  Listed as known but no longer diverging -- update "
              "KNOWN_DIVERGENCES and docs/RECONCILIATION.md:")
        for section, label in stale:
            print(f"  [{section}] {label}")

    if unchecked:
        print("\n  Claims that could not be checked here:")
        for u in unchecked:
            print(f"  - {u}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({
                "strict": args.strict,
                "quick": args.quick,
                "new": [vars(m) for m in new],
                "known": [vars(m) for m in known],
                "stale_known_rows": [list(s) for s in stale],
                "unchecked": unchecked,
            }, f, indent=2)
        print(f"\n  wrote {args.json}")

    if new or stale:
        return 1
    print("\n  All checked claims verified, or diverging by exactly the "
          "amount recorded in the ledger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
