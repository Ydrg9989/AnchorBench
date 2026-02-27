#!/usr/bin/env python3
"""Parsing-strategy ablation with mismatch diagnostics.

Compares strict / hier-last / hier-first on the same generations.jsonl,
reports anchoring-metric sensitivity and wrong-pick detection."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llm_anchoring.eval.metrics import compute_paired_anchoring, condition_summary
from llm_anchoring.eval.parse_utils import diagnose_parse
from llm_anchoring.eval.parsers import parse_hierarchical, parse_strict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

STRATEGIES = {
    "A_strict": lambda t: parse_strict(t),
    "B_hier_last": lambda t: parse_hierarchical(t, prefer_last=True),
    "B_hier_first": lambda t: parse_hierarchical(t, prefer_last=False),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse-strategy ablation + mismatch audit")
    p.add_argument("--generations", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--top_mismatches", type=int, default=10)
    return p.parse_args()


def _reparse(records: list[dict], strategy_fn) -> list[dict]:
    """Apply a parsing strategy, adding diagnostics for each record."""
    out = []
    for r in records:
        trace = strategy_fn(r["raw_output_text"])
        diag = diagnose_parse(
            trace,
            truth_value=r["truth_value"],
            anchor_value=r.get("anchor_value"),
        )
        nr = dict(r)
        nr["parsed_value"] = trace["parsed_value"]
        nr["parse_ok"] = trace["parse_ok"]
        nr["format_ok"] = trace["format_ok"]
        nr["strategy_used"] = trace["strategy_used"]
        nr["candidate_count"] = trace["candidate_count"]
        nr["chosen_idx"] = trace["chosen_idx"]
        nr["candidates"] = trace["candidates"]
        if trace["parse_ok"]:
            nr["abs_error"] = abs(trace["parsed_value"] - r["truth_value"])
            nr["signed_error"] = trace["parsed_value"] - r["truth_value"]
        else:
            nr["abs_error"] = nr["signed_error"] = None
        nr.update(diag)
        out.append(nr)
    return out


def _sanity_check(records: list[dict]) -> None:
    leaked = sum(
        1 for r in records
        if r["raw_output_text"][:60] and r["raw_output_text"][:60] in r.get("prompt", "")
    )
    log.info("SANITY: %d/%d prompt-leak detections (%s)", leaked, len(records), "WARN" if leaked else "OK")


def _spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")

    def _ranks(vals):
        idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[idx[j + 1]] == vals[idx[j]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    mx, my = float(np.mean(rx)), float(np.mean(ry))
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    records = [json.loads(l) for l in args.generations.read_text().strip().split("\n")]
    log.info("loaded %d records", len(records))

    out_dir = args.out_dir or args.generations.parent / "parse_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    _sanity_check(records)

    strat_results: dict[str, dict] = {}
    strat_anchoring: dict[str, dict] = {}
    strat_records: dict[str, list[dict]] = {}

    for name, fn in STRATEGIES.items():
        reparsed = _reparse(records, fn)
        strat_records[name] = reparsed
        strat_results[name] = condition_summary(reparsed)
        strat_anchoring[name] = compute_paired_anchoring(reparsed)

    conds = sorted({r["icl_condition"] for r in records})
    snames = list(STRATEGIES)

    # ── sections 1–5: existing report ────────────────────────────────
    print("\n══════ PARSING ABLATION REPORT ══════\n")

    print("1) Parse rate by condition × strategy")
    _print_table(conds, snames, strat_results, "parse_rate")

    print("\n2) MAE by condition × strategy")
    _print_table(conds, snames, strat_results, "mae")

    print("\n3) Anchoring metrics by strategy")
    print(f"  {'Strategy':<16} {'IAS_mean':>10} {'IAS_med':>10} {'CI_lo':>10} {'CI_hi':>10} {'OShift':>10}")
    for s in snames:
        a = strat_anchoring[s]
        ci = a.get("ias_ci_95", [None, None])
        print(
            f"  {s:<16} {_f(a.get('ias_mean')):>10} {_f(a.get('ias_median')):>10}"
            f" {_f(ci[0]):>10} {_f(ci[1]):>10} {_f(a.get('outlier_shift_mean')):>10}"
        )

    print("\n4) Cross-strategy IAS sensitivity")
    ias_vecs: dict[str, dict] = {}
    for s in snames:
        ias_vecs[s] = {row["base_id"]: row.get("IAS") for row in strat_anchoring[s]["per_base"]}
    pairs = [(snames[i], snames[j]) for i in range(len(snames)) for j in range(i + 1, len(snames))]
    print(f"  {'Pair':<36} {'N':>5} {'Spearman':>10} {'SignFlips':>10}")
    sensitivity_rows = []
    for sa, sb in pairs:
        va, vb = _paired_ias(ias_vecs[sa], ias_vecs[sb])
        flips = sum(1 for a, b in zip(va, vb) if a * b < 0)
        rho = _spearman(va, vb)
        print(f"  {sa+' vs '+sb:<36} {len(va):>5} {rho:>10.4f} {flips:>10}")
        sensitivity_rows.append({"pair": f"{sa} vs {sb}", "n": len(va), "spearman": round(rho, 4), "sign_flips": flips})

    print("\n5) Strategy-used distribution (B_hier_last)")
    dist: dict[str, int] = defaultdict(int)
    for r in strat_records["B_hier_last"]:
        dist[r["strategy_used"]] += 1
    for k, v in sorted(dist.items()):
        print(f"  {k:<20} {v:>5} ({100 * v / len(records):.1f}%)")

    # ── section 6: mismatch diagnostics ──────────────────────────────
    print("\n6) Mismatch diagnostics (B_hier_last)")
    bl = strat_records["B_hier_last"]
    _report_mismatches(bl, conds, args.top_mismatches, out_dir)

    # ── save ─────────────────────────────────────────────────────────
    summary = {
        "strategies": {
            s: {
                "condition": strat_results[s],
                "anchoring": {k: v for k, v in strat_anchoring[s].items() if k != "per_base"},
            }
            for s in snames
        },
        "sensitivity": sensitivity_rows,
    }
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
    for s in snames:
        pb = strat_anchoring[s]["per_base"]
        if pb:
            _write_csv(out_dir / f"per_base_{s}.csv", pb)

    print(f"\nOutputs saved to {out_dir}/")


def _report_mismatches(recs: list[dict], conds: list[str], top_n: int, out_dir: Path) -> None:
    """Print mismatch and anchor-collision rates, save worst cases."""
    print(f"  {'condition':<25} {'mismatch%':>10} {'anchor_col%':>10} {'candidates':>10}")
    for c in conds:
        cr = [r for r in recs if r["icl_condition"] == c and r["parse_ok"]]
        n = len(cr) or 1
        mm = sum(1 for r in cr if r.get("mismatch_suspect"))
        ac = sum(1 for r in cr if r.get("anchor_collision"))
        med_cands = float(np.median([r["candidate_count"] for r in cr])) if cr else 0
        print(f"  {c:<25} {100 * mm / n:>9.1f}% {100 * ac / n:>9.1f}% {med_cands:>10.0f}")

    suspects = sorted(
        (r for r in recs if r.get("mismatch_suspect")),
        key=lambda r: r.get("chosen_abs_error") or 0,
        reverse=True,
    )[:top_n]
    if suspects:
        print(f"\n  Top {len(suspects)} mismatch suspects (B_hier_last):")
        for r in suspects:
            cands_vals = [c["value"] for c in r.get("candidates", [])]
            print(
                f"    cond={r['icl_condition']:<22} truth={r['truth_value']:>10.2f}"
                f"  parsed={r['parsed_value']:>10.2f}  best_cand_err={r.get('best_candidate_error', 0):>8.2f}"
                f"  method={r['strategy_used']:<16} cands={cands_vals}"
            )
        _write_csv(
            out_dir / "mismatch_suspects.csv",
            [
                {
                    "item_id": r["item_id"],
                    "icl_condition": r["icl_condition"],
                    "truth": r["truth_value"],
                    "parsed": r["parsed_value"],
                    "chosen_err": r.get("chosen_abs_error"),
                    "best_cand_err": r.get("best_candidate_error"),
                    "method": r["strategy_used"],
                    "raw_output": r["raw_output_text"][:200],
                }
                for r in suspects
            ],
        )
    else:
        print("\n  No mismatch suspects detected.")


def _paired_ias(va: dict, vb: dict) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for bid in sorted(set(va) & set(vb)):
        a, b = va[bid], vb[bid]
        if a is not None and b is not None:
            xs.append(a)
            ys.append(b)
    return xs, ys


def _print_table(conds, snames, results, metric):
    header = f"  {'condition':<25}" + "".join(f" {s:>16}" for s in snames)
    print(header)
    for c in conds:
        row = f"  {c:<25}"
        for s in snames:
            row += f" {_f(results[s].get(c, {}).get(metric)):>16}"
        print(row)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def _f(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "N/A"


if __name__ == "__main__":
    main()
