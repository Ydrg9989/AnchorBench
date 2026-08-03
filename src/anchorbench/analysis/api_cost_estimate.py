"""Extrapolate API cost from a smoke-test run.

Reads the per-suite ``summary.json`` files produced by
``anchorbench.runners.api`` (which include `api_usage.prompt_tokens` and
`api_usage.completion_tokens` aggregates) and the OpenRouter price list,
then projects the full-panel cost for each model.

Usage::

    python -m anchorbench.analysis.api_cost_estimate \\
        --in_dir results/rebuttal/large_api_smoke \\
        --n_items_smoke 10 --n_items_full 360
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# OpenRouter prices (USD per 1M tokens). Updated 2026-05; live values
# are fetched at runtime when OPENROUTER_API_KEY is available.
FALLBACK_PRICES = {
    "openai/gpt-5.4":              {"in": 2.50, "out": 15.00},
    "openai/gpt-5.4-mini":         {"in": 0.75, "out":  4.50},
    "anthropic/claude-sonnet-4.6": {"in": 3.00, "out": 15.00},
    "google/gemini-2.5-pro":       {"in": 1.25, "out": 10.00},
    "x-ai/grok-4.3":               {"in": 1.25, "out":  2.50},
}


def _fetch_live_prices() -> dict[str, dict[str, float]]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {}
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        out: dict[str, dict[str, float]] = {}
        for m in data.get("data", []):
            mid = m.get("id", "")
            p = m.get("pricing", {})
            try:
                out[mid] = {
                    "in":  float(p.get("prompt", 0)) * 1e6,
                    "out": float(p.get("completion", 0)) * 1e6,
                }
            except (TypeError, ValueError):
                continue
        return out
    except Exception as e:
        log.warning("Failed to fetch live OpenRouter prices: %s", e)
        return {}


def _walk(in_dir: Path) -> list[dict]:
    """Walk in_dir/<suite>/<model_slug>/results.jsonl and sum per-record
    api_usage. Returns one row per (suite, model)."""
    rows: list[dict] = []
    for suite_dir in sorted(in_dir.iterdir()):
        if not suite_dir.is_dir() or suite_dir.name.startswith("_"):
            continue
        for slug_dir in sorted(suite_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            results_path = slug_dir / "results.jsonl"
            if not results_path.exists():
                continue
            in_tok = out_tok = reasoning_tok = 0
            cached_tok = 0
            actual_cost = 0.0
            n_recs = n_items = 0
            items_seen: set[str] = set()
            model_id = ""
            with open(results_path) as f:
                for line in f:
                    r = json.loads(line)
                    n_recs += 1
                    items_seen.add(r.get("item_id", ""))
                    model_id = r.get("model_id", model_id)
                    u = r.get("api_usage") or {}
                    in_tok += int(u.get("prompt_tokens", 0) or 0)
                    out_tok += int(u.get("completion_tokens", 0) or 0)
                    actual_cost += float(u.get("cost", 0) or 0)
                    cdet = u.get("completion_tokens_details") or {}
                    reasoning_tok += int(cdet.get("reasoning_tokens", 0) or 0)
                    pdet = u.get("prompt_tokens_details") or {}
                    cached_tok += int(pdet.get("cached_tokens", 0) or 0)
            n_items = len(items_seen)
            rows.append({
                "suite": suite_dir.name,
                "model": model_id or slug_dir.name.replace("_", "/", 1),
                "n_items": n_items,
                "n_records": n_recs,
                "in_tok": in_tok,
                "out_tok": out_tok,
                "reasoning_tok": reasoning_tok,
                "cached_tok": cached_tok,
                "actual_cost_usd": actual_cost,
            })
    return rows


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="API cost estimator")
    p.add_argument("--in_dir", type=Path, required=True,
                   help="Directory containing <suite>/<slug>/summary.json")
    p.add_argument("--n_items_smoke", type=int, required=True,
                   help="--max_items value used for the smoke run")
    p.add_argument("--n_items_full", type=int, default=360,
                   help="Item count for the FULL panel (default 360 = core size)")
    args = p.parse_args(argv)

    rows = _walk(args.in_dir)
    if not rows:
        log.error("No summaries found under %s", args.in_dir)
        raise SystemExit(1)

    live = _fetch_live_prices()
    prices = {**FALLBACK_PRICES, **{k: v for k, v in live.items() if v["in"] > 0}}

    # Aggregate by model
    by_model: dict[str, dict] = {}
    for r in rows:
        m = by_model.setdefault(r["model"], {
            "model": r["model"],
            "suites_seen": set(),
            "in_tok": 0, "out_tok": 0,
            "reasoning_tok": 0, "cached_tok": 0,
            "actual_cost_usd": 0.0,
            "n_items": 0, "n_records": 0,
        })
        m["suites_seen"].add(r["suite"])
        m["in_tok"] += r["in_tok"]
        m["out_tok"] += r["out_tok"]
        m["reasoning_tok"] += r["reasoning_tok"]
        m["cached_tok"] += r["cached_tok"]
        m["actual_cost_usd"] += r["actual_cost_usd"]
        m["n_records"] += r["n_records"] or 0

    scale = args.n_items_full / max(args.n_items_smoke, 1)

    print(f"\n{'='*78}")
    print(f"API cost extrapolation (smoke n_items={args.n_items_smoke}, "
          f"scale x{scale:.1f} for full n_items={args.n_items_full})")
    print(f"{'='*78}\n")
    print(f"{'Model':<33s} {'suites':<7s} {'smoke_in':>10s} {'smoke_out':>10s} "
          f"{'reason':>8s} {'smoke_$':>8s}  {'full_in':>11s} {'full_out':>11s} "
          f"{'full_$':>8s}")

    grand_smoke = 0.0
    grand_full = 0.0
    rows_out = []
    for slug, m in sorted(by_model.items()):
        p_in = prices.get(slug, {}).get("in")
        p_out = prices.get(slug, {}).get("out")
        if p_in is None or p_out is None:
            log.warning("No price for %s; skipping cost", slug)
            continue
        # Use OpenRouter-reported actual cost when available; fall back to
        # local price-list computation otherwise. The two should match.
        smoke_cost = m["actual_cost_usd"] or (
            (m["in_tok"] / 1e6) * p_in + (m["out_tok"] / 1e6) * p_out
        )
        full_in = m["in_tok"] * scale
        full_out = m["out_tok"] * scale
        full_cost = smoke_cost * scale
        grand_smoke += smoke_cost
        grand_full += full_cost
        print(f"{slug:<33s} {len(m['suites_seen']):<7d} "
              f"{m['in_tok']:>10,d} {m['out_tok']:>10,d} "
              f"{m['reasoning_tok']:>8,d} ${smoke_cost:>7.3f}  "
              f"{int(full_in):>11,d} {int(full_out):>11,d} "
              f"${full_cost:>7.2f}")
        rows_out.append({
            "model": slug, "suites": sorted(m["suites_seen"]),
            "smoke_in_tok": m["in_tok"], "smoke_out_tok": m["out_tok"],
            "smoke_reasoning_tok": m["reasoning_tok"],
            "smoke_cost_usd": round(smoke_cost, 4),
            "extrapolated_in_tok": int(full_in),
            "extrapolated_out_tok": int(full_out),
            "extrapolated_full_panel_usd": round(full_cost, 2),
            "price_in_per_M": p_in, "price_out_per_M": p_out,
        })

    print("-" * 78)
    print(f"{'TOTAL':<33s} {' ':<7s} {' ':>10s} {' ':>10s} "
          f"${grand_smoke:>7.3f}  {' ':>11s} {' ':>11s} ${grand_full:>7.2f}")
    print()
    print("Notes:")
    print(" - Full extrapolation assumes constant tokens-per-item across the dataset.")
    print(" - Output tokens may *under*-estimate if reasoning models emit hidden CoT.")
    print(" - History suite uses two-stage protocol (~2x records vs other suites).")

    (args.in_dir / "cost_estimate.json").write_text(
        json.dumps({"rows": rows_out, "totals": {
            "smoke_usd": round(grand_smoke, 4),
            "full_panel_usd": round(grand_full, 2),
        }}, indent=2)
    )


if __name__ == "__main__":
    main()
