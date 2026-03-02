#!/usr/bin/env python3
"""Run anchoring benchmark via the OpenRouter API.

Usage:
  python runner/run_openrouter.py --config runner/config.models.yaml
  python runner/run_openrouter.py --config runner/config.models.yaml --run_name api_v02
  python runner/run_openrouter.py --config runner/config.models.yaml --suites icl,rag --max_items 2
  python runner/run_openrouter.py --config runner/config.models.yaml --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    CTRL_LOW_HIGH,
    EXTERNAL_LIKE_SUITES,
    append_jsonl,
    load_completed,
    load_dataset,
    make_result,
    parse_answer_int,
    retry_with_backoff,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
BACKEND = "openrouter"


# ── API call ──────────────────────────────────────────────────────────

def _call_api(
    prompt: str,
    model: str,
    cfg: dict,
    seed: int | None = None,
) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    dec = cfg["decoding"]
    sys_prompt = cfg.get("system_prompt", "Follow the instructions exactly.")

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": dec["temperature"],
        "top_p": dec["top_p"],
        "max_tokens": dec["max_tokens"],
    }
    if seed is not None:
        payload["seed"] = seed

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = retry_with_backoff(
        lambda: requests.post(API_URL, headers=headers, json=payload, timeout=60)
    )
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ── main loop ─────────────────────────────────────────────────────────

def run(cfg: dict, args) -> None:
    run_name = args.run_name or cfg["run_name"]
    all_items = load_dataset(cfg["dataset_path"])
    models = cfg["openrouter_models"]
    n_samples = cfg["n_samples_per_item"]
    suites = set(args.suites.split(",")) if args.suites else None

    items = [
        it for it in all_items
        if suites is None or it["suite"] in suites
    ]
    if args.max_items:
        per_suite: dict[str, list] = {}
        for it in items:
            per_suite.setdefault(it["suite"], []).append(it)
        items = []
        for s_items in per_suite.values():
            items.extend(s_items[: args.max_items])

    log.info("Items: %d  (suites: %s)", len(items),
             dict(Counter(it["suite"] for it in items)))

    out_dir = Path(cfg.get("output_dir", "runner_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.jsonl"

    completed, turn1_answers = load_completed(out_path)
    log.info("Resume: %d completed records", len(completed))

    if args.dry_run:
        _dry_run(items, models, n_samples, completed)
        return

    fh = open(out_path, "a", encoding="utf-8")
    total_written = 0
    parse_stats: Counter = Counter()

    try:
        for model_id in models:
            log.info("=== Model: %s ===", model_id)
            for si in range(n_samples):
                for item in items:
                    iid = item["item_id"]
                    suite = item["suite"]

                    if suite in EXTERNAL_LIKE_SUITES:
                        for cond in CTRL_LOW_HIGH:
                            if (model_id, BACKEND, iid, cond, si) in completed:
                                continue
                            prompt = item["prompts"][cond]
                            raw = _call_api(prompt, model_id, cfg, seed=si)
                            ans, ok = parse_answer_int(raw, prompt)
                            rec = make_result(run_name, model_id, BACKEND,
                                              item, cond, si, ans, ok, raw)
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, cond, si))
                            total_written += 1
                            parse_stats[(suite, cond, ok)] += 1

                    elif suite == "self_generated":
                        # turn1
                        a1 = turn1_answers.get((model_id, BACKEND, iid, si))
                        if (model_id, BACKEND, iid, "turn1", si) not in completed:
                            prompt = item["prompts"]["turn1"]
                            raw = _call_api(prompt, model_id, cfg, seed=si)
                            a1, ok = parse_answer_int(raw, prompt)
                            rec = make_result(run_name, model_id, BACKEND,
                                              item, "turn1", si, a1, ok, raw)
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, "turn1", si))
                            turn1_answers[(model_id, BACKEND, iid, si)] = a1
                            total_written += 1
                            parse_stats[(suite, "turn1", ok)] += 1

                        # turn2_fresh
                        if (model_id, BACKEND, iid, "turn2_fresh", si) not in completed:
                            prompt = item["prompts"]["turn2_fresh"]
                            raw = _call_api(prompt, model_id, cfg, seed=si)
                            ans, ok = parse_answer_int(raw, prompt)
                            rec = make_result(run_name, model_id, BACKEND,
                                              item, "turn2_fresh", si, ans, ok, raw)
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, "turn2_fresh", si))
                            total_written += 1
                            parse_stats[(suite, "turn2_fresh", ok)] += 1

                        # turn2_history — skip if A1 failed
                        if (model_id, BACKEND, iid, "turn2_history", si) not in completed:
                            if a1 is not None:
                                prompt = item["prompts"]["turn2_history"].replace(
                                    "{{A1}}", str(a1)
                                )
                                raw = _call_api(prompt, model_id, cfg, seed=si)
                                ans, ok = parse_answer_int(raw, prompt)
                            else:
                                raw, ans, ok = "", None, False
                            rec = make_result(run_name, model_id, BACKEND,
                                              item, "turn2_history", si, ans, ok, raw)
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, "turn2_history", si))
                            total_written += 1
                            parse_stats[(suite, "turn2_history", ok)] += 1

                        # controlled-history variants
                        for ch in item["prompts"].get("turn2_controlled_history", []):
                            cond = f"turn2_controlled_history_{ch['anchor']}"
                            if (model_id, BACKEND, iid, cond, si) in completed:
                                continue
                            prompt = ch["prompt"]
                            raw = _call_api(prompt, model_id, cfg, seed=si)
                            ans, ok = parse_answer_int(raw, prompt)
                            rec = make_result(run_name, model_id, BACKEND,
                                              item, cond, si, ans, ok, raw)
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, cond, si))
                            total_written += 1
                            parse_stats[(suite, cond, ok)] += 1

                log.info("  sample %d/%d  (%d written)", si + 1, n_samples, total_written)
    finally:
        fh.close()

    log.info("Finished. Wrote %d records → %s", total_written, out_path)
    _log_parse_rates(parse_stats)


def _log_parse_rates(stats: Counter) -> None:
    by_suite_cond: dict[tuple, dict] = {}
    for (suite, cond, ok), n in stats.items():
        key = (suite, cond)
        if key not in by_suite_cond:
            by_suite_cond[key] = {"ok": 0, "fail": 0}
        if ok:
            by_suite_cond[key]["ok"] += n
        else:
            by_suite_cond[key]["fail"] += n

    log.info("Parse rates:")
    for (suite, cond), counts in sorted(by_suite_cond.items()):
        total = counts["ok"] + counts["fail"]
        rate = counts["ok"] / total if total else 0
        log.info("  %s / %-30s  %d/%d  (%.1f%%)", suite, cond, counts["ok"], total, rate * 100)


def _dry_run(items, models, n_samples, completed):
    by_suite = Counter(it["suite"] for it in items)
    n_ext_like = sum(n for s, n in by_suite.items() if s in EXTERNAL_LIKE_SUITES)
    n_self = by_suite.get("self_generated", 0)
    n_ctrl = sum(
        len(it["prompts"].get("turn2_controlled_history", []))
        for it in items if it["suite"] == "self_generated"
    )

    calls = (n_ext_like * 3 + n_self * 3 + n_ctrl) * n_samples * len(models)
    already = len(completed)

    log.info("DRY RUN")
    log.info("  Models:       %s", models)
    log.info("  Items by suite: %s", dict(by_suite))
    log.info("  Samples/item: %d", n_samples)
    log.info("  API calls:    ~%d  (completed: %d, remaining: ~%d)",
             calls, already, max(calls - already, 0))


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Run anchoring benchmark via OpenRouter API")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run_name", type=str, default=None,
                   help="Override config run_name")
    p.add_argument("--suites", type=str, default=None,
                   help="Comma-separated suites to run (default: all in dataset)")
    p.add_argument("--max_items", type=int, default=None,
                   help="Max items per suite (for testing)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    run(cfg, args)


if __name__ == "__main__":
    main()
