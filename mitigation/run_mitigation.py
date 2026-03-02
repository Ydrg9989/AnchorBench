#!/usr/bin/env python3
"""Run anchoring mitigation experiments via OpenRouter API.

Supports mitigations: B0, B1, B2, B3, EFR, CxDP.
Post-hoc mitigations (B4, CAC, PLD) are computed in eval_mitigations.py.

Usage:
  python mitigation/run_mitigation.py --config mitigation/config.yaml \\
      --mitigation B1 --run_name mit_b1
  python mitigation/run_mitigation.py --config mitigation/config.yaml \\
      --mitigation CxDP --run_name mit_cxdp --suites external,rag,tool
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies import (
    apply_mitigation,
    cxdp_distill_prompt,
    cxdp_estimate_prompt,
    get_distill_max_tokens,
    get_max_tokens,
    get_parser,
)
from utils import (
    CTRL_LOW_HIGH,
    EXTERNAL_LIKE_SUITES,
    append_jsonl,
    load_completed,
    load_dataset,
    retry_with_backoff,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
BACKEND = "openrouter"

PROMPT_MITIGATIONS = {"B0", "B1", "B2", "B3", "EFR", "CxDP"}


# ── API call ──────────────────────────────────────────────────────────

def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _call_api(
    prompt: str,
    model: str,
    cfg: dict,
    max_tokens: int = 8,
    seed: int | None = None,
) -> str:
    _load_env()
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
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    import time as _time
    for attempt in range(8):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        except requests.exceptions.RequestException as exc:
            log.warning("Request error (attempt %d): %s", attempt + 1, exc)
            _time.sleep(min(2 ** attempt, 60))
            continue
        if resp.status_code == 200:
            break
        if resp.status_code in (401, 403):
            raise RuntimeError(f"Auth failed ({resp.status_code}): {resp.text[:200]}")
        if resp.status_code in (400,):
            log.warning("400 error (attempt %d): %s", attempt + 1, resp.text[:200])
            _time.sleep(2)
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            _time.sleep(min(2 ** attempt, 60))
            continue
        log.warning("Unexpected status %d: %s", resp.status_code, resp.text[:200])
        _time.sleep(min(2 ** attempt, 60))
        continue
    else:
        log.error("API call failed after 8 retries, returning empty")
        return ""
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ── result record ────────────────────────────────────────────────────

def _make_result(
    run_name: str,
    model_id: str,
    item: dict,
    condition: str,
    sample_idx: int,
    answer_int: int | None,
    parsed_ok: bool,
    raw_text: str,
    mitigation: str,
) -> dict:
    return {
        "run_name": run_name,
        "model_id": model_id,
        "backend": BACKEND,
        "item_id": item["item_id"],
        "suite": item["suite"],
        "subtype": item["subtype"],
        "field": item["field"],
        "domain": item["domain"],
        "condition": condition,
        "sample_idx": sample_idx,
        "answer_int": answer_int,
        "parsed_ok": parsed_ok,
        "raw_text": raw_text,
        "mitigation": mitigation,
    }


# ── main loop ─────────────────────────────────────────────────────────

def run(cfg: dict, args) -> None:
    mitigation = args.mitigation
    run_name = args.run_name or f"mit_{mitigation.lower()}"
    all_items = load_dataset(cfg["dataset_path"])
    models = cfg["openrouter_models"]
    n_samples = cfg.get("n_samples_per_item", 5)
    default_max_tok = cfg["decoding"]["max_tokens"]

    suites = set(args.suites.split(",")) if args.suites else None
    items = [it for it in all_items if suites is None or it["suite"] in suites]
    if args.max_items:
        per_suite: dict[str, list] = {}
        for it in items:
            per_suite.setdefault(it["suite"], []).append(it)
        items = []
        for s_items in per_suite.values():
            items.extend(s_items[: args.max_items])

    log.info("Mitigation: %s | Items: %d | Models: %d | Samples: %d",
             mitigation, len(items), len(models), n_samples)

    out_dir = Path(cfg.get("output_dir", "runner_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.jsonl"

    completed, turn1_answers = load_completed(out_path)
    log.info("Resume: %d completed records", len(completed))

    if args.dry_run:
        _dry_run(items, models, n_samples, mitigation, completed)
        return

    parse_fn = get_parser(mitigation)
    max_tok = get_max_tokens(mitigation, default_max_tok)

    fh = open(out_path, "a", encoding="utf-8")
    total_written = 0
    parse_stats: Counter = Counter()

    try:
        for model_id in models:
            log.info("=== %s [%s] ===", model_id, mitigation)
            for si in range(n_samples):
                for item in items:
                    iid = item["item_id"]
                    suite = item["suite"]

                    if suite in EXTERNAL_LIKE_SUITES:
                        for cond in CTRL_LOW_HIGH:
                            if (model_id, BACKEND, iid, cond, si) in completed:
                                continue
                            orig_prompt = item["prompts"][cond]

                            if mitigation == "CxDP":
                                raw, ans, ok = _run_cxdp(
                                    orig_prompt, model_id, cfg, si
                                )
                            else:
                                prompt = apply_mitigation(
                                    orig_prompt, mitigation, suite
                                )
                                raw = _call_api(prompt, model_id, cfg,
                                                max_tokens=max_tok, seed=si)
                                ans, ok = parse_fn(raw, prompt)

                            rec = _make_result(
                                run_name, model_id, item, cond, si,
                                ans, ok, raw, mitigation,
                            )
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, cond, si))
                            total_written += 1
                            parse_stats[(suite, cond, ok)] += 1

                    elif suite == "self_generated":
                        a1 = turn1_answers.get((model_id, BACKEND, iid, si))

                        # turn1
                        if (model_id, BACKEND, iid, "turn1", si) not in completed:
                            orig = item["prompts"]["turn1"]
                            prompt = apply_mitigation(orig, mitigation, suite)
                            raw = _call_api(prompt, model_id, cfg,
                                            max_tokens=max_tok, seed=si)
                            a1, ok = parse_fn(raw, prompt)
                            rec = _make_result(
                                run_name, model_id, item, "turn1", si,
                                a1, ok, raw, mitigation,
                            )
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, "turn1", si))
                            turn1_answers[(model_id, BACKEND, iid, si)] = a1
                            total_written += 1

                        # turn2_fresh
                        if (model_id, BACKEND, iid, "turn2_fresh", si) not in completed:
                            orig = item["prompts"]["turn2_fresh"]
                            prompt = apply_mitigation(orig, mitigation, suite)
                            raw = _call_api(prompt, model_id, cfg,
                                            max_tokens=max_tok, seed=si)
                            ans, ok = parse_fn(raw, prompt)
                            rec = _make_result(
                                run_name, model_id, item, "turn2_fresh", si,
                                ans, ok, raw, mitigation,
                            )
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, "turn2_fresh", si))
                            total_written += 1

                        # turn2_history
                        if (model_id, BACKEND, iid, "turn2_history", si) not in completed:
                            if a1 is not None:
                                orig = item["prompts"]["turn2_history"].replace(
                                    "{{A1}}", str(a1)
                                )
                                prompt = apply_mitigation(orig, mitigation, suite)
                                raw = _call_api(prompt, model_id, cfg,
                                                max_tokens=max_tok, seed=si)
                                ans, ok = parse_fn(raw, prompt)
                            else:
                                raw, ans, ok = "", None, False
                            rec = _make_result(
                                run_name, model_id, item, "turn2_history", si,
                                ans, ok, raw, mitigation,
                            )
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, "turn2_history", si))
                            total_written += 1

                        # controlled history
                        for ch in item["prompts"].get("turn2_controlled_history", []):
                            cond = f"turn2_controlled_history_{ch['anchor']}"
                            if (model_id, BACKEND, iid, cond, si) in completed:
                                continue
                            orig = ch["prompt"]
                            prompt = apply_mitigation(orig, mitigation, suite)
                            raw = _call_api(prompt, model_id, cfg,
                                            max_tokens=max_tok, seed=si)
                            ans, ok = parse_fn(raw, prompt)
                            rec = _make_result(
                                run_name, model_id, item, cond, si,
                                ans, ok, raw, mitigation,
                            )
                            append_jsonl(fh, rec)
                            completed.add((model_id, BACKEND, iid, cond, si))
                            total_written += 1

                log.info("  sample %d/%d  (%d written)", si + 1, n_samples, total_written)
    finally:
        fh.close()

    log.info("Finished. Wrote %d records → %s", total_written, out_path)


def _run_cxdp(
    original_prompt: str, model_id: str, cfg: dict, seed: int,
) -> tuple[str, int | None, bool]:
    """Two-stage Context-Distilled Prompting."""
    from utils import parse_answer_int

    distill_p = cxdp_distill_prompt(original_prompt)
    summary = _call_api(
        distill_p, model_id, cfg,
        max_tokens=get_distill_max_tokens(), seed=seed,
    )
    estimate_p = cxdp_estimate_prompt(summary)
    raw = _call_api(estimate_p, model_id, cfg, max_tokens=8, seed=seed)
    ans, ok = parse_answer_int(raw, estimate_p)
    combined_raw = f"[distill]{summary}[/distill]\n{raw}"
    return combined_raw, ans, ok


def _dry_run(items, models, n_samples, mitigation, completed):
    by_suite = Counter(it["suite"] for it in items)
    n_ext_like = sum(n for s, n in by_suite.items() if s in EXTERNAL_LIKE_SUITES)
    n_self = by_suite.get("self_generated", 0)
    n_ctrl = sum(
        len(it["prompts"].get("turn2_controlled_history", []))
        for it in items if it["suite"] == "self_generated"
    )
    multiplier = 2 if mitigation == "CxDP" else 1
    calls = (n_ext_like * 3 + n_self * 3 + n_ctrl) * n_samples * len(models) * multiplier

    log.info("DRY RUN [%s]", mitigation)
    log.info("  Models:       %s", models)
    log.info("  Suites:       %s", dict(by_suite))
    log.info("  Samples/item: %d", n_samples)
    log.info("  API calls:    ~%d  (completed: %d)", calls, len(completed))


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Run anchoring mitigations")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mitigation", type=str, required=True,
                   choices=sorted(PROMPT_MITIGATIONS),
                   help="Mitigation strategy to apply")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--suites", type=str, default=None)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    run(cfg, args)


if __name__ == "__main__":
    main()
