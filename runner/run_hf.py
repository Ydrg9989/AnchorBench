#!/usr/bin/env python3
"""Run anchoring benchmark with local HuggingFace models.

Usage:
  python runner/run_hf.py --config runner/config.models.yaml
  python runner/run_hf.py --config runner/config.models.yaml --run_name hf_v02
  python runner/run_hf.py --config runner/config.models.yaml --suites icl,rag --max_items 2
  python runner/run_hf.py --config runner/config.models.yaml --dry_run
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    CTRL_LOW_HIGH,
    EXTERNAL_LIKE_SUITES,
    append_jsonl,
    load_completed,
    load_dataset,
    make_result,
    parse_answer_int,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BACKEND = "hf"


# ── model helpers ─────────────────────────────────────────────────────

def _load_model(model_id: str, dtype: str = "bfloat16", device_map: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = getattr(torch, dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    log.info("Loaded %s  dtype=%s  device=%s", model_id, dtype, model.device)
    return model, tokenizer


def _wrap_chat(prompt: str, system_prompt: str, tokenizer) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _generate_batch(
    prompts: list[str], model, tokenizer, gen_kwargs: dict, batch_size: int = 16,
) -> list[str]:
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc, **gen_kwargs, pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        results.extend(decoded)
    return results


# ── main loop ─────────────────────────────────────────────────────────

def run(cfg: dict, args) -> None:
    run_name = args.run_name or cfg["run_name"]
    all_items = load_dataset(cfg["dataset_path"])
    hf_models = cfg["hf_models"]
    n_samples = cfg["n_samples_per_item"]
    sys_prompt = cfg.get("system_prompt", "")
    batch_size = cfg.get("batch_size", 16)
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

    dec = cfg["decoding"]
    gen_kwargs = {
        "do_sample": True,
        "temperature": dec["temperature"],
        "top_p": dec["top_p"],
        "max_new_tokens": dec["max_tokens"],
    }

    out_dir = Path(cfg.get("output_dir", "runner_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.jsonl"

    completed, turn1_answers = load_completed(out_path)
    log.info("Resume: %d completed records", len(completed))

    if args.dry_run:
        _dry_run(items, hf_models, n_samples, completed)
        return

    for model_cfg in hf_models:
        model_id = model_cfg["model_id"]
        dtype = model_cfg.get("dtype", "bfloat16")
        device_map = model_cfg.get("device_map", "auto")

        log.info("=== Loading %s ===", model_id)
        model, tokenizer = _load_model(model_id, dtype, device_map)

        fh = open(out_path, "a", encoding="utf-8")
        written = 0

        try:
            for si in range(n_samples):
                torch.manual_seed(42 + si)
                log.info("  [%s] sample %d/%d",
                         model_id.split("/")[-1], si + 1, n_samples)

                # ── Pass 1: independent prompts ──
                pass1_jobs: list[tuple[dict, str, str, str]] = []

                for item in items:
                    iid = item["item_id"]
                    suite = item["suite"]

                    if suite in EXTERNAL_LIKE_SUITES:
                        for cond in CTRL_LOW_HIGH:
                            if (model_id, BACKEND, iid, cond, si) in completed:
                                continue
                            prompt = item["prompts"][cond]
                            chat = _wrap_chat(prompt, sys_prompt, tokenizer)
                            pass1_jobs.append((item, cond, prompt, chat))

                    elif suite == "self_generated":
                        if (model_id, BACKEND, iid, "turn1", si) not in completed:
                            prompt = item["prompts"]["turn1"]
                            chat = _wrap_chat(prompt, sys_prompt, tokenizer)
                            pass1_jobs.append((item, "turn1", prompt, chat))

                        if (model_id, BACKEND, iid, "turn2_fresh", si) not in completed:
                            prompt = item["prompts"]["turn2_fresh"]
                            chat = _wrap_chat(prompt, sys_prompt, tokenizer)
                            pass1_jobs.append((item, "turn2_fresh", prompt, chat))

                        for ch in item["prompts"].get("turn2_controlled_history", []):
                            cond = f"turn2_controlled_history_{ch['anchor']}"
                            if (model_id, BACKEND, iid, cond, si) not in completed:
                                prompt = ch["prompt"]
                                chat = _wrap_chat(prompt, sys_prompt, tokenizer)
                                pass1_jobs.append((item, cond, prompt, chat))

                if pass1_jobs:
                    chat_prompts = [j[3] for j in pass1_jobs]
                    outputs = _generate_batch(
                        chat_prompts, model, tokenizer, gen_kwargs, batch_size
                    )
                    for (item, cond, prompt, _), raw in zip(pass1_jobs, outputs):
                        ans, ok = parse_answer_int(raw, prompt)
                        rec = make_result(run_name, model_id, BACKEND,
                                          item, cond, si, ans, ok, raw)
                        append_jsonl(fh, rec)
                        completed.add((model_id, BACKEND, item["item_id"], cond, si))
                        if cond == "turn1":
                            turn1_answers[
                                (model_id, BACKEND, item["item_id"], si)
                            ] = ans
                        written += 1

                # ── Pass 2: turn2_history (depends on turn1) ──
                pass2_jobs: list[tuple[dict, str, str]] = []
                pass2_null: list[dict] = []

                for item in items:
                    if item["suite"] != "self_generated":
                        continue
                    iid = item["item_id"]
                    if (model_id, BACKEND, iid, "turn2_history", si) in completed:
                        continue
                    a1 = turn1_answers.get((model_id, BACKEND, iid, si))
                    if a1 is not None:
                        prompt = item["prompts"]["turn2_history"].replace(
                            "{{A1}}", str(a1)
                        )
                        chat = _wrap_chat(prompt, sys_prompt, tokenizer)
                        pass2_jobs.append((item, prompt, chat))
                    else:
                        pass2_null.append(item)

                if pass2_jobs:
                    chat_prompts = [j[2] for j in pass2_jobs]
                    outputs = _generate_batch(
                        chat_prompts, model, tokenizer, gen_kwargs, batch_size
                    )
                    for (item, prompt, _), raw in zip(pass2_jobs, outputs):
                        ans, ok = parse_answer_int(raw, prompt)
                        rec = make_result(run_name, model_id, BACKEND,
                                          item, "turn2_history", si, ans, ok, raw)
                        append_jsonl(fh, rec)
                        completed.add(
                            (model_id, BACKEND, item["item_id"], "turn2_history", si)
                        )
                        written += 1

                for item in pass2_null:
                    rec = make_result(run_name, model_id, BACKEND,
                                      item, "turn2_history", si, None, False, "")
                    append_jsonl(fh, rec)
                    completed.add(
                        (model_id, BACKEND, item["item_id"], "turn2_history", si)
                    )
                    written += 1

                log.info("    sample %d done (%d written)", si + 1, written)

        finally:
            fh.close()

        log.info("  Wrote %d records for %s", written, model_id)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    log.info("Finished → %s", out_path)


def _dry_run(items, hf_models, n_samples, completed):
    by_suite = Counter(it["suite"] for it in items)
    n_ext_like = sum(n for s, n in by_suite.items() if s in EXTERNAL_LIKE_SUITES)
    n_self = by_suite.get("self_generated", 0)
    n_ctrl = sum(
        len(it["prompts"].get("turn2_controlled_history", []))
        for it in items if it["suite"] == "self_generated"
    )
    models = [m["model_id"] for m in hf_models]
    total = (n_ext_like * 3 + n_self * 3 + n_ctrl) * n_samples * len(models)

    log.info("DRY RUN (HF)")
    log.info("  Models:       %s", models)
    log.info("  Items by suite: %s", dict(by_suite))
    log.info("  Samples/item: %d", n_samples)
    log.info("  Generations:  ~%d  (completed: %d)", total, len(completed))


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Run anchoring benchmark with HF models")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--suites", type=str, default=None,
                   help="Comma-separated suites to run")
    p.add_argument("--max_items", type=int, default=None,
                   help="Max items per suite (for testing)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    run(cfg, args)


if __name__ == "__main__":
    main()
