#!/usr/bin/env python3
"""Run anchoring mitigation experiments with local HuggingFace models.

Supports mitigations: B0, B1, B2, B3, B5 (DoLa), EFR.
Post-hoc mitigations (B4, CAC, PLD) are computed in eval_mitigations.py.

Usage:
  CUDA_VISIBLE_DEVICES=0 python mitigation/run_mitigation_hf.py \\
      --config mitigation/config.yaml --mitigation B1 --run_name hf_mit_b1
  CUDA_VISIBLE_DEVICES=1 python mitigation/run_mitigation_hf.py \\
      --config mitigation/config.yaml --mitigation B5 --run_name hf_mit_b5 \\
      --dola_layers low
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies import (
    apply_mitigation,
    get_max_tokens,
    parse_cot_answer,
)
from utils import (
    CTRL_LOW_HIGH,
    EXTERNAL_LIKE_SUITES,
    append_jsonl,
    load_completed,
    load_dataset,
    parse_answer_int,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BACKEND = "hf"
PROMPT_MITIGATIONS = {"B0", "B1", "B2", "B3", "B5", "EFR"}


def _load_model(model_id: str, dtype: str = "bfloat16", device_map: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = getattr(torch, dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device_map,
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
                trust_remote_code=True,
            )
        decoded = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        results.extend(decoded)
    return results


def _make_result(
    run_name: str, model_id: str, item: dict, condition: str,
    sample_idx: int, answer_int, parsed_ok: bool, raw_text: str,
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


def run(cfg: dict, args) -> None:
    mitigation = args.mitigation
    run_name = args.run_name or f"hf_mit_{mitigation.lower()}"
    all_items = load_dataset(cfg["dataset_path"])
    hf_models = cfg["hf_models"]
    n_samples = cfg.get("n_samples_per_item", 5)
    sys_prompt = cfg.get("system_prompt", "")
    batch_size = cfg.get("batch_size", 16)
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

    dec = cfg["decoding"]
    max_tok = get_max_tokens(mitigation, default_max_tok)
    gen_kwargs: dict = {
        "do_sample": True,
        "temperature": dec["temperature"],
        "top_p": dec["top_p"],
        "max_new_tokens": max_tok,
    }
    if mitigation == "B5" and args.dola_layers:
        gen_kwargs["dola_layers"] = args.dola_layers
        log.info("DoLa enabled: dola_layers=%s", args.dola_layers)

    log.info("Mitigation: %s | Items: %d | Models: %d | Samples: %d",
             mitigation, len(items), len(hf_models), n_samples)

    out_dir = Path(cfg.get("output_dir", "runner_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.jsonl"

    completed, turn1_answers = load_completed(out_path)
    log.info("Resume: %d completed records", len(completed))

    if args.dry_run:
        _dry_run(items, hf_models, n_samples, mitigation, completed)
        return

    parse_fn = parse_cot_answer if mitigation == "B3" else parse_answer_int

    if args.model_idx is not None:
        hf_models = [hf_models[args.model_idx]]

    for model_cfg in hf_models:
        model_id = model_cfg["model_id"]
        dtype = model_cfg.get("dtype", "bfloat16")
        device_map = model_cfg.get("device_map", "auto")

        log.info("=== Loading %s [%s] ===", model_id, mitigation)
        model, tokenizer = _load_model(model_id, dtype, device_map)

        fh = open(out_path, "a", encoding="utf-8")
        written = 0

        try:
            for si in range(n_samples):
                torch.manual_seed(42 + si)
                jobs: list[tuple[dict, str, str, str]] = []

                for item in items:
                    iid = item["item_id"]
                    suite = item["suite"]

                    if suite in EXTERNAL_LIKE_SUITES:
                        for cond in CTRL_LOW_HIGH:
                            if (model_id, BACKEND, iid, cond, si) in completed:
                                continue
                            orig_prompt = item["prompts"][cond]
                            prompt = apply_mitigation(orig_prompt, mitigation, suite)
                            chat = _wrap_chat(prompt, sys_prompt, tokenizer)
                            jobs.append((item, cond, prompt, chat))

                    elif suite == "self_generated":
                        for base_cond in ["turn1", "turn2_fresh"]:
                            if (model_id, BACKEND, iid, base_cond, si) in completed:
                                continue
                            orig = item["prompts"][base_cond]
                            prompt = apply_mitigation(orig, mitigation, suite)
                            chat = _wrap_chat(prompt, sys_prompt, tokenizer)
                            jobs.append((item, base_cond, prompt, chat))

                        for ch in item["prompts"].get("turn2_controlled_history", []):
                            cond = f"turn2_controlled_history_{ch['anchor']}"
                            if (model_id, BACKEND, iid, cond, si) in completed:
                                continue
                            orig = ch["prompt"]
                            prompt = apply_mitigation(orig, mitigation, suite)
                            chat = _wrap_chat(prompt, sys_prompt, tokenizer)
                            jobs.append((item, cond, prompt, chat))

                if jobs:
                    chat_prompts = [j[3] for j in jobs]
                    outputs = _generate_batch(
                        chat_prompts, model, tokenizer, gen_kwargs, batch_size,
                    )
                    for (item, cond, prompt, _), raw in zip(jobs, outputs):
                        ans, ok = parse_fn(raw, prompt)
                        rec = _make_result(
                            run_name, model_id, item, cond, si,
                            ans, ok, raw, mitigation,
                        )
                        append_jsonl(fh, rec)
                        completed.add((model_id, BACKEND, item["item_id"], cond, si))
                        if cond == "turn1":
                            turn1_answers[
                                (model_id, BACKEND, item["item_id"], si)
                            ] = ans
                        written += 1

                # Pass 2: turn2_history (depends on turn1)
                pass2_jobs: list[tuple[dict, str, str]] = []
                for item in items:
                    if item["suite"] != "self_generated":
                        continue
                    iid = item["item_id"]
                    if (model_id, BACKEND, iid, "turn2_history", si) in completed:
                        continue
                    a1 = turn1_answers.get((model_id, BACKEND, iid, si))
                    if a1 is not None:
                        orig = item["prompts"]["turn2_history"].replace(
                            "{{A1}}", str(a1),
                        )
                        prompt = apply_mitigation(orig, mitigation, suite)
                        chat = _wrap_chat(prompt, sys_prompt, tokenizer)
                        pass2_jobs.append((item, prompt, chat))
                    else:
                        rec = _make_result(
                            run_name, model_id, item, "turn2_history", si,
                            None, False, "", mitigation,
                        )
                        append_jsonl(fh, rec)
                        completed.add((model_id, BACKEND, iid, "turn2_history", si))
                        written += 1

                if pass2_jobs:
                    chat_prompts = [j[2] for j in pass2_jobs]
                    outputs = _generate_batch(
                        chat_prompts, model, tokenizer, gen_kwargs, batch_size,
                    )
                    for (item, prompt, _), raw in zip(pass2_jobs, outputs):
                        ans, ok = parse_fn(raw, prompt)
                        rec = _make_result(
                            run_name, model_id, item, "turn2_history", si,
                            ans, ok, raw, mitigation,
                        )
                        append_jsonl(fh, rec)
                        completed.add(
                            (model_id, BACKEND, item["item_id"], "turn2_history", si)
                        )
                        written += 1

                log.info("  [%s] sample %d/%d  (%d written)",
                         model_id.split("/")[-1], si + 1, n_samples, written)

        finally:
            fh.close()

        log.info("Wrote %d records for %s", written, model_id)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    log.info("Finished → %s", out_path)


def _dry_run(items, hf_models, n_samples, mitigation, completed):
    by_suite = Counter(it["suite"] for it in items)
    models = [m["model_id"] for m in hf_models]
    n_ext = sum(n for s, n in by_suite.items() if s in EXTERNAL_LIKE_SUITES)
    n_self = by_suite.get("self_generated", 0)
    n_ctrl = sum(
        len(it["prompts"].get("turn2_controlled_history", []))
        for it in items if it["suite"] == "self_generated"
    )
    total = (n_ext * 3 + n_self * 3 + n_ctrl) * n_samples * len(models)
    log.info("DRY RUN [%s HF]", mitigation)
    log.info("  Models: %s", models)
    log.info("  Suites: %s", dict(by_suite))
    log.info("  Samples/item: %d", n_samples)
    log.info("  Generations: ~%d  (completed: %d)", total, len(completed))


def main():
    p = argparse.ArgumentParser(description="Run HF mitigations")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mitigation", type=str, required=True,
                   choices=sorted(PROMPT_MITIGATIONS))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--suites", type=str, default=None)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--model_idx", type=int, default=None,
                   help="Run only the i-th HF model (0-indexed)")
    p.add_argument("--dola_layers", type=str, default=None,
                   help="DoLa layer spec for B5 (e.g., 'low')")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    run(cfg, args)


if __name__ == "__main__":
    main()
