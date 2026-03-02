#!/usr/bin/env python3
"""Run PoC anchoring benchmark with local HF models.

Usage:
  python runner/run_open_models.py --config runner/config.example.yaml
  python runner/run_open_models.py --config runner/config.example.yaml --logprob_ev
  python runner/run_open_models.py --config runner/config.example.yaml --dry_run
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    append_jsonl,
    load_completed,
    load_dataset,
    make_result,
    parse_answer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── model helpers ─────────────────────────────────────────────────────

def _load_model(model_id: str, dtype: str = "bfloat16"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = getattr(torch, dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    log.info("Loaded %s  dtype=%s  device=%s", model_id, dtype, model.device)
    return model, tokenizer


def _wrap_chat(prompt: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_batch(
    prompts: list[str],
    model,
    tokenizer,
    gen_kwargs: dict,
    batch_size: int = 16,
) -> list[str]:
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
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


# ── logprob expected value ────────────────────────────────────────────

def _compute_soft_ev(
    chat_prompt: str,
    model,
    tokenizer,
    chunk: int = 52,
) -> int:
    """Teacher-force each candidate answer 0..100 and return SoftEV."""
    enc = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc.input_ids[0]
    prompt_len = len(prompt_ids)

    all_ans_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(101)]
    max_ans_len = max(len(a) for a in all_ans_ids)

    logprobs = np.empty(101, dtype=np.float64)

    for start in range(0, 101, chunk):
        end = min(start + chunk, 101)
        batch_ids, batch_mask = [], []
        for idx in range(start, end):
            aids = all_ans_ids[idx]
            pad_len = max_ans_len - len(aids)
            full = torch.cat([
                prompt_ids,
                torch.tensor(aids, dtype=torch.long),
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long),
            ])
            mask = torch.cat([
                torch.ones(prompt_len + len(aids), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
            batch_ids.append(full)
            batch_mask.append(mask)

        input_ids = torch.stack(batch_ids).to(model.device)
        attention_mask = torch.stack(batch_mask).to(model.device)
        with torch.no_grad():
            out = model(input_ids, attention_mask=attention_mask)

        for k, idx in enumerate(range(start, end)):
            aids = all_ans_ids[idx]
            lp = 0.0
            for j, tok_id in enumerate(aids):
                pos = prompt_len + j - 1
                log_p = torch.log_softmax(out.logits[k, pos].float(), dim=-1)
                lp += log_p[tok_id].item()
            logprobs[idx] = lp

    probs = np.exp(logprobs - logprobs.max())
    probs /= probs.sum()
    return int(round(float(np.dot(np.arange(101), probs))))


# ── main loop ─────────────────────────────────────────────────────────

def run(cfg: dict, logprob_ev: bool = False, dry_run: bool = False) -> None:
    run_name = cfg["run_name"]
    items = load_dataset(cfg["dataset"])
    hf_cfg = cfg["hf"]
    models = hf_cfg["models"]
    n_samples = cfg["sampling"]["n_samples_per_item"]
    batch_size = hf_cfg.get("batch_size", 16)

    samp = cfg["sampling"]
    gen_kwargs = {
        "do_sample": True,
        "temperature": samp["temperature"],
        "top_p": samp["top_p"],
        "max_new_tokens": samp["max_tokens"],
    }

    out_dir = Path("runner_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_hf.jsonl"

    completed, turn1_answers = load_completed(out_path)
    log.info("Resume: %d completed records", len(completed))

    if dry_run:
        _dry_run(items, models, n_samples, completed, logprob_ev)
        return

    for model_id in models:
        log.info("=== Loading %s ===", model_id)
        model, tokenizer = _load_model(model_id, hf_cfg.get("dtype", "bfloat16"))

        fh = open(out_path, "a", encoding="utf-8")
        written = 0

        try:
            for si in range(n_samples):
                torch.manual_seed(42 + si)
                log.info("  [%s] sample %d/%d", model_id.split("/")[-1], si + 1, n_samples)

                # ── Pass 1: independent prompts ──
                pass1_jobs: list[tuple[dict, str, str]] = []  # (item, condition, chat_prompt)

                for item in items:
                    iid = item["item_id"]
                    if item["suite"] == "external":
                        for cond in ("control", "low_anchor", "high_anchor"):
                            if (model_id, iid, cond, si) in completed:
                                continue
                            pass1_jobs.append((item, cond, _wrap_chat(item["prompts"][cond], tokenizer)))

                    elif item["suite"] == "self_generated":
                        if (model_id, iid, "turn1", si) not in completed:
                            pass1_jobs.append((item, "turn1", _wrap_chat(item["prompts"]["turn1"], tokenizer)))
                        if (model_id, iid, "turn2_fresh", si) not in completed:
                            pass1_jobs.append((item, "turn2_fresh", _wrap_chat(item["prompts"]["turn2_fresh"], tokenizer)))

                        for ch in item["prompts"].get("turn2_controlled_history", []):
                            cond = f"turn2_controlled_history_{ch['anchor']}"
                            if (model_id, iid, cond, si) not in completed:
                                pass1_jobs.append((item, cond, _wrap_chat(ch["prompt"], tokenizer)))

                if pass1_jobs:
                    prompts_p1 = [j[2] for j in pass1_jobs]
                    outputs_p1 = _generate_batch(prompts_p1, model, tokenizer, gen_kwargs, batch_size)

                    for (item, cond, _), raw in zip(pass1_jobs, outputs_p1):
                        ans, ok = parse_answer(raw)
                        rec = make_result(run_name, model_id, "hf", item, cond, si, ans, ok, raw)
                        append_jsonl(fh, rec)
                        completed.add((model_id, item["item_id"], cond, si))
                        if cond == "turn1":
                            turn1_answers[(model_id, item["item_id"], si)] = ans
                        written += 1

                # ── Pass 2: turn2_history (depends on turn1) ──
                pass2_jobs: list[tuple[dict, str]] = []  # (item, chat_prompt)
                pass2_null: list[dict] = []

                for item in items:
                    if item["suite"] != "self_generated":
                        continue
                    iid = item["item_id"]
                    if (model_id, iid, "turn2_history", si) in completed:
                        continue
                    a1 = turn1_answers.get((model_id, iid, si))
                    if a1 is not None:
                        prompt = item["prompts"]["turn2_history"].replace("{{A1}}", str(a1))
                        pass2_jobs.append((item, _wrap_chat(prompt, tokenizer)))
                    else:
                        pass2_null.append(item)

                if pass2_jobs:
                    prompts_p2 = [j[1] for j in pass2_jobs]
                    outputs_p2 = _generate_batch(prompts_p2, model, tokenizer, gen_kwargs, batch_size)
                    for (item, _), raw in zip(pass2_jobs, outputs_p2):
                        ans, ok = parse_answer(raw)
                        rec = make_result(run_name, model_id, "hf", item, "turn2_history", si, ans, ok, raw)
                        append_jsonl(fh, rec)
                        completed.add((model_id, item["item_id"], "turn2_history", si))
                        written += 1

                for item in pass2_null:
                    rec = make_result(run_name, model_id, "hf", item, "turn2_history", si, None, False, "")
                    append_jsonl(fh, rec)
                    completed.add((model_id, item["item_id"], "turn2_history", si))
                    written += 1

            # ── Optional: logprob SoftEV (deterministic, single pass) ──
            if logprob_ev:
                log.info("  [%s] computing SoftEV logprobs ...", model_id.split("/")[-1])
                for item in items:
                    iid = item["item_id"]
                    prompts_map: list[tuple[str, str]] = []

                    if item["suite"] == "external":
                        for cond in ("control", "low_anchor", "high_anchor"):
                            sev_cond = f"{cond}_softEV"
                            if (model_id, iid, sev_cond, 0) in completed:
                                continue
                            prompts_map.append((sev_cond, item["prompts"][cond]))
                    elif item["suite"] == "self_generated":
                        for cond in ("turn1", "turn2_fresh"):
                            sev_cond = f"{cond}_softEV"
                            if (model_id, iid, sev_cond, 0) in completed:
                                continue
                            prompts_map.append((sev_cond, item["prompts"][cond]))
                        for ch in item["prompts"].get("turn2_controlled_history", []):
                            sev_cond = f"turn2_controlled_history_{ch['anchor']}_softEV"
                            if (model_id, iid, sev_cond, 0) in completed:
                                continue
                            prompts_map.append((sev_cond, ch["prompt"]))

                    for sev_cond, raw_prompt in prompts_map:
                        chat_p = _wrap_chat(raw_prompt, tokenizer)
                        sev = _compute_soft_ev(chat_p, model, tokenizer)
                        rec = make_result(run_name, model_id, "hf", item, sev_cond, 0,
                                          sev, True, f"softEV={sev}")
                        append_jsonl(fh, rec)
                        completed.add((model_id, iid, sev_cond, 0))
                        written += 1

                log.info("  SoftEV done")

        finally:
            fh.close()

        log.info("  Wrote %d records for %s", written, model_id)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    log.info("Finished → %s", out_path)


def _dry_run(items, models, n_samples, completed, logprob_ev):
    n_ext = sum(1 for it in items if it["suite"] == "external")
    n_self = sum(1 for it in items if it["suite"] == "self_generated")
    n_ctrl = sum(
        len(it["prompts"].get("turn2_controlled_history", []))
        for it in items if it["suite"] == "self_generated"
    )

    calls_ext = n_ext * 3 * n_samples
    calls_self = n_self * 3 * n_samples
    calls_ctrl = n_ctrl * n_samples
    total = (calls_ext + calls_self + calls_ctrl) * len(models)
    if logprob_ev:
        lp_calls = (n_ext * 3 + n_self * 2 + n_ctrl) * len(models)
        total += lp_calls

    log.info("DRY RUN (HF)")
    log.info("  Models:          %s", models)
    log.info("  Items:           %d external, %d self-gen", n_ext, n_self)
    log.info("  Controlled-hist: %d prompts", n_ctrl)
    log.info("  Samples/item:    %d", n_samples)
    log.info("  LogprobEV:       %s", logprob_ev)
    log.info("  Total generations: ~%d  (already done: %d)", total, len(completed))


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("runner/config.example.yaml"))
    p.add_argument("--logprob_ev", action="store_true",
                   help="Compute SoftEV via teacher-forced logprobs over 0..100")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    run(cfg, logprob_ev=args.logprob_ev, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
