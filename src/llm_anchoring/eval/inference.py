"""HuggingFace model loading and batch generation."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model(
    model_id: str, dtype: str = "auto"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer, configured for left-padded batch generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("loaded %s  dtype=%s  device=%s", model_id, dtype, model.device)
    return model, tokenizer


def prepare_prompt(
    prompt: str, tokenizer: AutoTokenizer, use_chat_template: bool
) -> str:
    """Optionally wrap a raw prompt in the tokenizer's chat template."""
    if use_chat_template:
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return prompt


def build_gen_kwargs(decoding: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    """Translate a decoding config dict into ``model.generate`` kwargs."""
    kw: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    temp = decoding.get("temperature", 0)
    if temp == 0:
        kw["do_sample"] = False
    else:
        kw["do_sample"] = True
        kw["temperature"] = temp
        kw["top_p"] = decoding.get("top_p", 0.95)
    return kw


def generate_batch(
    prompts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    gen_kwargs: dict[str, Any],
) -> list[str]:
    """Tokenize, generate, and decode a batch. Returns new-token text only."""
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    kw = {**gen_kwargs}
    if "pad_token_id" not in kw:
        kw["pad_token_id"] = tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    return tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
