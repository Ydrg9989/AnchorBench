"""Teacher-forced logit extraction: compute P(y | prompt) over integers 0-100."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .int_vocab import single_token_ids

log = logging.getLogger(__name__)


def wrap_chat(prompt: str, tokenizer: Any, system_prompt: str = "") -> str:
    """Apply chat template to a raw prompt string."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def score_integer_distribution(
    model: Any,
    tokenizer: Any,
    prompt: str,
    int_vocab: dict[int, list[int]],
    apply_chat: bool = True,
    system_prompt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute P(y) for y in 0..100 from next-token logits.

    Returns (probs shape (101,), raw_logits shape (101,)).
    """
    if apply_chat:
        text = wrap_chat(prompt, tokenizer, system_prompt)
    else:
        text = prompt

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        out = model(input_ids)
    last_logits = out.logits[0, -1, :].float()  # cast to fp32

    stid = single_token_ids(int_vocab)
    token_ids = torch.tensor([stid[y] for y in range(101)], device=last_logits.device)
    int_logits = last_logits[token_ids]
    int_probs = torch.softmax(int_logits, dim=0)

    return int_probs.cpu().numpy(), int_logits.cpu().numpy()


def char_to_token_span(
    prompt: str,
    char_start: int,
    char_end: int,
    tokenizer: Any,
    apply_chat: bool = True,
    system_prompt: str = "",
) -> tuple[int, int, list[int]]:
    """Convert character span in raw prompt to token indices in the tokenized sequence.

    Returns (tok_start, tok_end, input_ids_list).
    """
    if apply_chat:
        full_text = wrap_chat(prompt, tokenizer, system_prompt)
        prefix = full_text.split(prompt)[0] if prompt in full_text else ""
        offset = len(prefix)
    else:
        full_text = prompt
        offset = 0

    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

    adj_start = char_start + offset
    adj_end = char_end + offset

    tok_start, tok_end = None, None
    for i, (s, e) in enumerate(offsets):
        if s is None:
            continue
        if tok_start is None and e > adj_start:
            tok_start = i
        if s < adj_end:
            tok_end = i + 1

    if tok_start is None:
        tok_start = 0
    if tok_end is None:
        tok_end = len(input_ids)

    return tok_start, tok_end, input_ids
