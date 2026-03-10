"""Map integers 0-100 to token IDs for a given tokenizer."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def build_int_vocab(tokenizer: Any) -> dict[int, list[int]]:
    """Return {integer: [token_ids]} for y in 0..100.

    For single-token integers the list has length 1.
    For multi-token (e.g. "100" -> ["1","00"]) the full token sequence is stored.
    """
    vocab: dict[int, list[int]] = {}
    for y in range(101):
        token_ids = tokenizer.encode(str(y), add_special_tokens=False)
        vocab[y] = token_ids
    single = sum(1 for v in vocab.values() if len(v) == 1)
    log.info("int_vocab: %d/101 single-token, %d multi-token", single, 101 - single)
    return vocab


def single_token_ids(int_vocab: dict[int, list[int]]) -> dict[int, int]:
    """Return {integer: first_token_id} for all 0-100."""
    return {y: tids[0] for y, tids in int_vocab.items()}
