"""Output parsing, format checking, base_id, and mismatch diagnostics.

The primary parser is ``parse_number`` which uses the hierarchical
deterministic strategy (B) from ``parsers.py``: tag → answer-line →
last-number, with currency normalization.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .parsers import parse_hierarchical

FORMAT_RE = re.compile(r"^-?\d+\.\d{2}$")


def compute_base_id(item: dict) -> str:
    """Deterministic pairing key across ICL conditions for the same base task."""
    raw = (
        item["template_id"]
        + item["rendered_context"]
        + item["rendered_question"]
        + item["truth_formatted"]
    )
    return hashlib.sha1(raw.encode()).hexdigest()


def check_format(text: str) -> bool:
    """True if stripped text is exactly ``digits.dd`` (two-decimal, no commas)."""
    return bool(FORMAT_RE.match(text.strip()))


def parse_number(text: str) -> tuple[float | None, bool]:
    """Extract a numeric value using hierarchical last-number strategy."""
    result = parse_hierarchical(text, prefer_last=True)
    return result["parsed_value"], result["parse_ok"]


def diagnose_parse(
    trace: dict[str, Any],
    truth_value: float,
    anchor_value: float | None = None,
    collision_tol: float = 0.01,
    mismatch_ratio: float = 10.0,
) -> dict[str, Any]:
    """Compute mismatch diagnostics from a parse trace.

    Returns:
      anchor_collision  — parsed value ≈ anchor value (within *collision_tol*)
      chosen_abs_error  — |parsed − truth|
      best_candidate_error — min |candidate − truth| over all candidates
      mismatch_suspect  — a better candidate exists but the parser missed it
    """
    parsed = trace["parsed_value"]
    cands = trace.get("candidates", [])

    chosen_err = abs(parsed - truth_value) if parsed is not None else None

    best_err: float | None = None
    if cands:
        best_err = min(abs(c["value"] - truth_value) for c in cands)

    collision = False
    if parsed is not None and anchor_value is not None:
        collision = abs(parsed - anchor_value) < collision_tol

    suspect = False
    if (
        chosen_err is not None
        and best_err is not None
        and best_err < 1.0
        and chosen_err > mismatch_ratio * max(best_err, 0.01)
    ):
        suspect = True

    return {
        "anchor_collision": collision,
        "chosen_abs_error": chosen_err,
        "best_candidate_error": best_err,
        "mismatch_suspect": suspect,
    }
