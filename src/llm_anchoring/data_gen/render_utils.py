"""Formatting and rendering helpers for numeric display values."""

from __future__ import annotations

import random
from typing import Any


def currency_us(value: float) -> str:
    """Format *value* as US currency: ``$1,234.56``."""
    return f"${value:,.2f}"


def plain_2dp(value: float) -> str:
    """Format *value* as plain digits with exactly two decimals, no commas."""
    return f"{value:.2f}"


def sample_variables(
    ranges: dict[str, dict[str, Any]], rng: random.Random
) -> dict[str, float]:
    """Sample one value per variable from the ranges spec.

    Each entry in *ranges* maps a variable name to
    ``{"min": ..., "max": ..., "type": "int"|"float", "decimals": ...}``.
    """
    out: dict[str, float] = {}
    for name, spec in ranges.items():
        lo, hi = spec["min"], spec["max"]
        if spec.get("type") == "int":
            out[name] = float(rng.randint(int(lo), int(hi)))
        else:
            decimals = spec.get("decimals", 2)
            raw = rng.uniform(lo, hi)
            out[name] = round(raw, decimals)
    return out


def render_display_values(
    variables: dict[str, float],
    format_style: str = "currency_us",
) -> dict[str, str]:
    """Create ``{VAR}_DISPLAY`` strings for every numeric variable."""
    fmt = currency_us if format_style == "currency_us" else plain_2dp
    return {f"{k}_DISPLAY": fmt(v) for k, v in variables.items()}


def build_demo_text(
    anchor_raw: float | None,
    *,
    placeholder_mode: bool = False,
) -> tuple[str, str]:
    """Return (demo_question, demo_answer) for a single ICL shot.

    In *placeholder_mode* no digits appear at all.
    """
    if placeholder_mode:
        q = (
            "Format the amount <AMOUNT> as plain digits with exactly "
            "two decimals (no commas, no currency symbol). "
            "Output only the number."
        )
        a = "<AMOUNT_PLAIN>"
        return q, a

    assert anchor_raw is not None
    q = (
        f"Format the amount {currency_us(anchor_raw)} as plain digits "
        "with exactly two decimals (no commas, no currency symbol). "
        "Output only the number."
    )
    a = plain_2dp(anchor_raw)
    return q, a


def build_icl_block(
    demos: list[tuple[str, str]],
) -> str:
    """Assemble a multi-shot ICL block from (question, answer) pairs."""
    parts: list[str] = []
    for i, (q, a) in enumerate(demos, 1):
        parts.append(f"Example {i}:\nQ: {q}\nA: {a}")
    return "\n\n".join(parts)
