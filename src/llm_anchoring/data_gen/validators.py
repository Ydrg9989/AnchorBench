"""Validation helpers for LLM-generated templates."""

from __future__ import annotations

import re
from typing import Any

DIGIT_RE = re.compile(r"\d")
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def has_digits(text: str) -> bool:
    """Return True if *text* contains any ASCII digit."""
    return bool(DIGIT_RE.search(text))


def extract_placeholders(text: str) -> set[str]:
    """Return the set of ``{NAME}`` placeholder names found in *text*."""
    return set(PLACEHOLDER_RE.findall(text))


def validate_template(
    template: dict[str, Any],
    required_placeholders: list[str],
    allow_extra: bool = False,
) -> list[str]:
    """Check a template dict and return a list of error strings (empty = OK).

    Checks:
    1. context_template and question_template are digit-free.
    2. All required_placeholders appear in at least one template field.
    3. No extra placeholders unless *allow_extra* is True.
    """
    errors: list[str] = []
    ctx = template.get("context_template", "")
    q = template.get("question_template", "")

    if has_digits(ctx):
        errors.append("context_template contains digits")
    if has_digits(q):
        errors.append("question_template contains digits")

    found = extract_placeholders(ctx) | extract_placeholders(q)
    required_set = set(required_placeholders)

    missing = required_set - found
    if missing:
        errors.append(f"missing placeholders: {sorted(missing)}")

    if not allow_extra:
        extra = found - required_set
        if extra:
            errors.append(f"extra placeholders: {sorted(extra)}")

    return errors
