"""Parsing strategies for numeric extraction from LLM outputs.

Strategies ordered by invasiveness:
  A  strict       — exact format match only
  B  hierarchical — tag → bottom-up answer-line → equation RHS → last-number
  C  llm_extract  — small-model extractor on completion-only text (fallback)

Every parse result includes a full *trace*: candidate list with character
spans, chosen index, and candidate count — enabling mismatch diagnostics.
"""

from __future__ import annotations

import re
from typing import Any, Callable

FORMAT_RE = re.compile(r"^-?\d+\.\d{2}$")
CANDIDATE_RE = re.compile(
    r"\$?\s*-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)
VALUE_TAG_RE = re.compile(r"<value>\s*([^<]+?)\s*</value>", re.IGNORECASE)
ANSWER_KW = re.compile(
    r"\b(?:answer|total|final|result|amount\s*due|due)\b", re.IGNORECASE
)


def _find_candidates(text: str) -> list[dict[str, Any]]:
    """Find all numeric tokens in *text* with spans and normalized values."""
    out: list[dict[str, Any]] = []
    for m in CANDIDATE_RE.finditer(text):
        raw = m.group()
        clean = raw.replace("$", "").replace(",", "").strip()
        if not clean or clean == "-":
            continue
        try:
            val = float(clean)
        except ValueError:
            continue
        out.append({"value": val, "span": [m.start(), m.end()], "raw": raw})
    return out


def _line_spans(text: str) -> list[tuple[str, int, int]]:
    """Split *text* into ``(line_text, start_offset, end_offset)`` tuples."""
    spans = []
    offset = 0
    for line in text.split("\n"):
        spans.append((line, offset, offset + len(line)))
        offset += len(line) + 1
    return spans


def _cands_in_range(
    cands: list[dict], start: int, end: int
) -> list[int]:
    """Return indices of candidates whose span starts within [start, end)."""
    return [i for i, c in enumerate(cands) if start <= c["span"][0] < end]


def _trace(
    value: float | None,
    strategy: str,
    candidates: list[dict],
    chosen_idx: int | None,
    fmt_ok: bool = False,
) -> dict[str, Any]:
    return {
        "parsed_value": value,
        "parse_ok": value is not None,
        "format_ok": fmt_ok,
        "strategy_used": strategy,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "chosen_idx": chosen_idx,
    }


# ── Strategy A ───────────────────────────────────────────────────────

def parse_strict(text: str) -> dict[str, Any]:
    """Accept only if stripped output is exactly ``digits.dd``."""
    s = text.strip()
    cands = _find_candidates(s)
    if FORMAT_RE.match(s):
        return _trace(float(s), "A_strict", cands, 0 if cands else None, fmt_ok=True)
    return _trace(None, "A_strict", cands, None)


# ── Strategy B ───────────────────────────────────────────────────────

def parse_hierarchical(text: str, *, prefer_last: bool = True) -> dict[str, Any]:
    """Bottom-up deterministic hierarchy with candidate tracking.

    Levels:
      B0  strict format
      B1  ``<value>`` tag
      B2  answer-keyword line, scanned **bottom-up** (last match first)
      B3  equation RHS (line with '='), scanned **bottom-up**
      B4  last (or first) numeric token in entire completion
    """
    stripped = text.strip()
    cands = _find_candidates(stripped)

    # B0: exact strict format
    if FORMAT_RE.match(stripped):
        return _trace(float(stripped), "B0_strict", cands,
                       0 if cands else None, fmt_ok=True)

    # B1: <value> tag
    tag = VALUE_TAG_RE.search(stripped)
    if tag:
        idxs = _cands_in_range(cands, tag.start(1), tag.end(1))
        if idxs:
            return _trace(cands[idxs[-1]]["value"], "B1_tag", cands, idxs[-1])

    lines = _line_spans(stripped)

    # B2: bottom-up answer-keyword line
    for line_text, ls, le in reversed(lines):
        if ANSWER_KW.search(line_text):
            idxs = _cands_in_range(cands, ls, le)
            if idxs:
                ci = idxs[-1]
                return _trace(cands[ci]["value"], "B2_answer_bottom",
                               cands, ci)

    # B3: equation RHS, bottom-up
    for line_text, ls, le in reversed(lines):
        eq_pos = line_text.rfind("=")
        if eq_pos >= 0:
            rhs_start = ls + eq_pos + 1
            idxs = _cands_in_range(cands, rhs_start, le)
            if idxs:
                ci = idxs[-1]
                return _trace(cands[ci]["value"], "B3_equation_rhs",
                               cands, ci)

    # B4: last/first number
    if cands:
        ci = -1 if prefer_last else 0
        label = "B4_last" if prefer_last else "B4_first"
        return _trace(cands[ci]["value"], label, cands, ci)

    return _trace(None, "B_none", [], None)


# ── Strategy C ───────────────────────────────────────────────────────

_EXTRACTOR_PROMPT = (
    "Extract the single final numeric answer from the text below.\n"
    "Return ONLY the number inside <value></value> tags.\n"
    "If no clear answer exists, return <value>NONE</value>.\n\n"
    "Text: {completion}\n\nYour response:"
)


def parse_llm_extract(
    text: str,
    extractor_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Small-LLM extractor on completion-only text (never sees the prompt)."""
    if extractor_fn is None:
        return parse_hierarchical(text)

    raw = extractor_fn(_EXTRACTOR_PROMPT.format(completion=text.strip()))
    resp_cands = _find_candidates(raw)

    tag = VALUE_TAG_RE.search(raw)
    if tag:
        inner = tag.group(1).strip()
        if inner.upper() == "NONE":
            return _trace(None, "C_llm_none", resp_cands, None)
        ts, te = tag.start(1), tag.end(1)
        idxs = [i for i, c in enumerate(resp_cands) if ts <= c["span"][0] < te]
        if idxs:
            ci = idxs[-1]
            return _trace(resp_cands[ci]["value"], "C_llm_tag", resp_cands, ci)

    if resp_cands:
        return _trace(resp_cands[-1]["value"], "C_llm_fallback", resp_cands, -1)
    return _trace(None, "C_llm_fail", [], None)
