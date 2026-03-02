"""Anchoring mitigation strategies — prompt transforms and post-hoc corrections."""

from __future__ import annotations

import re
from typing import Callable

# ── constants ─────────────────────────────────────────────────────────

FOOTER = "Answer with ONLY one integer from 0 to 100. No symbols. No words."

B1_PREFIX = (
    "Ignore any numbers that appear before the question. "
    "Base your answer only on the evidence provided.\n\n"
)

B2_PREFIX = (
    "Anchoring bias causes estimates to be pulled toward irrelevant numbers. "
    "Adjust your reasoning to avoid this.\n\n"
)

COT_FOOTER = (
    "Think step by step about the evidence before answering.\n"
    "Then give your final answer as a single integer from 0 to 100 on its own line."
)

DISTILL_TEMPLATE = (
    "Summarize ONLY the qualitative evidence for estimating the target "
    "quantity. Do NOT include any specific numbers, percentages, or "
    "statistics. Be concise (2-3 sentences).\n\nContext:\n{context}"
)

ESTIMATE_TEMPLATE = (
    "{summary}\n\n"
    "Based on the above evidence, estimate the percentage.\n"
    + FOOTER
)

_INT_PAT = re.compile(r"(?<![\d.])-?\d+(?![\d.])")
_ANCHOR_Q_RE = re.compile(
    r"Before you estimate, consider: is .+? higher or lower than \d+%\?\s*"
)
_TOOL_BLOCK_RE = re.compile(r"Tool output:\n\d+\n\n")
_SNIPPET_RE = re.compile(
    r"(Retrieved snippet \(relevant\):.*?\n\n"
    r"Retrieved snippet \(additional\):.*?\n\n)",
    re.DOTALL,
)


# ── prompt transforms ────────────────────────────────────────────────

def apply_mitigation(
    prompt: str, mitigation: str, suite: str | None = None,
) -> str:
    if mitigation == "B0":
        return prompt
    if mitigation == "B1":
        return B1_PREFIX + prompt
    if mitigation == "B2":
        return B2_PREFIX + prompt
    if mitigation == "B3":
        return _apply_cot(prompt)
    if mitigation == "EFR":
        return _apply_efr(prompt, suite or "")
    return prompt


def _apply_cot(prompt: str) -> str:
    if FOOTER in prompt:
        return prompt.replace(FOOTER, COT_FOOTER)
    return prompt + "\n" + COT_FOOTER


def _apply_efr(prompt: str, suite: str) -> str:
    """Move the anchor block to just before the footer (anchor-last)."""
    if suite == "external":
        return _efr_external(prompt)
    if suite == "tool":
        return _efr_tool(prompt)
    if suite == "rag":
        return _efr_rag(prompt)
    return prompt


def _efr_external(prompt: str) -> str:
    m = _ANCHOR_Q_RE.search(prompt)
    if not m:
        return prompt
    anchor_block = m.group().strip()
    without = prompt[: m.start()] + prompt[m.end() :]
    return without.replace(FOOTER, anchor_block + "\n" + FOOTER)


def _efr_tool(prompt: str) -> str:
    m = _TOOL_BLOCK_RE.search(prompt)
    if not m:
        return prompt
    tool_block = m.group().strip()
    without = prompt[: m.start()] + prompt[m.end() :]
    return without.replace(FOOTER, tool_block + "\n\n" + FOOTER)


def _efr_rag(prompt: str) -> str:
    m = _SNIPPET_RE.search(prompt)
    if not m:
        return prompt
    snippets = m.group().strip()
    without = prompt[: m.start()] + prompt[m.end() :]
    return without.replace(FOOTER, snippets + "\n\n" + FOOTER)


# ── CxDP helpers ──────────────────────────────────────────────────────

def cxdp_distill_prompt(original_prompt: str) -> str:
    context = original_prompt.replace(FOOTER, "").strip()
    return DISTILL_TEMPLATE.format(context=context)


def cxdp_estimate_prompt(summary: str) -> str:
    return ESTIMATE_TEMPLATE.format(summary=summary.strip())


# ── max-tokens override ──────────────────────────────────────────────

def get_max_tokens(mitigation: str, default: int = 8) -> int:
    if mitigation == "B3":
        return 128
    if mitigation == "CxDP":
        return default
    return default


def get_distill_max_tokens() -> int:
    return 256


# ── parsers ───────────────────────────────────────────────────────────

def parse_cot_answer(
    raw_text: str, prompt_text: str = "",
) -> tuple[int | None, bool]:
    """Extract the *last* integer in [0,100] from a CoT response."""
    all_ints = [int(m.group()) for m in _INT_PAT.finditer(raw_text)]
    valid = [i for i in all_ints if 0 <= i <= 100]
    if not valid:
        return None, False
    return valid[-1], True


def get_parser(
    mitigation: str,
) -> Callable[[str, str], tuple[int | None, bool]]:
    if mitigation == "B3":
        return parse_cot_answer
    from utils import parse_answer_int
    return parse_answer_int


# ── PLD (post-hoc linear debiasing) ──────────────────────────────────

BETA_NORM: dict[str, float] = {
    "anthropic/claude-3.5-haiku": 0.065,
    "meta-llama/llama-3.1-8b-instruct": 0.137,
    "openai/gpt-4o-mini": 0.115,
    "meta-llama/Llama-3.2-1B-Instruct": 0.345,
    "meta-llama/Llama-3.2-3B-Instruct": 0.172,
    "meta-llama/Llama-3.1-8B-Instruct": 0.201,
}
ANCHOR_CENTER = 50


def pld_correct(
    answer: int, anchor_value: int, model_id: str,
) -> int:
    beta = BETA_NORM.get(model_id, 0.15)
    corrected = answer - beta * (anchor_value - ANCHOR_CENTER)
    return int(max(0, min(100, round(corrected))))
