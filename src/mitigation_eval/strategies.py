"""New mitigation strategies: SELFHELP, COUNTERFACTUAL_ENSEMBLE, LATE_BINDING.

Also re-exports existing mitigations (B0-B3, EFR, CxDP) from mitigation/strategies.py.
"""

from __future__ import annotations

import re
from typing import Any

FOOTER = "Answer with ONLY one integer from 0 to 100. No symbols. No words."

# ── SELFHELP ─────────────────────────────────────────────────────────

SELFHELP_REWRITE_INSTRUCTION = (
    "Rewrite the following prompt to remove any potentially biasing "
    "numerical anchors or leading information, while preserving the "
    "core question and all factual evidence. Return ONLY the rewritten prompt, "
    "nothing else.\n\n"
    "Original prompt:\n{prompt}"
)


def selfhelp_rewrite_prompt(anchored_prompt: str) -> str:
    """Pass 1: generate a prompt asking the model to debiase the input."""
    return SELFHELP_REWRITE_INSTRUCTION.format(prompt=anchored_prompt)


def selfhelp_answer_prompt(rewritten_prompt: str) -> str:
    """Pass 2: use the model's rewritten prompt as-is for answering."""
    if FOOTER not in rewritten_prompt:
        return rewritten_prompt.strip() + "\n" + FOOTER
    return rewritten_prompt


# ── COUNTERFACTUAL_ENSEMBLE ──────────────────────────────────────────

DEFAULT_CF_OFFSETS = [-20, 20]


def counterfactual_prompts(
    anchored_prompt: str,
    anchor_string: str,
    anchor_span: list[int],
    offsets: list[int] | None = None,
) -> list[str]:
    """Generate K counterfactual prompts by replacing the anchor value.

    Each prompt substitutes the anchor number with (anchor + offset),
    clamped to [0, 100].
    """
    if offsets is None:
        offsets = DEFAULT_CF_OFFSETS

    try:
        anchor_val = int(anchor_string)
    except (ValueError, TypeError):
        return [anchored_prompt]

    start, end = anchor_span
    prefix = anchored_prompt[:start]
    suffix = anchored_prompt[end:]

    prompts = []
    for off in offsets:
        new_val = max(0, min(100, anchor_val + off))
        prompts.append(prefix + str(new_val) + suffix)
    return prompts


def counterfactual_aggregate(answers: list[int | None]) -> tuple[int | None, float]:
    """Aggregate counterfactual answers by median. Returns (median, IQR)."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0
    import numpy as np
    arr = np.array(valid)
    median = int(np.median(arr))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    return median, iqr


# ── LATE_BINDING ─────────────────────────────────────────────────────

LATE_BINDING_PASS1_SUFFIX = (
    "\nNote: Some information in the context above has been [REDACTED]. "
    "Based on the available evidence, provide your best estimate as a single "
    "integer 0-100, along with a brief confidence interval (e.g., '60 [50-70]')."
)

LATE_BINDING_PASS2_PREFIX = (
    "You previously estimated {prev_answer} for this question. "
    "The redacted information was: {anchor_info}. "
    "Update your estimate ONLY if this information is independently justified "
    "by the evidence. Otherwise keep your original estimate.\n\n"
)


def late_binding_pass1(
    anchored_prompt: str,
    anchor_string: str,
    anchor_span: list[int],
) -> str:
    """Pass 1: Replace anchor with [REDACTED], ask for estimate with uncertainty."""
    start, end = anchor_span
    redacted = anchored_prompt[:start] + "[REDACTED]" + anchored_prompt[end:]
    if FOOTER in redacted:
        redacted = redacted.replace(FOOTER, LATE_BINDING_PASS1_SUFFIX.strip())
    else:
        redacted += LATE_BINDING_PASS1_SUFFIX
    return redacted


def late_binding_pass2(
    original_prompt: str,
    prev_answer: str,
    anchor_string: str,
) -> str:
    """Pass 2: Reintroduce anchor, ask model to update only if justified."""
    prefix = LATE_BINDING_PASS2_PREFIX.format(
        prev_answer=prev_answer,
        anchor_info=anchor_string,
    )
    return prefix + original_prompt


# ── Existing mitigations (wrappers) ──────────────────────────────────

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

_INT_PAT = re.compile(r"(?<![\d.])-?\d+(?!\d)(?!\.\d)")


def apply_existing_mitigation(prompt: str, mitigation: str) -> str:
    """Apply B0-B3 or return unchanged."""
    if mitigation == "B0":
        return prompt
    if mitigation == "B1":
        return B1_PREFIX + prompt
    if mitigation == "B2":
        return B2_PREFIX + prompt
    if mitigation == "B3":
        if FOOTER in prompt:
            return prompt.replace(FOOTER, COT_FOOTER)
        return prompt + "\n" + COT_FOOTER
    return prompt


def parse_cot_answer(raw_text: str) -> tuple[int | None, bool]:
    """Extract the final answer integer in [0,100] from a CoT response.

    Strategy: prefer the last line's integer (models are instructed to put
    the final answer on its own line). Fall back to last valid integer.
    """
    if not raw_text:
        return None, False

    lines = [ln.strip() for ln in raw_text.strip().split("\n") if ln.strip()]
    if lines:
        last_line = lines[-1]
        last_ints = [int(m.group()) for m in _INT_PAT.finditer(last_line)]
        last_valid = [i for i in last_ints if 0 <= i <= 100]
        if len(last_valid) == 1:
            return last_valid[0], True

    all_ints = [int(m.group()) for m in _INT_PAT.finditer(raw_text)]
    valid = [i for i in all_ints if 0 <= i <= 100]
    if not valid:
        return None, False
    return valid[-1], True


# ── Strategy registry ────────────────────────────────────────────────

ALL_STRATEGIES = [
    "B0", "B1", "B2", "B3",
    "SELFHELP", "COUNTERFACTUAL_ENSEMBLE", "LATE_BINDING",
]


def get_max_tokens(strategy: str, default: int = 512,
                   backend: str = "hf") -> int:
    """Return max_tokens for a strategy.

    API models (OpenRouter) need much higher budgets because:
    - Reasoning models (GPT-5.x) consume hidden reasoning tokens
    - Claude/Gemini produce verbose analyses before giving the answer
    HF models are more concise and don't need large budgets.
    """
    if backend == "openrouter":
        if strategy in ("B3", "SELFHELP", "LATE_BINDING"):
            return 4096
        return 2048

    if strategy in ("B3", "SELFHELP", "LATE_BINDING"):
        return 1024
    return default
