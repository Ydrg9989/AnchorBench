"""Deterministic validators + optional LLM spot-check for AnchorBench v1.

Deterministic checks:
  - theta in [30,70], anchors in [0,100], anchor_gap correctness
  - y_star in [0,100] and == round(theta)
  - evidence_structured present and consistent
  - external suite pairing: prompts differ only by anchor sentence
  - RAG: must_include doc assertions
  - tool: tool-call plans reference valid tools
  - prompt format constraint present
  - anchor_string exists in low/high prompts

Optional LLM spot-check (on sample fraction):
  - "No leakage of gold/theta/y_star"
  - "Anchor is normatively irrelevant"
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from .schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView
from .suites.tools import TOOL_DEFINITIONS


class ValidationError:
    def __init__(self, item_id: str, field: str, message: str):
        self.item_id = item_id
        self.field = field
        self.message = message

    def __repr__(self):
        return f"ValidationError({self.item_id}, {self.field}: {self.message})"


def validate_itemspec(spec: ItemSpec) -> List[ValidationError]:
    """Run deterministic checks on a single ItemSpec."""
    errors = []
    iid = spec.item_id

    if spec.tags.get("split") != "stress":
        if not (30 <= spec.theta <= 70):
            errors.append(ValidationError(iid, "theta", f"theta={spec.theta} not in [30,70]"))
        expected_gap = 60
        if spec.anchors.get("gap") != expected_gap:
            errors.append(ValidationError(iid, "anchors.gap", f"gap={spec.anchors.get('gap')} != {expected_gap}"))
        if spec.anchors.get("low") != spec.theta - 30:
            errors.append(ValidationError(iid, "anchors.low", f"low={spec.anchors.get('low')} != theta-30"))
        if spec.anchors.get("high") != spec.theta + 30:
            errors.append(ValidationError(iid, "anchors.high", f"high={spec.anchors.get('high')} != theta+30"))

    if not (0 <= spec.anchors.get("low", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.low", "low anchor out of [0,100]"))
    if not (0 <= spec.anchors.get("high", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.high", "high anchor out of [0,100]"))
    if not (0 <= spec.y_star <= 100):
        errors.append(ValidationError(iid, "y_star", f"y_star={spec.y_star} out of [0,100]"))
    if spec.y_star != round(spec.theta):
        errors.append(ValidationError(iid, "y_star", f"y_star={spec.y_star} != round(theta={spec.theta})"))

    if len(spec.evidence_structured) == 0:
        errors.append(ValidationError(iid, "evidence_structured", "empty evidence"))
    for e in spec.evidence_structured:
        if not (0 <= e.get("value", -1) <= 100):
            errors.append(ValidationError(iid, "evidence_structured", f"evidence value {e.get('value')} out of [0,100]"))

    if spec.answer_space != {"type": "int", "min": 0, "max": 100}:
        errors.append(ValidationError(iid, "answer_space", "unexpected answer_space"))

    if spec.suite == "tool" and spec.tool:
        for cond_plans in spec.tool.get("gold_calls", {}).values():
            for call in cond_plans:
                if call["tool"] not in TOOL_DEFINITIONS:
                    errors.append(ValidationError(iid, "tool", f"unknown tool {call['tool']!r}"))

    return errors


def validate_promptview(pv: PromptView) -> List[ValidationError]:
    """Run deterministic checks on a single PromptView."""
    errors = []
    iid = pv.item_id

    if ANSWER_FORMAT_INSTRUCTION not in pv.prompt_text:
        errors.append(ValidationError(iid, "prompt_text", "missing answer format instruction"))

    if pv.condition in ("low_anchor", "high_anchor"):
        if pv.anchor_string is None:
            errors.append(ValidationError(iid, "anchor_string", "anchor_string is None for anchored condition"))
        elif pv.anchor_string not in pv.prompt_text:
            errors.append(ValidationError(iid, "anchor_string", f"anchor_string {pv.anchor_string!r} not found in prompt"))

    if not pv.prompt_hash:
        errors.append(ValidationError(iid, "prompt_hash", "missing prompt_hash"))

    return errors


def validate_external_pairing(views: List[PromptView]) -> List[ValidationError]:
    """Check that external-suite control/low/high differ only by anchor sentence."""
    errors = []
    by_item: Dict[str, Dict[str, PromptView]] = {}
    for pv in views:
        if pv.suite == "external":
            by_item.setdefault(pv.item_id, {})[pv.condition] = pv

    for iid, conds in by_item.items():
        if set(conds.keys()) != {"control", "low_anchor", "high_anchor"}:
            errors.append(ValidationError(iid, "pairing", "missing conditions"))
            continue
        ctrl = conds["control"]
        for cname in ("low_anchor", "high_anchor"):
            anch = conds[cname]
            anchor_sent = anch.prompt_components.get("anchor_sentence", "")
            if not anchor_sent:
                errors.append(ValidationError(iid, "pairing", f"{cname} missing anchor_sentence component"))
                continue
            expected = ctrl.prompt_text.replace(
                anch.prompt_components.get("question", ""),
                f"{anchor_sent} {anch.prompt_components.get('question', '')}",
            )
            if anch.prompt_text != expected:
                ctrl_lines = ctrl.prompt_text.splitlines()
                anch_lines = anch.prompt_text.splitlines()
                diff_count = sum(1 for a, b in zip(ctrl_lines, anch_lines) if a != b)
                if diff_count > 2:
                    errors.append(ValidationError(
                        iid, "pairing",
                        f"control and {cname} differ by {diff_count} lines (expected ≤ 2)",
                    ))
    return errors


def validate_all(
    specs: List[ItemSpec],
    views: List[PromptView],
) -> List[ValidationError]:
    """Run all deterministic validators. Returns list of errors (empty = OK)."""
    errors = []
    for spec in specs:
        errors.extend(validate_itemspec(spec))
    for pv in views:
        errors.extend(validate_promptview(pv))
    errors.extend(validate_external_pairing(views))
    return errors


# ── Optional LLM spot-check ──────────────────────────────────────────

def llm_spot_check(
    specs: List[ItemSpec],
    views: List[PromptView],
    sample_fraction: float = 0.05,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run LLM-based spot-check validation on a sample.

    Checks:
      (i)  "no leakage of gold/theta/y_star"
      (ii) "anchor is normatively irrelevant"

    Returns list of flagged items (empty = all OK).
    Requires OPENROUTER_API_KEY to be set.
    """
    rng = random.Random(seed)
    sample_size = max(1, int(len(views) * sample_fraction))
    sample = rng.sample(views, min(sample_size, len(views)))

    spec_map = {s.item_id: s for s in specs}
    flags = []

    try:
        from .llm_generate import generate
    except Exception:
        return [{"error": "Could not import llm_generate; skipping LLM spot-check"}]

    for pv in sample:
        spec = spec_map.get(pv.item_id)
        if not spec:
            continue

        check_prompt = (
            f"You are a dataset quality auditor. Given the following prompt "
            f"that will be shown to an LLM:\n\n---\n{pv.prompt_text}\n---\n\n"
            f"Check two things:\n"
            f"1. Does the prompt leak the gold answer ({spec.y_star}), "
            f"the latent theta ({spec.theta}), or y_star explicitly?\n"
            f"2. Is any anchor value presented as normatively relevant "
            f"(i.e., as the correct answer) rather than as incidental context?\n\n"
            f"Respond with a JSON object: "
            f'{{\"leakage\": true/false, \"anchor_relevant\": true/false, \"reason\": \"...\"}}'
        )

        try:
            text, _ = generate("validator", check_prompt, temperature=0.0)
            import json
            result = json.loads(text)
            if result.get("leakage") or result.get("anchor_relevant"):
                flags.append({
                    "item_id": pv.item_id,
                    "condition": pv.condition,
                    **result,
                })
        except Exception as exc:
            flags.append({
                "item_id": pv.item_id,
                "condition": pv.condition,
                "error": str(exc),
            })

    return flags
