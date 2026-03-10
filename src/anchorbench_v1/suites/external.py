"""External-anchor suite: anchor is a normatively irrelevant numeric cue
injected as an "industry report" or "recent study" sentence.

v1: 3 conditions (control / low_anchor / high_anchor)
v2: 5 conditions (control / irrelevant_low / irrelevant_high
                           / plausible_low  / plausible_high)

Pairing invariant: all conditions share the same item stem; only the
anchor-bearing sentence changes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..domains import DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView

_MISSING_VALUE_DISPLAY = "[data not available]"


def _format_evidence(spec: ItemSpec) -> str:
    lines = []
    for e in spec.evidence_structured:
        lines.append(f"  - {e['label']}: {e['value']}")
    return "\n".join(lines)


def _format_evidence_v2(spec: ItemSpec) -> str:
    """Format evidence, rendering missing values for hard items."""
    lines = []
    for e in spec.evidence_structured:
        if e.get("missing"):
            lines.append(f"  - {e['label']}: {_MISSING_VALUE_DISPLAY}")
        else:
            lines.append(f"  - {e['label']}: {e['value']}")
    return "\n".join(lines)


def _build_prompt(
    spec: ItemSpec,
    condition: str,
    anchor_value: Optional[int],
) -> PromptView:
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    scenario = spec.scenario_text or dcfg.scenario_templates[tidx % len(dcfg.scenario_templates)]
    question = dcfg.question_templates[tidx % len(dcfg.question_templates)]
    evidence_block = _format_evidence(spec)

    components = {
        "scenario": scenario,
        "evidence": evidence_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
    }

    base_text = f"{scenario}\n\nEvidence:\n{evidence_block}\n\n"

    anchor_string = None
    anchor_span = None

    if condition == "control":
        prompt_text = f"{base_text}{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    else:
        anchor_sentence = dcfg.anchor_preamble.format(anchor=anchor_value)
        components["anchor_sentence"] = anchor_sentence
        anchor_string = str(anchor_value)

        prompt_text = (
            f"{base_text}{anchor_sentence} "
            f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
        )
        start = prompt_text.find(str(anchor_value))
        if start >= 0:
            anchor_span = [start, start + len(str(anchor_value))]

    return PromptView(
        item_id=spec.item_id,
        suite=spec.suite,
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text,
        prompt_components=components,
        anchor_string=anchor_string,
        anchor_span=anchor_span,
    )


def render_external(spec: ItemSpec) -> List[PromptView]:
    """Render control / low_anchor / high_anchor for an external-suite item."""
    return [
        _build_prompt(spec, "control", None),
        _build_prompt(spec, "low_anchor", spec.anchors["low"]),
        _build_prompt(spec, "high_anchor", spec.anchors["high"]),
    ]


# ── External v2 renderer ────────────────────────────────────────────

_V2_CONDITIONS: List[Tuple[str, str, str]] = [
    # (condition_name, relevance_type, anchor_direction)
    ("control",         "none",       ""),
    ("irrelevant_low",  "irrelevant", "low"),
    ("irrelevant_high", "irrelevant", "high"),
    ("plausible_low",   "plausible",  "low"),
    ("plausible_high",  "plausible",  "high"),
]


def _build_prompt_v2(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: Optional[int],
) -> PromptView:
    """Build a single PromptView for External v2."""
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    scenario = (
        spec.scenario_text
        or dcfg.scenario_templates[tidx % len(dcfg.scenario_templates)]
    )
    question = dcfg.question_templates[tidx % len(dcfg.question_templates)]
    evidence_block = _format_evidence_v2(spec)

    components = {
        "scenario": scenario,
        "evidence": evidence_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
    }

    base_text = f"{scenario}\n\nEvidence:\n{evidence_block}\n\n"

    anchor_string: Optional[str] = None
    anchor_span: Optional[List[int]] = None

    if condition == "control":
        prompt_text = f"{base_text}{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    else:
        preamble_template = dcfg.anchor_preambles[relevance]
        anchor_sentence = preamble_template.format(anchor=anchor_value)
        components["anchor_sentence"] = anchor_sentence
        anchor_string = str(anchor_value)

        prompt_text = (
            f"{base_text}{anchor_sentence} "
            f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
        )
        start = prompt_text.find(anchor_string, len(base_text))
        if start >= 0:
            anchor_span = [start, start + len(anchor_string)]

    return PromptView(
        item_id=spec.item_id,
        suite=spec.suite,
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text,
        prompt_components=components,
        anchor_string=anchor_string,
        anchor_span=anchor_span,
        anchor_relevance=relevance,
        anchor_value=anchor_value,
    )


def render_external_v2(spec: ItemSpec) -> List[PromptView]:
    """Render 5 conditions for an External v2 item.

    Conditions: control, irrelevant_low, irrelevant_high,
                plausible_low, plausible_high.
    All share the same item stem; only the anchor sentence differs.
    """
    views: List[PromptView] = []
    for cond_name, relevance, direction in _V2_CONDITIONS:
        if direction:
            anchor_val = spec.anchors[direction]
        else:
            anchor_val = None
        views.append(_build_prompt_v2(spec, cond_name, relevance, anchor_val))
    return views
