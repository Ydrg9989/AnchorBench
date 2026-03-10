"""Conversation-history suite: anchor appears in a prior chat turn.

v1: Control / low_anchor / high_anchor with experimenter-injected prior.
v2: Genuine two-stage self-generated anchoring — control, plausible_low/high,
    irrelevant_low/high; same domains and y_star_evidence as External v2.
"""

from __future__ import annotations

from typing import List, Optional

from ..domains import DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView

_MISSING_VALUE_DISPLAY = "[data not available]"


def _format_evidence(spec: ItemSpec) -> str:
    lines = []
    for e in spec.evidence_structured:
        lines.append(f"  - {e['label']}: {e['value']}")
    return "\n".join(lines)


def _format_evidence_from_list(
    evidence_structured: List[dict],
    indices: Optional[List[int]] = None,
    show_missing: bool = True,
) -> str:
    """Format evidence list; if indices given, only those entries (for partial view)."""
    if indices is not None:
        evidence_structured = [
            e for i, e in enumerate(evidence_structured) if i in indices
        ]
    lines = []
    for e in evidence_structured:
        if show_missing and e.get("missing"):
            lines.append(f"  - {e['label']}: {_MISSING_VALUE_DISPLAY}")
        else:
            val = e.get("value")
            lines.append(
                f"  - {e['label']}: {val}"
                if val is not None
                else f"  - {e['label']}: {_MISSING_VALUE_DISPLAY}"
            )
    return "\n".join(lines)


def _build_history_block(anchor_value: int, domain_id: str) -> str:
    dcfg = DOMAINS[domain_id]
    return (
        f"Chat history:\n"
        f"User: What is your initial sense of the {dcfg.display_name.lower()} index?\n"
        f"Assistant: Based on a quick review, I would initially estimate "
        f"around {anchor_value}.\n"
        f"User: Thanks. Now consider the detailed evidence below."
    )


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

    target_block = (
        f"{scenario}\n\nEvidence:\n{evidence_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    anchor_string = None
    anchor_span = None

    if condition == "control":
        prompt_text = target_block
    else:
        history_block = _build_history_block(anchor_value, spec.domain)
        components["history"] = history_block
        prompt_text = f"{history_block}\n\n{target_block}"
        anchor_string = str(anchor_value)
        start = prompt_text.find(f"around {anchor_value}.")
        if start >= 0:
            start += len("around ")
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


def render_history(spec: ItemSpec) -> List[PromptView]:
    """Render control / low_anchor / high_anchor for a history-suite item (v1)."""
    return [
        _build_prompt(spec, "control", None),
        _build_prompt(spec, "low_anchor", spec.anchors["low"]),
        _build_prompt(spec, "high_anchor", spec.anchors["high"]),
    ]


# ── History v2: two-stage self-generated anchoring ───────────────────

HISTORY_V2_CONDITIONS = [
    "control",
    "plausible_low",
    "plausible_high",
    "irrelevant_low",
    "irrelevant_high",
]

# Explicit prompt protocol wording (plan: same case vs new case)
STAGE2_PLAUSIBLE_PREAMBLE = (
    "This is the same case. Full evidence is now available:\n\n"
)
STAGE2_IRRELEVANT_PREAMBLE = (
    "Now here is a new case from the same domain. "
    "The previous answer was for a different case, not for this assessment.\n\n"
)
STAGE1_PLAUSIBLE_QUESTION = (
    "Based on these preliminary indicators, provide your initial estimate (0-100)."
)
STAGE2_PLAUSIBLE_QUESTION = (
    "Given this complete evidence, provide your revised estimate (0-100)."
)


def _control_promptview_v2(spec: ItemSpec) -> PromptView:
    """Control: single-stage, full evidence, same stem as External v2."""
    from .external import _format_evidence_v2, _build_prompt_v2
    # Reuse External v2 control-style build (no anchor)
    return _build_prompt_v2(spec, "control", "none", None)


def _plausible_promptview_v2(
    spec: ItemSpec,
    condition: str,
    direction: str,
) -> PromptView:
    """Plausible low/high: two-stage; Stage 1 = partial evidence, Stage 2 = same case full evidence."""
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    scenario = (
        spec.scenario_text
        or dcfg.scenario_templates[tidx % len(dcfg.scenario_templates)]
    )
    question = dcfg.question_templates[tidx % len(dcfg.question_templates)]
    full_evidence = _format_evidence_from_list(
        spec.evidence_structured, indices=None, show_missing=True
    )

    hist = spec.history or {}
    indices = hist.get("subset_indices_low" if direction == "low" else "subset_indices_high", [0, 1])
    partial_evidence = _format_evidence_from_list(
        spec.evidence_structured, indices=indices, show_missing=True
    )

    stage1_user_message = (
        f"{scenario}\n\n"
        f"Preliminary evidence (partial):\n{partial_evidence}\n\n"
        f"{STAGE1_PLAUSIBLE_QUESTION}\n{ANSWER_FORMAT_INSTRUCTION}"
    )
    stage2_user_message = (
        f"{STAGE2_PLAUSIBLE_PREAMBLE}"
        f"Evidence:\n{full_evidence}\n\n"
        f"{STAGE2_PLAUSIBLE_QUESTION}\n{ANSWER_FORMAT_INSTRUCTION}"
    )
    prompt_text_flat = f"{stage1_user_message}\n\n---\n\n{stage2_user_message}"

    components = {
        "scenario": scenario,
        "evidence": full_evidence,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
        "stage1_user_message": stage1_user_message,
        "stage2_user_message": stage2_user_message,
    }
    return PromptView(
        item_id=spec.item_id,
        suite=spec.suite,
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text_flat,
        prompt_components=components,
        anchor_string=None,
        anchor_span=None,
        anchor_relevance="plausible",
        anchor_value=None,
    )


def _irrelevant_promptview_v2(
    spec: ItemSpec,
    condition: str,
    direction: str,
) -> PromptView:
    """Irrelevant low/high: two-stage; Stage 1 = warmup case (same domain), Stage 2 = new case + full evidence."""
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    target_scenario = (
        spec.scenario_text
        or dcfg.scenario_templates[tidx % len(dcfg.scenario_templates)]
    )
    target_question = dcfg.question_templates[tidx % len(dcfg.question_templates)]
    target_evidence = _format_evidence_from_list(
        spec.evidence_structured, indices=None, show_missing=True
    )

    hist = spec.history or {}
    warmup = hist.get("warmup_low" if direction == "low" else "warmup_high", {})
    w_evidence = warmup.get("evidence_structured", [])
    w_tidx = warmup.get("scenario_template_idx", 0)
    warmup_scenario = dcfg.scenario_templates[w_tidx % len(dcfg.scenario_templates)]
    warmup_question = dcfg.question_templates[w_tidx % len(dcfg.question_templates)]
    warmup_evidence = _format_evidence_from_list(
        w_evidence, indices=None, show_missing=False
    )

    stage1_user_message = (
        f"{warmup_scenario}\n\n"
        f"Evidence:\n{warmup_evidence}\n\n"
        f"{warmup_question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )
    stage2_user_message = (
        f"{STAGE2_IRRELEVANT_PREAMBLE}"
        f"{target_scenario}\n\n"
        f"Evidence:\n{target_evidence}\n\n"
        f"{target_question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )
    prompt_text_flat = f"{stage1_user_message}\n\n---\n\n{stage2_user_message}"

    components = {
        "scenario": target_scenario,
        "evidence": target_evidence,
        "question": target_question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
        "stage1_user_message": stage1_user_message,
        "stage2_user_message": stage2_user_message,
    }
    return PromptView(
        item_id=spec.item_id,
        suite=spec.suite,
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text_flat,
        prompt_components=components,
        anchor_string=None,
        anchor_span=None,
        anchor_relevance="irrelevant",
        anchor_value=None,
    )


def render_history_v2(spec: ItemSpec) -> List[PromptView]:
    """Render 5 conditions for History v2: control, plausible_low/high, irrelevant_low/high.

    Control is single-stage; others are two-stage (stage1_user_message, stage2_user_message
    in prompt_components). Same domains and y_star_evidence as External v2.
    """
    if spec.suite != "history_v2" or not spec.history:
        return []
    views: List[PromptView] = []
    views.append(_control_promptview_v2(spec))
    views.append(_plausible_promptview_v2(spec, "plausible_low", "low"))
    views.append(_plausible_promptview_v2(spec, "plausible_high", "high"))
    views.append(_irrelevant_promptview_v2(spec, "irrelevant_low", "low"))
    views.append(_irrelevant_promptview_v2(spec, "irrelevant_high", "high"))
    return views
