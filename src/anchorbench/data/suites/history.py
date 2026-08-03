"""History suite: two-stage self-generated anchoring.

Plausible: Stage 1 shows partial evidence (2-of-5 subset) → model generates
  an initial estimate (the anchor). Stage 2 reveals full evidence.
Irrelevant: Stage 1 shows a warmup case (same domain, different item) →
  model generates an estimate. Stage 2 switches to the target case.
Control: single-stage, full evidence (identical to External v2 control).
Control_twostage: two-stage matched control — Stage 1 asks a non-numeric
  qualitative question with full evidence; Stage 2 asks for the numeric
  estimate.  Matches the two-stage format without eliciting an anchor.

6 conditions: control, control_twostage, plausible_low/high, irrelevant_low/high.
"""

from __future__ import annotations

from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView
from ._shared import format_evidence, resolve_templates
from .external import _build_prompt as _external_build_prompt

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

STAGE1_NEUTRAL_QUESTION = (
    "Summarize the key factors relevant to this assessment."
)
STAGE2_NEUTRAL_PREAMBLE = (
    "Thank you. Now, based on the same evidence, provide a numeric estimate.\n\n"
)


def _plausible_promptview(spec: ItemSpec, condition: str, direction: str) -> PromptView:
    """Plausible: two-stage; Stage 1 = partial evidence, Stage 2 = same case full."""
    scenario, question, _, _ = resolve_templates(spec)
    full_evidence = format_evidence(spec.evidence_structured)

    hist = spec.history or {}
    key = "subset_indices_low" if direction == "low" else "subset_indices_high"
    indices = hist.get(key, [0, 1])
    partial_evidence = format_evidence(spec.evidence_structured, indices=indices)

    stage1 = (
        f"{scenario}\n\n"
        f"Preliminary evidence (partial):\n{partial_evidence}\n\n"
        f"{STAGE1_PLAUSIBLE_QUESTION}\n{ANSWER_FORMAT_INSTRUCTION}"
    )
    stage2 = (
        f"{STAGE2_PLAUSIBLE_PREAMBLE}"
        f"Evidence:\n{full_evidence}\n\n"
        f"{STAGE2_PLAUSIBLE_QUESTION}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    return PromptView(
        item_id=spec.item_id, suite=spec.suite, domain=spec.domain,
        condition=condition,
        prompt_text=f"{stage1}\n\n---\n\n{stage2}",
        prompt_components={
            "scenario": scenario, "evidence": full_evidence,
            "question": question, "answer_format": ANSWER_FORMAT_INSTRUCTION,
            "stage1_user_message": stage1, "stage2_user_message": stage2,
        },
        anchor_relevance="plausible",
    )


def _irrelevant_promptview(spec: ItemSpec, condition: str, direction: str) -> PromptView:
    """Irrelevant: two-stage; Stage 1 = warmup case, Stage 2 = new target case."""
    scenario, question, _, dcfg = resolve_templates(spec)
    target_evidence = format_evidence(spec.evidence_structured)

    hist = spec.history or {}
    warmup = hist.get("warmup_low" if direction == "low" else "warmup_high", {})
    w_tidx = warmup.get("scenario_template_idx", 0)
    warmup_scenario = dcfg.scenario_templates[w_tidx % len(dcfg.scenario_templates)]
    warmup_question = dcfg.question_templates[w_tidx % len(dcfg.question_templates)]
    warmup_evidence = format_evidence(warmup.get("evidence_structured", []), show_missing=False)

    stage1 = (
        f"{warmup_scenario}\n\n"
        f"Evidence:\n{warmup_evidence}\n\n"
        f"{warmup_question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )
    stage2 = (
        f"{STAGE2_IRRELEVANT_PREAMBLE}"
        f"{scenario}\n\n"
        f"Evidence:\n{target_evidence}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    return PromptView(
        item_id=spec.item_id, suite=spec.suite, domain=spec.domain,
        condition=condition,
        prompt_text=f"{stage1}\n\n---\n\n{stage2}",
        prompt_components={
            "scenario": scenario, "evidence": target_evidence,
            "question": question, "answer_format": ANSWER_FORMAT_INSTRUCTION,
            "stage1_user_message": stage1, "stage2_user_message": stage2,
        },
        anchor_relevance="irrelevant",
    )


def _neutral_twostage_promptview(spec: ItemSpec) -> PromptView:
    """Matched two-stage control: Stage 1 = qualitative (no numeric answer),
    Stage 2 = numeric estimate with same evidence.  Preserves the two-stage
    conversational structure without eliciting a self-generated anchor."""
    scenario, question, _, _ = resolve_templates(spec)
    full_evidence = format_evidence(spec.evidence_structured)

    stage1 = (
        f"{scenario}\n\n"
        f"Evidence:\n{full_evidence}\n\n"
        f"{STAGE1_NEUTRAL_QUESTION}"
    )
    stage2 = (
        f"{STAGE2_NEUTRAL_PREAMBLE}"
        f"Evidence:\n{full_evidence}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    return PromptView(
        item_id=spec.item_id, suite=spec.suite, domain=spec.domain,
        condition="control_twostage",
        prompt_text=f"{stage1}\n\n---\n\n{stage2}",
        prompt_components={
            "scenario": scenario, "evidence": full_evidence,
            "question": question, "answer_format": ANSWER_FORMAT_INSTRUCTION,
            "stage1_user_message": stage1, "stage2_user_message": stage2,
        },
        anchor_relevance="none",
    )


def _intensity_promptview(
    spec: ItemSpec, condition: str, direction: str, intensity: str,
) -> PromptView:
    """P1 cross-pathway intensity for History.

    Stage 1 is identical to the standard plausible condition (model
    generates a self-anchor from partial evidence). Stage 2 reveals the
    full evidence AND inserts an external plausible preamble at the
    requested credibility level. This tests whether an external
    plausibility cue *compounds* with the self-anchor pathway.

    intensity is one of ``plausible_mild`` / ``plausible_strong``.
    """
    assert intensity in ("plausible_mild", "plausible_strong")
    scenario, question, _, dcfg = resolve_templates(spec)
    full_evidence = format_evidence(spec.evidence_structured)

    hist = spec.history or {}
    key = "subset_indices_low" if direction == "low" else "subset_indices_high"
    indices = hist.get(key, [0, 1])
    partial_evidence = format_evidence(spec.evidence_structured, indices=indices)

    anchor_val = spec.anchors[direction]
    pool = dcfg.anchor_preambles.get(intensity, [])
    if not pool:
        raise KeyError(
            f"Domain {spec.domain!r} missing {intensity!r} preamble pool"
        )
    pidx = getattr(spec, "anchor_phrasing_idx", 0)
    anchor_sentence = pool[pidx % len(pool)].format(anchor=anchor_val)

    stage1 = (
        f"{scenario}\n\n"
        f"Preliminary evidence (partial):\n{partial_evidence}\n\n"
        f"{STAGE1_PLAUSIBLE_QUESTION}\n{ANSWER_FORMAT_INSTRUCTION}"
    )
    stage2 = (
        f"{STAGE2_PLAUSIBLE_PREAMBLE}"
        f"Evidence:\n{full_evidence}\n\n"
        f"{anchor_sentence} "
        f"{STAGE2_PLAUSIBLE_QUESTION}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    return PromptView(
        item_id=spec.item_id, suite=spec.suite, domain=spec.domain,
        condition=condition,
        prompt_text=f"{stage1}\n\n---\n\n{stage2}",
        prompt_components={
            "scenario": scenario, "evidence": full_evidence,
            "question": question, "answer_format": ANSWER_FORMAT_INSTRUCTION,
            "stage1_user_message": stage1, "stage2_user_message": stage2,
            "intensity_preamble": anchor_sentence,
        },
        anchor_string=str(anchor_val),
        anchor_relevance=intensity,
        anchor_value=anchor_val,
    )


def build_intensity_promptviews(spec: ItemSpec) -> list[PromptView]:
    """P1 cross-pathway intensity probe for History.

    Renders 4 conditions: ``plausible_mild_low/high`` and
    ``plausible_strong_low/high``. The standard ``plausible_low/high``
    is reused from the existing History run for the middle point.
    """
    if spec.suite != "history" or not spec.history:
        return []
    return [
        _intensity_promptview(spec, "plausible_mild_low", "low", "plausible_mild"),
        _intensity_promptview(spec, "plausible_mild_high", "high", "plausible_mild"),
        _intensity_promptview(spec, "plausible_strong_low", "low", "plausible_strong"),
        _intensity_promptview(spec, "plausible_strong_high", "high", "plausible_strong"),
    ]


def render_history(spec: ItemSpec) -> list[PromptView]:
    """Render 6 conditions for History.

    control          — single-stage (same as External control)
    control_twostage — two-stage matched neutral control (qualitative Stage 1)
    plausible/irrelevant — two-stage with different Stage 1 content
    """
    if spec.suite != "history" or not spec.history:
        return []
    return [
        _external_build_prompt(spec, "control", "none", None),
        _neutral_twostage_promptview(spec),
        _plausible_promptview(spec, "plausible_low", "low"),
        _plausible_promptview(spec, "plausible_high", "high"),
        _irrelevant_promptview(spec, "irrelevant_low", "low"),
        _irrelevant_promptview(spec, "irrelevant_high", "high"),
    ]
