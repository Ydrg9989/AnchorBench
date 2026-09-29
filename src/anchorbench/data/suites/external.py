"""External-anchor suite: anchor is a numeric cue injected as a sentence.

9 conditions: control / irrelevant_low / irrelevant_high
                      / plausible_low  / plausible_high
                      / placebo_low    / placebo_high
                      / authority_low  / authority_high

Pairing invariant: all conditions share the same item stem; only the
anchor-bearing sentence changes.
"""

from __future__ import annotations

from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView
from ._shared import (
    EXTENDED_CONDITIONS,
    anchored_prompt,
    format_evidence,
    resolve_templates,
)


def _build_prompt(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: int | None,
) -> PromptView:
    """Build a single PromptView for External."""
    scenario, question, _, dcfg = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)

    components = {
        "scenario": scenario,
        "evidence": evidence_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
    }

    base_text = f"{scenario}\n\nEvidence:\n{evidence_block}\n\n"
    prompt_text, anchor_sentence, anchor_string, anchor_span = anchored_prompt(
        base_text, question, dcfg, relevance, spec.anchor_phrasing_idx, anchor_value,
    )
    if anchor_sentence is not None:
        components["anchor_sentence"] = anchor_sentence

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


def render_external(spec: ItemSpec) -> list[PromptView]:
    """Render 9 conditions for an External item."""
    return [
        _build_prompt(
            spec, cond, rel,
            spec.anchors.get(d) if d else None,
        )
        for cond, rel, d in EXTENDED_CONDITIONS
    ]
