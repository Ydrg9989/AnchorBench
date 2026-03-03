"""External-anchor suite: anchor is a normatively irrelevant numeric cue
injected as an "industry report" or "recent study" sentence.

Pairing invariant: control/low/high prompts differ ONLY in the anchor sentence.
"""

from __future__ import annotations

from typing import List, Optional

from ..domains import DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView


def _format_evidence(spec: ItemSpec) -> str:
    lines = []
    for e in spec.evidence_structured:
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
