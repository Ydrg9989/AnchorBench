"""Conversation-history suite: anchor appears in a prior chat turn.

Control:   no prior conversation history.
Low/High:  earlier exchange where an "assistant" mentioned the anchor value.
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
    """Render control / low_anchor / high_anchor for a history-suite item."""
    return [
        _build_prompt(spec, "control", None),
        _build_prompt(spec, "low_anchor", spec.anchors["low"]),
        _build_prompt(spec, "high_anchor", spec.anchors["high"]),
    ]
