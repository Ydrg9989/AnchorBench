"""ICL (in-context learning) suite: anchor is embedded in demonstration answers.

Control: 3 demos with neutral answers [25, 50, 75]
Low:     3 demos with low-biased answers close to anchor_low
High:    3 demos with high-biased answers close to anchor_high

The scenario/question for the target item is identical across conditions.
"""

from __future__ import annotations

from typing import List, Optional

from ..domains import DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView

CONTROL_DEMO_ANSWERS = [25, 50, 75]
DEMO_SPREAD = 5  # answers cluster within ±DEMO_SPREAD of anchor


def _make_demo_answers(anchor: int) -> List[int]:
    return [
        max(0, min(100, anchor - DEMO_SPREAD)),
        anchor,
        max(0, min(100, anchor + DEMO_SPREAD)),
    ]


def _format_evidence(spec: ItemSpec) -> str:
    lines = []
    for e in spec.evidence_structured:
        lines.append(f"  - {e['label']}: {e['value']}")
    return "\n".join(lines)


def _build_demo_block(demo_answers: List[int], domain_id: str) -> str:
    dcfg = DOMAINS[domain_id]
    demos = []
    for i, ans in enumerate(demo_answers):
        demos.append(
            f"Example {i+1}: Given similar evidence for a comparable case, "
            f"the estimated index was {ans}."
        )
    return "\n".join(demos)


def _build_prompt(
    spec: ItemSpec,
    condition: str,
    demo_answers: List[int],
) -> PromptView:
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    scenario = spec.scenario_text or dcfg.scenario_templates[tidx % len(dcfg.scenario_templates)]
    question = dcfg.question_templates[tidx % len(dcfg.question_templates)]
    evidence_block = _format_evidence(spec)
    demo_block = _build_demo_block(demo_answers, spec.domain)

    components = {
        "demos": demo_block,
        "scenario": scenario,
        "evidence": evidence_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
    }

    prompt_text = (
        f"{demo_block}\n\n"
        f"Now estimate for a new case:\n\n"
        f"{scenario}\n\nEvidence:\n{evidence_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    anchor_string = None
    anchor_span = None
    if condition != "control":
        anchor_val = demo_answers[1]  # the center demo answer is the anchor
        anchor_string = str(anchor_val)
        start = prompt_text.find(f"was {anchor_val}.")
        if start >= 0:
            start += len("was ")
            anchor_span = [start, start + len(str(anchor_val))]

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


def render_icl(spec: ItemSpec) -> List[PromptView]:
    """Render control / low_anchor / high_anchor for an ICL-suite item."""
    low_demos = _make_demo_answers(spec.anchors["low"])
    high_demos = _make_demo_answers(spec.anchors["high"])
    return [
        _build_prompt(spec, "control", CONTROL_DEMO_ANSWERS),
        _build_prompt(spec, "low_anchor", low_demos),
        _build_prompt(spec, "high_anchor", high_demos),
    ]
