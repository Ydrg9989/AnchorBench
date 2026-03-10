"""ICL (in-context learning) suite renderers.

Legacy (v1): anchor is embedded in demonstration answers.
  3 conditions: control, low_anchor, high_anchor.
  Tests ICL distribution-matching / demonstration imitation.

v2: anchor appears only in demo header metadata (numeric priming).
  5 conditions: control, irrelevant_low/high, plausible_low/high.
  Demo answers remain neutral and identical across all conditions.
  Tests whether incidental metadata numbers in few-shot context
  prime the model's numeric estimate (selective accessibility).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from ..domains import DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView

# ── Legacy (v1) ICL renderer ─────────────────────────────────────────

CONTROL_DEMO_ANSWERS = [25, 50, 75]
DEMO_SPREAD = 5


def _make_demo_answers(anchor: int) -> List[int]:
    return [
        max(0, min(100, anchor - DEMO_SPREAD)),
        anchor,
        max(0, min(100, anchor + DEMO_SPREAD)),
    ]


def _format_evidence_legacy(spec: ItemSpec) -> str:
    lines = []
    for e in spec.evidence_structured:
        lines.append(f"  - {e['label']}: {e['value']}")
    return "\n".join(lines)


def _build_demo_block_legacy(demo_answers: List[int], domain_id: str) -> str:
    demos = []
    for i, ans in enumerate(demo_answers):
        demos.append(
            f"Example {i+1}: Given similar evidence for a comparable case, "
            f"the estimated index was {ans}."
        )
    return "\n".join(demos)


def _build_prompt_legacy(
    spec: ItemSpec,
    condition: str,
    demo_answers: List[int],
) -> PromptView:
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    scenario = spec.scenario_text or dcfg.scenario_templates[tidx % len(dcfg.scenario_templates)]
    question = dcfg.question_templates[tidx % len(dcfg.question_templates)]
    evidence_block = _format_evidence_legacy(spec)
    demo_block = _build_demo_block_legacy(demo_answers, spec.domain)

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
        anchor_val = demo_answers[1]
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


def render_icl_legacy(spec: ItemSpec) -> List[PromptView]:
    """Legacy renderer: anchor in demo answers (3 conditions)."""
    low_demos = _make_demo_answers(spec.anchors["low"])
    high_demos = _make_demo_answers(spec.anchors["high"])
    return [
        _build_prompt_legacy(spec, "control", CONTROL_DEMO_ANSWERS),
        _build_prompt_legacy(spec, "low_anchor", low_demos),
        _build_prompt_legacy(spec, "high_anchor", high_demos),
    ]


# backward compat alias
render_icl = render_icl_legacy


# ── ICL v2 renderer (metadata priming) ───────────────────────────────

_MISSING_VALUE_DISPLAY = "[data not available]"

ICL_V2_CONDITIONS: List[Tuple[str, str, str]] = [
    # (condition_name, relevance_type, anchor_direction)
    ("control",         "none",       ""),
    ("irrelevant_low",  "irrelevant", "low"),
    ("irrelevant_high", "irrelevant", "high"),
    ("plausible_low",   "plausible",  "low"),
    ("plausible_high",  "plausible",  "high"),
]

N_DEMOS = 3
DEMO_EVIDENCE_COUNT = 3


def _generate_demo_evidence(
    rng: random.Random,
    domain_id: str,
    demo_idx: int,
) -> Tuple[List[dict], int]:
    """Generate evidence + neutral answer for one demo.

    Returns (evidence_list, answer). The answer is round(mean(values)),
    drawn from a moderate range to avoid correlation with anchors.
    """
    dcfg = DOMAINS[domain_id]
    labels = dcfg.evidence_labels[:DEMO_EVIDENCE_COUNT]

    theta_demo = rng.randint(35, 65)
    values = []
    for lbl in labels:
        v = max(0, min(100, round(rng.gauss(theta_demo, 6))))
        values.append({"label": lbl, "value": v})

    answer = round(sum(e["value"] for e in values) / len(values))
    return values, answer


def _format_evidence_v2(spec: ItemSpec) -> str:
    """Format target evidence, rendering missing values for hard items."""
    lines = []
    for e in spec.evidence_structured:
        if e.get("missing"):
            lines.append(f"  - {e['label']}: {_MISSING_VALUE_DISPLAY}")
        else:
            lines.append(f"  - {e['label']}: {e['value']}")
    return "\n".join(lines)


def _format_demo_evidence(evidence: List[dict]) -> str:
    lines = []
    for e in evidence:
        lines.append(f"  - {e['label']}: {e['value']}")
    return "\n".join(lines)


def _make_demo_header(
    demo_idx: int,
    relevance: str,
    anchor_value: Optional[int],
) -> str:
    """Build the demo header string that varies across conditions."""
    num = demo_idx + 1
    if relevance == "none" or anchor_value is None:
        return f"Example {num}:"
    elif relevance == "irrelevant":
        return f"Example {num} [Case #{anchor_value}, Batch {anchor_value}]:"
    elif relevance == "plausible":
        return f"Example {num} [Prior estimate for similar cases: {anchor_value}, confidence: low]:"
    return f"Example {num}:"


def _build_prompt_v2(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: Optional[int],
    demos: List[Tuple[List[dict], int]],
) -> PromptView:
    """Build a single ICL v2 PromptView."""
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    scenario = (
        spec.scenario_text
        or dcfg.scenario_templates[tidx % len(dcfg.scenario_templates)]
    )
    question = dcfg.question_templates[tidx % len(dcfg.question_templates)]
    target_evidence = _format_evidence_v2(spec)

    demo_blocks = []
    demo_headers = []
    demo_answers_list = []
    demo_evidence_strs = []
    for di, (demo_ev, demo_ans) in enumerate(demos):
        header = _make_demo_header(di, relevance, anchor_value)
        demo_headers.append(header)
        demo_answers_list.append(demo_ans)
        ev_str = _format_demo_evidence(demo_ev)
        demo_evidence_strs.append(ev_str)
        demo_blocks.append(
            f"{header}\n{ev_str}\nAnswer: {demo_ans}"
        )

    demos_text = "\n\n".join(demo_blocks)

    prompt_text = (
        f"Below are examples of similar estimation tasks, followed by a new case.\n\n"
        f"{demos_text}\n\n"
        f"Now estimate for a new case:\n\n"
        f"{scenario}\n\nEvidence:\n{target_evidence}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    components = {
        "demos": demos_text,
        "demo_headers": demo_headers,
        "demo_answers": demo_answers_list,
        "demo_evidence": demo_evidence_strs,
        "scenario": scenario,
        "evidence": target_evidence,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
    }

    anchor_string: Optional[str] = None
    anchor_span: Optional[List[int]] = None

    if condition != "control" and anchor_value is not None:
        anchor_string = str(anchor_value)
        start = prompt_text.find(anchor_string)
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


def render_icl_v2(spec: ItemSpec) -> List[PromptView]:
    """Render 5 conditions for an ICL v2 item (metadata priming).

    Demo evidence and answers are identical across all 5 conditions;
    only the demo header metadata changes.
    """
    icl_demos_tag = spec.tags.get("icl_demos")
    if icl_demos_tag:
        demos = [
            (d["evidence"], d["answer"])
            for d in icl_demos_tag
        ]
    else:
        rng = random.Random(spec.seed * 10000 + hash(spec.item_id) % 10000)
        demos = [
            _generate_demo_evidence(rng, spec.domain, di)
            for di in range(N_DEMOS)
        ]

    views: List[PromptView] = []
    for cond_name, relevance, direction in ICL_V2_CONDITIONS:
        anchor_val = spec.anchors.get(direction) if direction else None
        views.append(
            _build_prompt_v2(spec, cond_name, relevance, anchor_val, demos)
        )
    return views
