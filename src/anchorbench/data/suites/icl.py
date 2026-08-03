"""ICL renderer: anchor appears only in demo header metadata.

7 conditions: control, irrelevant_low/high, plausible_low/high,
              neutral_low/high.

Demo evidence and answers are neutral and identical across all seven
conditions.  Only the demo header line varies.

Pairing invariant: across conditions, every part of the prompt is
identical except the demo header line.
"""

from __future__ import annotations

import hashlib
import random

from ..domains import ALL_DOMAINS as DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView
from ._shared import ICL_CONDITIONS, format_evidence, resolve_templates

N_DEMOS = 3

# ── Header templates per relevance type ──────────────────────────────

_IRRELEVANT_HEADERS = [
    "Example {num} [Case #{anchor}, Batch {anchor}]:",
    "Example {num} [Queue #{anchor}, Ref {anchor}]:",
    "Example {num} [Request #{anchor}, Seq {anchor}]:",
    "Example {num} [Log entry #{anchor}, ID {anchor}]:",
]

_PLAUSIBLE_HEADERS = [
    "Example {num} [Prior estimate for similar cases: {anchor}, confidence: low]:",
    "Example {num} [Preliminary analysis suggested approximately {anchor}]:",
    "Example {num} [Historical reference for comparable items: {anchor}]:",
]

_NEUTRAL_HEADERS = [
    "Example {num} [Ref: {anchor}]:",
]


def _make_demo_header(
    demo_idx: int,
    relevance: str,
    anchor_value: int | None,
    phrasing_idx: int = 0,
) -> str:
    """Build the demo header that varies across conditions."""
    num = demo_idx + 1
    if relevance == "irrelevant" and anchor_value is not None:
        pool = _IRRELEVANT_HEADERS
        template = pool[phrasing_idx % len(pool)]
        return template.format(num=num, anchor=anchor_value)
    if relevance == "plausible" and anchor_value is not None:
        pool = _PLAUSIBLE_HEADERS
        template = pool[phrasing_idx % len(pool)]
        return template.format(num=num, anchor=anchor_value)
    if relevance == "neutral" and anchor_value is not None:
        pool = _NEUTRAL_HEADERS
        template = pool[phrasing_idx % len(pool)]
        return template.format(num=num, anchor=anchor_value)
    return f"Example {num}:"


def _build_prompt(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: int | None,
    demos: list[tuple[list[dict], int]],
) -> PromptView:
    """Build a single ICL PromptView for one condition."""
    scenario, question, _, _ = resolve_templates(spec)
    target_evidence = format_evidence(spec.evidence_structured)

    demo_blocks = []
    demo_headers = []
    demo_answers_list = []
    demo_evidence_strs = []
    for di, (demo_ev, demo_ans) in enumerate(demos):
        header = _make_demo_header(di, relevance, anchor_value, spec.anchor_phrasing_idx)
        ev_str = format_evidence(demo_ev, show_missing=False)
        demo_blocks.append(f"{header}\n{ev_str}\nAnswer: {demo_ans}")
        demo_headers.append(header)
        demo_answers_list.append(demo_ans)
        demo_evidence_strs.append(ev_str)

    demos_text = "\n\n".join(demo_blocks)

    prompt_text = (
        f"Below are examples of similar estimation tasks, followed by a new case.\n\n"
        f"{demos_text}\n\n"
        f"Now estimate for a new case:\n\n"
        f"{scenario}\n\nEvidence:\n{target_evidence}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    anchor_string: str | None = None
    anchor_span: list[int] | None = None
    if condition != "control" and anchor_value is not None:
        anchor_string = str(anchor_value)
        start = prompt_text.find(anchor_string)
        if start >= 0:
            anchor_span = [start, start + len(anchor_string)]

    return PromptView(
        item_id=spec.item_id, suite=spec.suite, domain=spec.domain,
        condition=condition, prompt_text=prompt_text,
        prompt_components={
            "demos": demos_text,
            "demo_headers": demo_headers,
            "demo_answers": demo_answers_list,
            "demo_evidence": demo_evidence_strs,
            "scenario": scenario, "evidence": target_evidence,
            "question": question, "answer_format": ANSWER_FORMAT_INSTRUCTION,
        },
        anchor_string=anchor_string, anchor_span=anchor_span,
        anchor_relevance=relevance, anchor_value=anchor_value,
    )


def render_icl(spec: ItemSpec) -> list[PromptView]:
    """Render 7 conditions for an ICL item."""
    icl_demos_tag = spec.tags.get("icl_demos")
    if icl_demos_tag:
        demos = [(d["evidence"], d["answer"]) for d in icl_demos_tag]
    else:
        stable_hash = int(hashlib.sha256(spec.item_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(spec.seed * 10000 + stable_hash % 10000)
        labels = DOMAINS[spec.domain].evidence_labels[:3]
        demos = []
        for _ in range(N_DEMOS):
            theta = rng.randint(35, 65)
            ev = [{"label": lbl, "value": max(0, min(100, round(rng.gauss(theta, 6))))} for lbl in labels]
            demos.append((ev, round(sum(e["value"] for e in ev) / len(ev))))

    return [
        _build_prompt(spec, cond, rel, spec.anchors.get(d) if d else None, demos)
        for cond, rel, d in ICL_CONDITIONS
    ]
