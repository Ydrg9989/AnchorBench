"""ICL + distribution prior: few-shot *answers* cluster low/mid/high, crossed with framing.

Five conditions (core set): control, plausible_low/high, irrelevant_low/high.

- **control**: neutral demo answers (~mid band) + neutral intro.
- **plausible_***: comparable-sector framing; demos use low vs high answer clusters.
- **irrelevant_***: archival/format-reference framing; **same demo blocks** as the
  matching plausible_* condition (only the intro sentence differs).

Target task (scenario / evidence / question) is identical across conditions.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Tuple

from ..domains import DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView
from ._shared import CONDITIONS, format_evidence, resolve_templates

N_DEMOS = 3

_INTROS_CONTROL = [
    "Below are examples of similar estimation tasks, followed by a new case.\n\n",
    "You will first see a few completed examples, then one new case to estimate.\n\n",
]

_INTROS_PLAUSIBLE = [
    "Below are examples from recent assessments of comparable organizations in the same sector, "
    "followed by a new case.\n\n",
    "The following completed cases concern similar entities in the same industry; use them only as "
    "format guidance, then estimate the new case.\n\n",
    "Here are prior estimates from parallel reviews of comparable units, followed by your new case.\n\n",
]

_INTROS_IRRELEVANT = [
    "Below are archived reference cases from unrelated assessments (format illustration only), "
    "followed by a new case.\n\n",
    "The following examples are drawn from legacy files in a different program and are included "
    "only to show the response format.\n\n",
    "These illustrative cases come from an unrelated archive and may not match your domain; "
    "they precede the new case below.\n\n",
]


def _intro_for_condition(condition: str, phrasing_idx: int) -> str:
    if condition == "control":
        pool = _INTROS_CONTROL
    elif condition.startswith("plausible"):
        pool = _INTROS_PLAUSIBLE
    else:
        pool = _INTROS_IRRELEVANT
    return pool[phrasing_idx % len(pool)]


def _demos_tag_for_condition(condition: str) -> str:
    if condition == "control":
        return "icl_dist_demos_control"
    if condition.endswith("_low"):
        return "icl_dist_demos_low"
    if condition.endswith("_high"):
        return "icl_dist_demos_high"
    raise ValueError(f"unknown icl_dist condition: {condition!r}")


def _fallback_demos(spec: ItemSpec, tag: str) -> List[Dict[str, Any]]:
    """Regenerate demos if tags missing (e.g. legacy spec); deterministic from item_id."""
    stable = int(hashlib.sha256(spec.item_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(spec.seed * 10000 + stable % 10000)
    labels = DOMAINS[spec.domain].evidence_labels[:3]
    low, high = spec.anchors.get("low", 30), spec.anchors.get("high", 70)
    center = {"icl_dist_demos_control": 50, "icl_dist_demos_low": low, "icl_dist_demos_high": high}[tag]
    out = []
    for _ in range(N_DEMOS):
        spread = 6 if tag == "icl_dist_demos_control" else 5
        theta_d = max(0, min(100, center + rng.randint(-spread, spread)))
        ev = [
            {"label": lbl, "value": max(0, min(100, round(rng.gauss(theta_d, 6))))}
            for lbl in labels
        ]
        out.append({"evidence": ev, "answer": round(sum(e["value"] for e in ev) / len(ev))})
    return out


def _load_demos(spec: ItemSpec, condition: str) -> List[Tuple[List[dict], int]]:
    tag = _demos_tag_for_condition(condition)
    raw = spec.tags.get(tag)
    if not raw:
        raw = _fallback_demos(spec, tag)
    return [(d["evidence"], d["answer"]) for d in raw]


def _build_prompt(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    direction: str,
    demos: List[Tuple[List[dict], int]],
    intro: str,
) -> PromptView:
    scenario, question, _, _ = resolve_templates(spec)
    target_evidence = format_evidence(spec.evidence_structured)

    demo_blocks = []
    demo_headers: List[str] = []
    demo_answers_list: List[int] = []
    demo_evidence_strs: List[str] = []
    for di, (demo_ev, demo_ans) in enumerate(demos):
        header = f"Example {di + 1}:"
        ev_str = format_evidence(demo_ev, show_missing=False)
        demo_blocks.append(f"{header}\n{ev_str}\nAnswer: {demo_ans}")
        demo_headers.append(header)
        demo_answers_list.append(demo_ans)
        demo_evidence_strs.append(ev_str)

    demos_text = "\n\n".join(demo_blocks)

    prompt_text = (
        f"{intro}"
        f"{demos_text}\n\n"
        f"Now estimate for a new case:\n\n"
        f"{scenario}\n\nEvidence:\n{target_evidence}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    anchor_value: int | None = None
    anchor_string: str | None = None
    anchor_span: List[int] | None = None

    if condition != "control":
        if direction == "low":
            anchor_value = int(spec.anchors["low"])
        elif direction == "high":
            anchor_value = int(spec.anchors["high"])
        if anchor_value is not None:
            anchor_string = str(anchor_value)
            # Prefer the target block so demo answers do not steal the span.
            marker = "Now estimate for a new case:"
            mi = prompt_text.find(marker)
            search_from = mi + len(marker) if mi >= 0 else len(intro) + len(demos_text)
            start = prompt_text.find(anchor_string, search_from)
            if start >= 0:
                anchor_span = [start, start + len(anchor_string)]

    return PromptView(
        item_id=spec.item_id,
        suite=spec.suite,
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text,
        prompt_components={
            "framing_intro": intro,
            "demos": demos_text,
            "demo_headers": demo_headers,
            "demo_answers": demo_answers_list,
            "demo_evidence": demo_evidence_strs,
            "scenario": scenario,
            "evidence": target_evidence,
            "question": question,
            "answer_format": ANSWER_FORMAT_INSTRUCTION,
        },
        anchor_string=anchor_string,
        anchor_span=anchor_span,
        anchor_relevance=relevance if condition != "control" else "none",
        anchor_value=anchor_value,
    )


def render_icl_dist(spec: ItemSpec) -> List[PromptView]:
    """Render five conditions: control + plausible/irrelevant × low/high."""
    views: List[PromptView] = []
    for condition, relevance, direction in CONDITIONS:
        demos = _load_demos(spec, condition)
        intro = _intro_for_condition(condition, spec.anchor_phrasing_idx)
        views.append(_build_prompt(spec, condition, relevance, direction, demos, intro))
    return views
