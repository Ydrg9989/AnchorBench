"""Tool-Read suite: anchoring through structured JSON in a plain prompt.

Same tool output content as Tool-Agentic but rendered as plain structured
JSON blocks within a standard user message — no role:"tool", no function-
calling schemas, no assistant tool-call messages.  Isolates whether
anchors in structured data influence estimates, independent of
tool-calling format competence.

Conditions (5):
  control         — no external reference data available
  irrelevant_low  — request_id field contains the anchor (low)
  irrelevant_high — request_id field contains the anchor (high)
  plausible_low   — reference_value field contains the anchor (low)
  plausible_high  — reference_value field contains the anchor (high)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView
from ._shared import CONDITIONS, format_evidence, resolve_templates
from .tool_agentic import (
    execute_check_external_reference,
    execute_get_evidence_summary,
    _get_visible_ratings,
)


def _build_prompt(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: Optional[int],
) -> PromptView:
    """Build a single PromptView with tool outputs as plain JSON blocks."""
    scenario, question, _, dcfg = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)
    metric = dcfg.metric_name or dcfg.display_name.lower()

    ratings = _get_visible_ratings(spec)
    evidence_result = execute_get_evidence_summary(ratings)
    reference_result = execute_check_external_reference(
        domain=spec.domain, metric=metric,
        condition=condition, anchor_value=anchor_value,
    )

    prompt_text = (
        f"{scenario}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"The following analysis results are available:\n\n"
        f"Evidence summary:\n{json.dumps(evidence_result, indent=2)}\n\n"
        f"External reference lookup:\n{json.dumps(reference_result, indent=2)}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    anchor_string: Optional[str] = None
    anchor_span: Optional[List[int]] = None
    anchor_val_out: Optional[int] = None

    if condition != "control" and anchor_value is not None:
        anchor_val_out = anchor_value
        anchor_string = str(anchor_value)
        if "irrelevant" in condition:
            search_str = f'"request_id": {anchor_value}'
        else:
            search_str = f'"reference_value": {anchor_value}'
        start = prompt_text.find(search_str)
        if start >= 0:
            field_prefix_len = search_str.index(str(anchor_value))
            anchor_start = start + field_prefix_len
            anchor_span = [anchor_start, anchor_start + len(anchor_string)]

    components: Dict[str, Any] = {
        "scenario": scenario,
        "evidence": evidence_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
        "evidence_summary": json.dumps(evidence_result),
        "reference_result": json.dumps(reference_result),
    }

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
        anchor_value=anchor_val_out,
    )


def render_tool_read(spec: ItemSpec) -> List[PromptView]:
    """Render 5 conditions for a Tool-Read item.

    All share the same item stem and evidence; only the external reference
    lookup JSON differs.
    """
    return [
        _build_prompt(spec, cond, rel, spec.anchors.get(d) if d else None)
        for cond, rel, d in CONDITIONS
    ]
