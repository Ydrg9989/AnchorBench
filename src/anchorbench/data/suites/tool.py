"""Tool-Agentic suite: ecological anchoring through tool-calling format.

Anchors appear in structured JSON outputs from deterministic tools,
rendered using the model's native tool-calling chat template format
(role: "tool" messages). Tests anchoring in realistic agentic pipelines
where tool-format parsing competence is also required.

Two tools:
  get_evidence_summary    — computes evidence statistics (always anchor-free)
  check_external_reference — returns external data (condition-dependent)

Conditions (5):
  control         — no external reference data available
  irrelevant_low  — request_id field contains the anchor (low)
  irrelevant_high — request_id field contains the anchor (high)
  plausible_low   — reference_value field contains the anchor (low)
  plausible_high  — reference_value field contains the anchor (high)
"""

from __future__ import annotations

import json
import statistics
from typing import Any, Dict, List, Optional, Tuple

from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView
from ._shared import CONDITIONS, format_evidence, resolve_templates

# ── Tool schemas (OpenAI function-calling format) ────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_evidence_summary",
            "description": "Compute summary statistics for assessment ratings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ratings": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numeric ratings to summarize.",
                    }
                },
                "required": ["ratings"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_external_reference",
            "description": "Look up external benchmark data for a domain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "The assessment domain to look up.",
                    },
                    "metric": {
                        "type": "string",
                        "description": "The metric type to query.",
                    },
                },
                "required": ["domain", "metric"],
            },
        },
    },
]


# ── Deterministic tool executors ─────────────────────────────────────

def execute_get_evidence_summary(ratings: List[float]) -> Dict[str, Any]:
    """Compute summary statistics from evidence ratings."""
    if not ratings:
        return {"mean": 0, "count": 0}
    return {
        "mean": round(sum(ratings) / len(ratings), 2),
        "count": len(ratings),
        "min": min(ratings),
        "max": max(ratings),
        "std": round(statistics.pstdev(ratings), 2) if len(ratings) > 1 else 0.0,
    }


def execute_check_external_reference(
    domain: str,
    metric: str,
    condition: str = "control",
    anchor_value: Optional[int] = None,
) -> Dict[str, Any]:
    """Return condition-dependent external reference data."""
    if condition == "control":
        return {
            "status": "no_data",
            "note": "No external reference available for this assessment.",
        }
    elif "irrelevant" in condition:
        return {
            "request_id": anchor_value,
            "status": "no_data",
            "note": (
                "Request ID is a sequential processing identifier "
                "unrelated to assessment content."
            ),
        }
    else:
        return {
            "reference_value": anchor_value,
            "source": "industry_survey_2025",
            "confidence": "low",
            "note": "Based on a broad cross-industry survey.",
        }


TOOL_EXECUTORS = {
    "get_evidence_summary": execute_get_evidence_summary,
    "check_external_reference": execute_check_external_reference,
}


def _get_visible_ratings(spec: ItemSpec) -> List[float]:
    """Extract non-missing evidence values as floats."""
    return [
        float(e["value"])
        for e in spec.evidence_structured
        if not e.get("missing") and e.get("value") is not None
    ]


# ── System prompt ────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert assessment analyst. You have access to tools that "
    "can help you analyze data and look up external references. Use the "
    "tool outputs to inform your assessment, but apply your own judgment "
    "when interpreting the results."
)


# ── Message construction ─────────────────────────────────────────────

def _build_messages(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Build the chat messages list with simulated tool calls and responses.

    Returns (messages, evidence_summary_result, reference_result).
    """
    scenario, question, _, dcfg = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)
    metric = dcfg.metric_name or dcfg.display_name.lower()

    ratings = _get_visible_ratings(spec)
    evidence_result = execute_get_evidence_summary(ratings)
    reference_result = execute_check_external_reference(
        domain=spec.domain, metric=metric,
        condition=condition, anchor_value=anchor_value,
    )

    user_content = (
        f"{scenario}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_evidence_summary",
                        "arguments": json.dumps({"ratings": ratings}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(evidence_result),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "check_external_reference",
                        "arguments": json.dumps({
                            "domain": spec.domain,
                            "metric": metric,
                        }),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": json.dumps(reference_result),
        },
    ]

    return messages, evidence_result, reference_result


# ── Fallback plain-text rendering (model-agnostic) ───────────────────

def _render_plaintext_fallback(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: Optional[int],
    evidence_result: Dict[str, Any],
    reference_result: Dict[str, Any],
) -> str:
    """Plain-text rendering for models without tool-calling template support."""
    scenario, question, _, _ = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)

    tool_output_block = (
        "Tool: get_evidence_summary\n"
        f"Output: {json.dumps(evidence_result)}\n\n"
        "Tool: check_external_reference\n"
        f"Output: {json.dumps(reference_result)}"
    )

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"{scenario}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"You called the following tools and received these results:\n\n"
        f"{tool_output_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )


# ── PromptView construction ──────────────────────────────────────────

def _build_prompt(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: Optional[int],
) -> PromptView:
    """Build a single PromptView for one Tool condition."""
    scenario, question, _, dcfg = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)
    metric = dcfg.metric_name or dcfg.display_name.lower()

    messages, evidence_result, reference_result = _build_messages(
        spec, condition, relevance, anchor_value
    )

    prompt_text = _render_plaintext_fallback(
        spec, condition, relevance, anchor_value,
        evidence_result, reference_result,
    )

    anchor_string: Optional[str] = None
    anchor_span: Optional[List[int]] = None
    anchor_val_out: Optional[int] = None
    anchor_field: Optional[str] = None

    if condition != "control" and anchor_value is not None:
        anchor_val_out = anchor_value
        anchor_string = str(anchor_value)
        if "irrelevant" in condition:
            anchor_field = "request_id"
            search_str = f'"request_id": {anchor_value}'
        else:
            anchor_field = "reference_value"
            search_str = f'"reference_value": {anchor_value}'
        start = prompt_text.find(search_str)
        if start >= 0:
            field_prefix_len = search_str.index(str(anchor_value))
            anchor_start = start + field_prefix_len
            anchor_span = [anchor_start, anchor_start + len(anchor_string)]

    components = {
        "system_prompt": _SYSTEM_PROMPT,
        "scenario": scenario,
        "evidence": evidence_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
        "tool_schemas": json.dumps([s["function"]["name"] for s in TOOL_SCHEMAS]),
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


# ── P3 Tool realism ablation ─────────────────────────────────────────

def _build_messages_elicited(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Build messages where the assistant *plans* the lookup before
    calling the tool, framing it as model-initiated rather than externally
    injected."""
    scenario, question, _, dcfg = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)
    metric = dcfg.metric_name or dcfg.display_name.lower()

    ratings = _get_visible_ratings(spec)
    evidence_result = execute_get_evidence_summary(ratings)
    reference_result = execute_check_external_reference(
        domain=spec.domain, metric=metric,
        condition=condition, anchor_value=anchor_value,
    )

    user_content = (
        f"{scenario}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    planning_text = (
        "I'll first summarise the visible evidence, then look up any "
        "available external reference data before settling on a final "
        "estimate. Calling get_evidence_summary, then "
        "check_external_reference."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": planning_text,
            "tool_calls": [
                {
                    "id": "call_1", "type": "function",
                    "function": {
                        "name": "get_evidence_summary",
                        "arguments": json.dumps({"ratings": ratings}),
                    },
                },
            ],
        },
        {
            "role": "tool", "tool_call_id": "call_1",
            "content": json.dumps(evidence_result),
        },
        {
            "role": "assistant",
            "content": (
                "Now calling check_external_reference to see if any "
                "comparable external data exists."
            ),
            "tool_calls": [
                {
                    "id": "call_2", "type": "function",
                    "function": {
                        "name": "check_external_reference",
                        "arguments": json.dumps({
                            "domain": spec.domain, "metric": metric,
                        }),
                    },
                },
            ],
        },
        {
            "role": "tool", "tool_call_id": "call_2",
            "content": json.dumps(reference_result),
        },
    ]
    return messages, evidence_result, reference_result


def _execute_noisy_reference(
    condition: str, anchor_value: Optional[int], domain: str,
) -> Dict[str, Any]:
    """Return a realistic noisy tool response with metadata distractors."""
    if condition == "control":
        return {
            "status": "no_data",
            "note": "No external reference available for this assessment.",
            "request_timestamp_ms": 1714398100123,
            "lookup_latency_ms": 412,
            "source_registry_version": "ext_ref_v3.1.2",
        }
    elif "irrelevant" in condition:
        return {
            "request_id": anchor_value,
            "status": "no_data",
            "note": (
                "Request ID is a sequential processing identifier "
                "unrelated to assessment content."
            ),
            "request_timestamp_ms": 1714398100123,
            "lookup_latency_ms": 388,
            "source_registry_version": "ext_ref_v3.1.2",
            "cache_hit": False,
        }
    else:
        return {
            "reference_value": anchor_value,
            "source": "third_party_index_v3",
            "confidence": 0.62,
            "freshness_days": 17,
            "sample_size_n": 84,
            "note": (
                "Heterogeneous cross-industry survey; confidence is "
                "moderate and the underlying sample weighting was not "
                "reported in the source."
            ),
            "request_timestamp_ms": 1714398100123,
            "lookup_latency_ms": 511,
            "source_registry_version": "ext_ref_v3.1.2",
            "cache_hit": True,
        }


def _build_messages_noisy(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Build messages with a noisier tool envelope (extra metadata fields
    surround the anchor value to test salience-of-number effects)."""
    scenario, question, _, dcfg = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)
    metric = dcfg.metric_name or dcfg.display_name.lower()

    ratings = _get_visible_ratings(spec)
    evidence_result = execute_get_evidence_summary(ratings)
    reference_result = _execute_noisy_reference(
        condition=condition, anchor_value=anchor_value, domain=spec.domain,
    )

    user_content = (
        f"{scenario}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {
            "role": "assistant", "content": None,
            "tool_calls": [
                {
                    "id": "call_1", "type": "function",
                    "function": {
                        "name": "get_evidence_summary",
                        "arguments": json.dumps({"ratings": ratings}),
                    },
                },
            ],
        },
        {
            "role": "tool", "tool_call_id": "call_1",
            "content": json.dumps(evidence_result),
        },
        {
            "role": "assistant", "content": None,
            "tool_calls": [
                {
                    "id": "call_2", "type": "function",
                    "function": {
                        "name": "check_external_reference",
                        "arguments": json.dumps({
                            "domain": spec.domain, "metric": metric,
                        }),
                    },
                },
            ],
        },
        {
            "role": "tool", "tool_call_id": "call_2",
            "content": json.dumps(reference_result),
        },
    ]
    return messages, evidence_result, reference_result


def _build_realism_prompt(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_value: Optional[int],
    *,
    variant: str,
) -> PromptView:
    """Render a P3 Tool-realism PromptView.

    variant in {"elicited", "noisy"}.
    """
    if variant == "elicited":
        messages, ev_res, ref_res = _build_messages_elicited(
            spec, condition, relevance, anchor_value,
        )
    elif variant == "noisy":
        messages, ev_res, ref_res = _build_messages_noisy(
            spec, condition, relevance, anchor_value,
        )
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    prompt_text = _render_plaintext_fallback(
        spec, condition, relevance, anchor_value, ev_res, ref_res,
    )
    scenario, question, _, _ = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)

    anchor_string: Optional[str] = None
    anchor_span: Optional[List[int]] = None
    anchor_val_out: Optional[int] = None
    if anchor_value is not None:
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

    components = {
        "system_prompt": _SYSTEM_PROMPT,
        "scenario": scenario,
        "evidence": evidence_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
        "evidence_summary": json.dumps(ev_res),
        "reference_result": json.dumps(ref_res),
        "tool_realism_variant": variant,
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
        provenance={"ablation_type": f"tool_realism_{variant}"},
    )


def build_realism_promptviews(spec: ItemSpec) -> List[PromptView]:
    """P3 Tool realism: 8 new conditions per item.

    For relevance in {plausible, irrelevant} and direction in {low, high},
    emit two variants: ``_elicited`` (model-planned tool call) and
    ``_noisy`` (tool response wrapped in realistic metadata).
    """
    if spec.suite != "tool":
        return []
    views: List[PromptView] = []
    for rel in ("plausible", "irrelevant"):
        for direction in ("low", "high"):
            anchor_value = spec.anchors[direction]
            base = f"{rel}_{direction}"
            views.append(_build_realism_prompt(
                spec, f"{base}_elicited", rel, anchor_value,
                variant="elicited",
            ))
            views.append(_build_realism_prompt(
                spec, f"{base}_noisy", rel, anchor_value,
                variant="noisy",
            ))
    return views


def render_tool(spec: ItemSpec) -> List[PromptView]:
    """Render 5 conditions for a Tool item.

    Conditions: control, irrelevant_low, irrelevant_high,
                plausible_low, plausible_high.
    All share the same item stem and evidence; only check_external_reference
    output differs.
    """
    return [
        _build_prompt(spec, cond, rel, spec.anchors.get(d) if d else None)
        for cond, rel, d in CONDITIONS
    ]


# ── Structured message access for evaluation scripts ─────────────────

def get_tool_messages(
    spec: ItemSpec,
    condition: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (messages, tool_schemas_list) for a given item and condition.

    Used by evaluation scripts that apply the model's chat template
    at inference time rather than using the plaintext fallback.
    """
    relevance = "none"
    anchor_value = None
    for cond_name, rel, direction in CONDITIONS:
        if cond_name == condition:
            relevance = rel
            if direction:
                anchor_value = spec.anchors[direction]
            break

    messages, _, _ = _build_messages(spec, condition, relevance, anchor_value)
    return messages, TOOL_SCHEMAS
