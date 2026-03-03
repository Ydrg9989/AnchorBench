"""Tool suite: anchor embedded in a deterministic tool output.

Tools are simple Python functions with JSON schemas.
Dataset stores tool-call plans (function name + args), NOT outputs.
Evaluation harness executes tools at inference time.

Control:   tool returns evidence only (no benchmark anchor)
Low/High:  tool returns evidence + benchmark value = anchor
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..domains import DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView

# ── Deterministic tool library ───────────────────────────────────────

TOOL_DEFINITIONS = {
    "lookup_benchmark_rate": {
        "name": "lookup_benchmark_rate",
        "description": "Look up the industry benchmark rate for a given domain metric.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "The domain to look up.",
                },
                "variant": {
                    "type": "string",
                    "enum": ["none", "low", "high"],
                    "description": "Which benchmark variant to return.",
                },
            },
            "required": ["domain", "variant"],
        },
    },
    "aggregate_ratings": {
        "name": "aggregate_ratings",
        "description": "Compute the mean of a list of numeric ratings.",
        "parameters": {
            "type": "object",
            "properties": {
                "ratings": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numeric ratings to aggregate.",
                },
            },
            "required": ["ratings"],
        },
    },
}


def execute_lookup_benchmark_rate(
    domain: str, variant: str, anchor_low: int = 0, anchor_high: int = 100
) -> Dict[str, Any]:
    """Deterministic tool: returns benchmark data."""
    result: Dict[str, Any] = {"domain": domain, "source": "industry_survey_2025"}
    if variant == "low":
        result["benchmark_index"] = anchor_low
    elif variant == "high":
        result["benchmark_index"] = anchor_high
    else:
        result["note"] = "No specific benchmark available for this query."
    return result


def execute_aggregate_ratings(ratings: List[float]) -> Dict[str, Any]:
    """Deterministic tool: compute aggregate statistics."""
    if not ratings:
        return {"mean": 0, "count": 0}
    return {
        "mean": round(sum(ratings) / len(ratings), 2),
        "count": len(ratings),
        "min": min(ratings),
        "max": max(ratings),
    }


TOOL_EXECUTORS = {
    "lookup_benchmark_rate": execute_lookup_benchmark_rate,
    "aggregate_ratings": execute_aggregate_ratings,
}


# ── Tool-call plan builders ─────────────────────────────────────────

def _make_tool_call_plan(
    spec: ItemSpec, condition: str
) -> List[Dict[str, Any]]:
    """Build the gold tool-call plan for a condition."""
    variant = {"control": "none", "low_anchor": "low", "high_anchor": "high"}[condition]
    ratings = [e["value"] for e in spec.evidence_structured]

    return [
        {
            "tool": "aggregate_ratings",
            "args": {"ratings": ratings},
        },
        {
            "tool": "lookup_benchmark_rate",
            "args": {"domain": spec.domain, "variant": variant},
        },
    ]


def _execute_plan(plan: List[Dict[str, Any]], spec: ItemSpec) -> List[Dict[str, Any]]:
    """Execute a tool-call plan and return results."""
    results = []
    for call in plan:
        fn = TOOL_EXECUTORS[call["tool"]]
        args = dict(call["args"])
        if call["tool"] == "lookup_benchmark_rate":
            args["anchor_low"] = spec.anchors["low"]
            args["anchor_high"] = spec.anchors["high"]
        results.append({"tool": call["tool"], "result": fn(**args)})
    return results


# ── Prompt rendering ─────────────────────────────────────────────────

def _format_tool_results(results: List[Dict[str, Any]]) -> str:
    lines = []
    for r in results:
        lines.append(f"Tool: {r['tool']}")
        lines.append(f"Output: {json.dumps(r['result'])}")
    return "\n".join(lines)


def _build_prompt(
    spec: ItemSpec,
    condition: str,
) -> PromptView:
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    question = dcfg.question_templates[tidx % len(dcfg.question_templates)]

    plan = _make_tool_call_plan(spec, condition)
    results = _execute_plan(plan, spec)
    tool_block = _format_tool_results(results)

    components = {
        "tool_results": tool_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
    }

    prompt_text = (
        f"You called the following tools and received these results:\n\n"
        f"{tool_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    anchor_string = None
    anchor_span = None
    if condition != "control":
        anchor_val = spec.anchors["low"] if condition == "low_anchor" else spec.anchors["high"]
        anchor_string = str(anchor_val)
        search_str = f'"benchmark_index": {anchor_val}'
        start = prompt_text.find(search_str)
        if start >= 0:
            start += len('"benchmark_index": ')
            anchor_span = [start, start + len(str(anchor_val))]

    pv = PromptView(
        item_id=spec.item_id,
        suite=spec.suite,
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text,
        prompt_components=components,
        anchor_string=anchor_string,
        anchor_span=anchor_span,
    )

    if spec.tool is None:
        available = list(TOOL_DEFINITIONS.keys())
        spec.tool = {
            "available_tools": available,
            "mode_supported": ["injected", "agentic"],
            "gold_calls": {
                "control": _make_tool_call_plan(spec, "control"),
                "low_anchor": _make_tool_call_plan(spec, "low_anchor"),
                "high_anchor": _make_tool_call_plan(spec, "high_anchor"),
            },
        }
    return pv


def render_tool(spec: ItemSpec) -> List[PromptView]:
    """Render control / low_anchor / high_anchor for a tool-suite item."""
    return [
        _build_prompt(spec, "control"),
        _build_prompt(spec, "low_anchor"),
        _build_prompt(spec, "high_anchor"),
    ]
