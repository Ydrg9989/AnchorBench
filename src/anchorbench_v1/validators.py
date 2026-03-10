"""Deterministic validators + optional LLM spot-check for AnchorBench v1.

Deterministic checks:
  - theta in [30,70], anchors in [0,100], anchor_gap correctness
  - y_star in [0,100] and == round(theta)
  - evidence_structured present and consistent
  - external suite pairing: prompts differ only by anchor sentence
  - RAG: must_include doc assertions
  - tool: tool-call plans reference valid tools
  - prompt format constraint present
  - anchor_string exists in low/high prompts

Optional LLM spot-check (on sample fraction):
  - "No leakage of gold/theta/y_star"
  - "Anchor is normatively irrelevant"
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from .schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView
from .suites.tools import TOOL_DEFINITIONS
from .suites.tool_v2 import TOOL_V2_SCHEMAS


class ValidationError:
    def __init__(self, item_id: str, field: str, message: str):
        self.item_id = item_id
        self.field = field
        self.message = message

    def __repr__(self):
        return f"ValidationError({self.item_id}, {self.field}: {self.message})"


def _is_v2(spec: ItemSpec) -> bool:
    return spec.difficulty in ("easy", "hard")


def validate_itemspec(spec: ItemSpec) -> List[ValidationError]:
    """Run deterministic checks on a single ItemSpec (v1 items)."""
    errors = []
    iid = spec.item_id

    if getattr(spec, "suite", None) == "history_v2":
        return validate_history_v2_itemspec(spec)
    if getattr(spec, "suite", None) == "rag_v2":
        return validate_rag_v2_itemspec(spec)
    if getattr(spec, "suite", None) == "tool_v2":
        return validate_tool_v2_itemspec(spec)
    if getattr(spec, "suite", None) == "icl_v2":
        return validate_icl_v2_itemspec(spec)
    if _is_v2(spec):
        return validate_external_v2_itemspec(spec)

    if spec.tags.get("split") != "stress":
        if not (30 <= spec.theta <= 70):
            errors.append(ValidationError(iid, "theta", f"theta={spec.theta} not in [30,70]"))
        expected_gap = 60
        if spec.anchors.get("gap") != expected_gap:
            errors.append(ValidationError(iid, "anchors.gap", f"gap={spec.anchors.get('gap')} != {expected_gap}"))
        if spec.anchors.get("low") != spec.theta - 30:
            errors.append(ValidationError(iid, "anchors.low", f"low={spec.anchors.get('low')} != theta-30"))
        if spec.anchors.get("high") != spec.theta + 30:
            errors.append(ValidationError(iid, "anchors.high", f"high={spec.anchors.get('high')} != theta+30"))

    if not (0 <= spec.anchors.get("low", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.low", "low anchor out of [0,100]"))
    if not (0 <= spec.anchors.get("high", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.high", "high anchor out of [0,100]"))
    if not (0 <= spec.y_star <= 100):
        errors.append(ValidationError(iid, "y_star", f"y_star={spec.y_star} out of [0,100]"))
    if spec.y_star != round(spec.theta):
        errors.append(ValidationError(iid, "y_star", f"y_star={spec.y_star} != round(theta={spec.theta})"))

    if len(spec.evidence_structured) == 0:
        errors.append(ValidationError(iid, "evidence_structured", "empty evidence"))
    for e in spec.evidence_structured:
        if not (0 <= e.get("value", -1) <= 100):
            errors.append(ValidationError(iid, "evidence_structured", f"evidence value {e.get('value')} out of [0,100]"))

    if spec.answer_space != {"type": "int", "min": 0, "max": 100}:
        errors.append(ValidationError(iid, "answer_space", "unexpected answer_space"))

    if spec.suite == "tool" and spec.tool:
        for cond_plans in spec.tool.get("gold_calls", {}).values():
            for call in cond_plans:
                if call["tool"] not in TOOL_DEFINITIONS:
                    errors.append(ValidationError(iid, "tool", f"unknown tool {call['tool']!r}"))

    return errors


_V1_ANCHORED = {"low_anchor", "high_anchor"}
_V2_ANCHORED = {"irrelevant_low", "irrelevant_high", "plausible_low", "plausible_high"}


def validate_promptview(pv: PromptView) -> List[ValidationError]:
    """Run deterministic checks on a single PromptView."""
    errors = []
    iid = pv.item_id

    if ANSWER_FORMAT_INSTRUCTION not in pv.prompt_text:
        errors.append(ValidationError(iid, "prompt_text", "missing answer format instruction"))

    is_anchored = pv.condition in _V1_ANCHORED or pv.condition in _V2_ANCHORED
    is_history_v2 = getattr(pv, "suite", None) == "history_v2"
    if is_anchored and not is_history_v2:
        if pv.anchor_string is None:
            errors.append(ValidationError(iid, "anchor_string", "anchor_string is None for anchored condition"))
        elif pv.anchor_string not in pv.prompt_text:
            errors.append(ValidationError(iid, "anchor_string", f"anchor_string {pv.anchor_string!r} not found in prompt"))

    is_rag_v2 = getattr(pv, "suite", None) == "rag_v2"
    is_tool_v2 = getattr(pv, "suite", None) == "tool_v2"
    is_icl_v2 = getattr(pv, "suite", None) == "icl_v2"
    if is_icl_v2:
        errors.extend(validate_icl_v2_promptview(pv))
    elif is_tool_v2:
        errors.extend(validate_tool_v2_promptview(pv))
    elif is_rag_v2:
        errors.extend(validate_rag_v2_promptview(pv))
    elif is_history_v2:
        errors.extend(validate_history_v2_promptview(pv))
    elif pv.condition in _V2_ANCHORED:
        errors.extend(validate_external_v2_promptview(pv))

    if not pv.prompt_hash:
        errors.append(ValidationError(iid, "prompt_hash", "missing prompt_hash"))

    return errors


_V1_EXTERNAL_CONDS = {"control", "low_anchor", "high_anchor"}
_V2_EXTERNAL_CONDS = {
    "control", "irrelevant_low", "irrelevant_high",
    "plausible_low", "plausible_high",
}


def validate_external_pairing(views: List[PromptView]) -> List[ValidationError]:
    """Check that external-suite prompts differ only by anchor sentence."""
    errors = []
    by_item: Dict[str, Dict[str, PromptView]] = {}
    for pv in views:
        if pv.suite == "external":
            by_item.setdefault(pv.item_id, {})[pv.condition] = pv

    for iid, conds in by_item.items():
        cond_keys = set(conds.keys())

        if cond_keys == _V2_EXTERNAL_CONDS:
            errors.extend(_validate_v2_pairing(iid, conds))
        elif cond_keys == _V1_EXTERNAL_CONDS:
            errors.extend(_validate_v1_pairing(iid, conds))
        else:
            errors.append(ValidationError(
                iid, "pairing",
                f"unexpected condition set: {sorted(cond_keys)}",
            ))

    return errors


def _validate_v1_pairing(
    iid: str, conds: Dict[str, PromptView]
) -> List[ValidationError]:
    errors = []
    ctrl = conds["control"]
    for cname in ("low_anchor", "high_anchor"):
        anch = conds[cname]
        anchor_sent = anch.prompt_components.get("anchor_sentence", "")
        if not anchor_sent:
            errors.append(ValidationError(iid, "pairing", f"{cname} missing anchor_sentence component"))
            continue
        expected = ctrl.prompt_text.replace(
            anch.prompt_components.get("question", ""),
            f"{anchor_sent} {anch.prompt_components.get('question', '')}",
        )
        if anch.prompt_text != expected:
            ctrl_lines = ctrl.prompt_text.splitlines()
            anch_lines = anch.prompt_text.splitlines()
            diff_count = sum(1 for a, b in zip(ctrl_lines, anch_lines) if a != b)
            if diff_count > 2:
                errors.append(ValidationError(
                    iid, "pairing",
                    f"control and {cname} differ by {diff_count} lines (expected ≤ 2)",
                ))
    return errors


def _validate_v2_pairing(
    iid: str, conds: Dict[str, PromptView]
) -> List[ValidationError]:
    """Check that all 5 v2 conditions share the same stem."""
    errors = []
    ctrl = conds["control"]
    ctrl_evidence = ctrl.prompt_components.get("evidence", "")
    ctrl_scenario = ctrl.prompt_components.get("scenario", "")

    anchored_conds = [c for c in _V2_EXTERNAL_CONDS if c != "control"]
    for cname in anchored_conds:
        anch = conds[cname]
        anchor_sent = anch.prompt_components.get("anchor_sentence", "")
        if not anchor_sent:
            errors.append(ValidationError(
                iid, "pairing", f"{cname} missing anchor_sentence component"
            ))
            continue

        if anch.prompt_components.get("evidence", "") != ctrl_evidence:
            errors.append(ValidationError(
                iid, "pairing", f"{cname} evidence differs from control"
            ))
        if anch.prompt_components.get("scenario", "") != ctrl_scenario:
            errors.append(ValidationError(
                iid, "pairing", f"{cname} scenario differs from control"
            ))

        expected = ctrl.prompt_text.replace(
            anch.prompt_components.get("question", ""),
            f"{anchor_sent} {anch.prompt_components.get('question', '')}",
        )
        if anch.prompt_text != expected:
            ctrl_lines = ctrl.prompt_text.splitlines()
            anch_lines = anch.prompt_text.splitlines()
            diff_count = sum(1 for a, b in zip(ctrl_lines, anch_lines) if a != b)
            if diff_count > 2:
                errors.append(ValidationError(
                    iid, "pairing",
                    f"control and {cname} differ by {diff_count} lines (expected ≤ 2)",
                ))

    hashes = [conds[c].prompt_hash for c in sorted(conds)]
    if len(set(hashes)) != len(hashes):
        errors.append(ValidationError(
            iid, "pairing", "duplicate prompt hashes among conditions"
        ))

    return errors


# ── External v2 validators ───────────────────────────────────────────

def validate_external_v2_itemspec(spec: ItemSpec) -> List[ValidationError]:
    """Validate a v2 External ItemSpec."""
    errors = []
    iid = spec.item_id

    if not (30 <= spec.theta <= 70):
        errors.append(ValidationError(iid, "theta", f"theta={spec.theta} not in [30,70]"))

    if spec.difficulty not in ("easy", "hard"):
        errors.append(ValidationError(
            iid, "difficulty", f"difficulty={spec.difficulty!r} not in (easy, hard)"
        ))

    if not (0 <= spec.y_star_theta <= 100):
        errors.append(ValidationError(
            iid, "y_star_theta", f"y_star_theta={spec.y_star_theta} out of [0,100]"
        ))
    if not (0 <= spec.y_star_evidence <= 100):
        errors.append(ValidationError(
            iid, "y_star_evidence", f"y_star_evidence={spec.y_star_evidence} out of [0,100]"
        ))
    if spec.y_star_theta != spec.theta:
        errors.append(ValidationError(
            iid, "y_star_theta", f"y_star_theta={spec.y_star_theta} != theta={spec.theta}"
        ))
    if spec.y_star != spec.y_star_evidence:
        errors.append(ValidationError(
            iid, "y_star",
            f"y_star={spec.y_star} != y_star_evidence={spec.y_star_evidence}",
        ))

    if not (0 <= spec.anchors.get("low", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.low", "low anchor out of [0,100]"))
    if not (0 <= spec.anchors.get("high", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.high", "high anchor out of [0,100]"))

    offset = spec.anchors.get("offset", 0)
    if offset not in (15, 25, 40):
        errors.append(ValidationError(
            iid, "anchors.offset", f"offset={offset} not in {{15, 25, 40}}"
        ))

    if len(spec.evidence_structured) == 0:
        errors.append(ValidationError(iid, "evidence_structured", "empty evidence"))

    missing_count = sum(1 for e in spec.evidence_structured if e.get("missing"))
    visible_count = len(spec.evidence_structured) - missing_count

    if spec.difficulty == "easy" and missing_count != 0:
        errors.append(ValidationError(
            iid, "evidence_structured",
            f"easy item has {missing_count} missing evidence (expected 0)",
        ))
    if spec.difficulty == "hard" and missing_count != 2:
        errors.append(ValidationError(
            iid, "evidence_structured",
            f"hard item has {missing_count} missing evidence (expected 2)",
        ))

    for e in spec.evidence_structured:
        if not e.get("missing"):
            v = e.get("value", -1)
            if v is None or not (0 <= v <= 100):
                errors.append(ValidationError(
                    iid, "evidence_structured",
                    f"visible evidence value {v} out of [0,100]",
                ))

    visible_vals = [
        e["value"] for e in spec.evidence_structured if not e.get("missing")
    ]
    if visible_vals:
        expected_y = round(sum(visible_vals) / len(visible_vals))
        if spec.y_star_evidence != expected_y:
            errors.append(ValidationError(
                iid, "y_star_evidence",
                f"y_star_evidence={spec.y_star_evidence} != "
                f"round(mean(visible))={expected_y}",
            ))

    if spec.answer_space != {"type": "int", "min": 0, "max": 100}:
        errors.append(ValidationError(iid, "answer_space", "unexpected answer_space"))

    return errors


def validate_external_v2_promptview(pv: PromptView) -> List[ValidationError]:
    """Extra checks for v2 PromptView (called from validate_promptview)."""
    errors = []
    iid = pv.item_id

    if pv.anchor_relevance not in ("none", "irrelevant", "plausible"):
        errors.append(ValidationError(
            iid, "anchor_relevance",
            f"anchor_relevance={pv.anchor_relevance!r} invalid",
        ))

    if pv.condition == "control":
        if pv.anchor_relevance != "none":
            errors.append(ValidationError(
                iid, "anchor_relevance",
                "control condition should have anchor_relevance='none'",
            ))
        if pv.anchor_value is not None:
            errors.append(ValidationError(
                iid, "anchor_value", "control has non-None anchor_value"
            ))
    else:
        if pv.anchor_value is None:
            errors.append(ValidationError(
                iid, "anchor_value", f"{pv.condition} has None anchor_value"
            ))

        if "irrelevant" in pv.condition:
            if pv.anchor_relevance != "irrelevant":
                errors.append(ValidationError(
                    iid, "anchor_relevance",
                    f"{pv.condition} should have relevance='irrelevant'",
                ))
            if pv.anchor_string and f"case #{pv.anchor_string}" not in pv.prompt_text:
                errors.append(ValidationError(
                    iid, "prompt_text",
                    f"irrelevant anchor pattern 'case #{pv.anchor_string}' "
                    f"not found in prompt",
                ))
        elif "plausible" in pv.condition:
            if pv.anchor_relevance != "plausible":
                errors.append(ValidationError(
                    iid, "anchor_relevance",
                    f"{pv.condition} should have relevance='plausible'",
                ))
            if pv.anchor_string and f"around {pv.anchor_string}" not in pv.prompt_text:
                if pv.anchor_string and f"near {pv.anchor_string}" not in pv.prompt_text:
                    errors.append(ValidationError(
                        iid, "prompt_text",
                        f"plausible anchor pattern not found in prompt "
                        f"(expected 'around {pv.anchor_string}' or "
                        f"'near {pv.anchor_string}')",
                    ))

    return errors


# ── History v2 validators ─────────────────────────────────────────────

def validate_history_v2_itemspec(spec: ItemSpec) -> List[ValidationError]:
    """Validate History v2 ItemSpec: history dict, same-domain warmup, subset pressure."""
    errors = []
    iid = spec.item_id

    if spec.suite != "history_v2":
        errors.append(ValidationError(iid, "suite", "expected suite history_v2"))
    if not spec.history:
        errors.append(ValidationError(iid, "history", "missing history dict"))
        return errors

    h = spec.history
    # Subset indices valid (0..4)
    for key in ("subset_indices_low", "subset_indices_high"):
        idx = h.get(key, [])
        if not isinstance(idx, list) or len(idx) != 2:
            errors.append(ValidationError(iid, f"history.{key}", f"expected list of 2 indices, got {idx!r}"))
        else:
            for i in idx:
                if not (0 <= i < 5):
                    errors.append(ValidationError(iid, f"history.{key}", f"index {i} not in [0,4]"))

    # Low vs high subset pressure differs (subset means should bracket or differ from y_star_evidence)
    y = spec.y_star_evidence
    mean_low = h.get("subset_mean_low")
    mean_high = h.get("subset_mean_high")
    if mean_low is not None and mean_high is not None and y is not None:
        if abs(mean_low - mean_high) < 1e-6:
            errors.append(ValidationError(
                iid, "history",
                f"low and high subset means should differ (got {mean_low}, {mean_high})",
            ))
        elif (mean_low - y) * (mean_high - y) > 0:
            errors.append(ValidationError(
                iid, "history",
                f"low/high subset pressure should bracket y_star_evidence={y} (got means {mean_low}, {mean_high})",
            ))

    # Warmup and target same domain; warmup has different case (different evidence)
    for key in ("warmup_low", "warmup_high"):
        w = h.get(key, {})
        if not w:
            errors.append(ValidationError(iid, f"history.{key}", f"missing {key}"))
            continue
        if "evidence_structured" not in w or "y_star_evidence" not in w:
            errors.append(ValidationError(iid, f"history.{key}", "warmup must have evidence_structured and y_star_evidence"))
        # Same domain: warmup is built from same domain in itemspec_gen; no separate domain field in warmup

    # y_star_evidence is target item gold
    if not (0 <= spec.y_star_evidence <= 100):
        errors.append(ValidationError(iid, "y_star_evidence", f"y_star_evidence={spec.y_star_evidence} out of [0,100]"))

    return errors


def validate_history_v2_promptview(pv: PromptView) -> List[ValidationError]:
    """Validate History v2 PromptView: two-stage components, anchor_relevance, control has no prior."""
    errors = []
    iid = pv.item_id

    if pv.condition == "control":
        if "stage1_user_message" in (pv.prompt_components or {}):
            errors.append(ValidationError(iid, "prompt_components", "control should not have stage1_user_message"))
        if pv.anchor_relevance != "none":
            errors.append(ValidationError(iid, "anchor_relevance", "control should have anchor_relevance=none"))
    else:
        comp = pv.prompt_components or {}
        if "stage1_user_message" not in comp:
            errors.append(ValidationError(iid, "prompt_components", f"{pv.condition} missing stage1_user_message"))
        if "stage2_user_message" not in comp:
            errors.append(ValidationError(iid, "prompt_components", f"{pv.condition} missing stage2_user_message"))
        if "plausible" in pv.condition and pv.anchor_relevance != "plausible":
            errors.append(ValidationError(iid, "anchor_relevance", f"{pv.condition} should have anchor_relevance=plausible"))
        if "irrelevant" in pv.condition and pv.anchor_relevance != "irrelevant":
            errors.append(ValidationError(iid, "anchor_relevance", f"{pv.condition} should have anchor_relevance=irrelevant"))

    if not pv.prompt_hash:
        errors.append(ValidationError(iid, "prompt_hash", "missing prompt_hash"))

    return errors


# ── RAG v2 validators ─────────────────────────────────────────────────

_RAG_V2_CONDS = {
    "control", "irrelevant_low", "irrelevant_high",
    "plausible_low", "plausible_high",
}


def validate_rag_v2_itemspec(spec: ItemSpec) -> List[ValidationError]:
    """Validate RAG v2 ItemSpec: rag dict, evidence, anchors, difficulty."""
    errors = []
    iid = spec.item_id

    if spec.suite != "rag_v2":
        errors.append(ValidationError(iid, "suite", "expected suite rag_v2"))

    if not spec.rag:
        errors.append(ValidationError(iid, "rag", "missing rag dict"))
    else:
        if spec.rag.get("corpus_size") != 3:
            errors.append(ValidationError(
                iid, "rag.corpus_size",
                f"expected corpus_size=3, got {spec.rag.get('corpus_size')}",
            ))
        if spec.rag.get("retrieval_method") != "deterministic_fixed_order":
            errors.append(ValidationError(
                iid, "rag.retrieval_method",
                f"expected deterministic_fixed_order, got {spec.rag.get('retrieval_method')!r}",
            ))

    if not (30 <= spec.theta <= 70):
        errors.append(ValidationError(iid, "theta", f"theta={spec.theta} not in [30,70]"))

    if spec.difficulty not in ("easy", "hard"):
        errors.append(ValidationError(
            iid, "difficulty", f"difficulty={spec.difficulty!r} not in (easy, hard)"
        ))

    if not (0 <= spec.y_star_evidence <= 100):
        errors.append(ValidationError(
            iid, "y_star_evidence", f"y_star_evidence={spec.y_star_evidence} out of [0,100]"
        ))
    if spec.y_star != spec.y_star_evidence:
        errors.append(ValidationError(
            iid, "y_star",
            f"y_star={spec.y_star} != y_star_evidence={spec.y_star_evidence}",
        ))

    if not (0 <= spec.anchors.get("low", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.low", "low anchor out of [0,100]"))
    if not (0 <= spec.anchors.get("high", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.high", "high anchor out of [0,100]"))

    offset = spec.anchors.get("offset", 0)
    if offset not in (15, 25, 40):
        errors.append(ValidationError(
            iid, "anchors.offset", f"offset={offset} not in {{15, 25, 40}}"
        ))

    if len(spec.evidence_structured) == 0:
        errors.append(ValidationError(iid, "evidence_structured", "empty evidence"))

    missing_count = sum(1 for e in spec.evidence_structured if e.get("missing"))
    if spec.difficulty == "easy" and missing_count != 0:
        errors.append(ValidationError(
            iid, "evidence_structured",
            f"easy item has {missing_count} missing evidence (expected 0)",
        ))
    if spec.difficulty == "hard" and missing_count != 2:
        errors.append(ValidationError(
            iid, "evidence_structured",
            f"hard item has {missing_count} missing evidence (expected 2)",
        ))

    visible_vals = [
        e["value"] for e in spec.evidence_structured if not e.get("missing")
    ]
    if visible_vals:
        expected_y = round(sum(visible_vals) / len(visible_vals))
        if spec.y_star_evidence != expected_y:
            errors.append(ValidationError(
                iid, "y_star_evidence",
                f"y_star_evidence={spec.y_star_evidence} != "
                f"round(mean(visible))={expected_y}",
            ))

    return errors


def validate_rag_v2_promptview(pv: PromptView) -> List[ValidationError]:
    """Validate RAG v2 PromptView: anchor relevance, anchor presence, doc structure."""
    errors = []
    iid = pv.item_id

    if pv.condition not in _RAG_V2_CONDS:
        errors.append(ValidationError(
            iid, "condition", f"unexpected RAG v2 condition {pv.condition!r}"
        ))

    if pv.condition == "control":
        if pv.anchor_relevance != "none":
            errors.append(ValidationError(
                iid, "anchor_relevance",
                "control should have anchor_relevance='none'",
            ))
        if pv.anchor_value is not None:
            errors.append(ValidationError(
                iid, "anchor_value", "control has non-None anchor_value"
            ))
        if pv.anchor_string is not None:
            errors.append(ValidationError(
                iid, "anchor_string", "control has non-None anchor_string"
            ))
    else:
        if pv.anchor_value is None:
            errors.append(ValidationError(
                iid, "anchor_value", f"{pv.condition} has None anchor_value"
            ))
        if pv.anchor_string is None:
            errors.append(ValidationError(
                iid, "anchor_string", f"{pv.condition} has None anchor_string"
            ))

        if "irrelevant" in pv.condition:
            if pv.anchor_relevance != "irrelevant":
                errors.append(ValidationError(
                    iid, "anchor_relevance",
                    f"{pv.condition} should have relevance='irrelevant'",
                ))
            if pv.anchor_string and f"#{pv.anchor_string}" not in pv.prompt_text:
                errors.append(ValidationError(
                    iid, "prompt_text",
                    f"irrelevant anchor '#{pv.anchor_string}' not in prompt",
                ))
        elif "plausible" in pv.condition:
            if pv.anchor_relevance != "plausible":
                errors.append(ValidationError(
                    iid, "anchor_relevance",
                    f"{pv.condition} should have relevance='plausible'",
                ))
            if pv.anchor_string and f"approximately {pv.anchor_string}" not in pv.prompt_text:
                errors.append(ValidationError(
                    iid, "prompt_text",
                    f"plausible anchor 'approximately {pv.anchor_string}' not in prompt",
                ))

    comp = pv.prompt_components or {}
    if "retrieved_doc_ids" not in comp:
        errors.append(ValidationError(
            iid, "prompt_components", "missing retrieved_doc_ids"
        ))
    if "anchor_doc_id" not in comp:
        errors.append(ValidationError(
            iid, "prompt_components", "missing anchor_doc_id"
        ))

    if not pv.prompt_hash:
        errors.append(ValidationError(iid, "prompt_hash", "missing prompt_hash"))

    return errors


def validate_rag_v2_pairing(views: List[PromptView]) -> List[ValidationError]:
    """Check that RAG v2 matched conditions share the same stem.

    For each item: evidence, scenario, question, doc_core, doc_filler must
    be identical across all 5 conditions. Only doc_anchor_slot differs.
    """
    errors = []
    by_item: Dict[str, Dict[str, PromptView]] = {}
    for pv in views:
        if pv.suite == "rag_v2":
            by_item.setdefault(pv.item_id, {})[pv.condition] = pv

    for iid, conds in by_item.items():
        cond_keys = set(conds.keys())
        if cond_keys != _RAG_V2_CONDS:
            errors.append(ValidationError(
                iid, "pairing",
                f"expected {sorted(_RAG_V2_CONDS)} conditions, got {sorted(cond_keys)}",
            ))
            continue

        ctrl = conds["control"]
        ctrl_evidence = ctrl.prompt_components.get("evidence", "")
        ctrl_scenario = ctrl.prompt_components.get("scenario", "")
        ctrl_question = ctrl.prompt_components.get("question", "")

        for cname in sorted(_RAG_V2_CONDS - {"control"}):
            anch = conds[cname]
            if anch.prompt_components.get("evidence", "") != ctrl_evidence:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} evidence differs from control"
                ))
            if anch.prompt_components.get("scenario", "") != ctrl_scenario:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} scenario differs from control"
                ))
            if anch.prompt_components.get("question", "") != ctrl_question:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} question differs from control"
                ))

        hashes = [conds[c].prompt_hash for c in sorted(conds)]
        if len(set(hashes)) != len(hashes):
            errors.append(ValidationError(
                iid, "pairing", "duplicate prompt hashes among conditions"
            ))

    return errors


# ── Tool v2 validators ─────────────────────────────────────────────────

_TOOL_V2_CONDS = {
    "control", "irrelevant_low", "irrelevant_high",
    "plausible_low", "plausible_high",
}


def validate_tool_v2_itemspec(spec: ItemSpec) -> List[ValidationError]:
    """Validate Tool v2 ItemSpec: tool dict, evidence, anchors, difficulty."""
    errors = []
    iid = spec.item_id

    if spec.suite != "tool_v2":
        errors.append(ValidationError(iid, "suite", "expected suite tool_v2"))

    if not spec.tool:
        errors.append(ValidationError(iid, "tool", "missing tool dict"))
    else:
        expected_tools = {"get_evidence_summary", "check_external_reference"}
        actual_tools = set(spec.tool.get("available_tools", []))
        if actual_tools != expected_tools:
            errors.append(ValidationError(
                iid, "tool.available_tools",
                f"expected {sorted(expected_tools)}, got {sorted(actual_tools)}",
            ))
        if spec.tool.get("tool_schema_version") != "v2":
            errors.append(ValidationError(
                iid, "tool.tool_schema_version",
                f"expected 'v2', got {spec.tool.get('tool_schema_version')!r}",
            ))

    if not (30 <= spec.theta <= 70):
        errors.append(ValidationError(iid, "theta", f"theta={spec.theta} not in [30,70]"))

    if spec.difficulty not in ("easy", "hard"):
        errors.append(ValidationError(
            iid, "difficulty", f"difficulty={spec.difficulty!r} not in (easy, hard)"
        ))

    if not (0 <= spec.y_star_evidence <= 100):
        errors.append(ValidationError(
            iid, "y_star_evidence", f"y_star_evidence={spec.y_star_evidence} out of [0,100]"
        ))
    if spec.y_star != spec.y_star_evidence:
        errors.append(ValidationError(
            iid, "y_star",
            f"y_star={spec.y_star} != y_star_evidence={spec.y_star_evidence}",
        ))

    if not (0 <= spec.anchors.get("low", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.low", "low anchor out of [0,100]"))
    if not (0 <= spec.anchors.get("high", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.high", "high anchor out of [0,100]"))

    offset = spec.anchors.get("offset", 0)
    if offset not in (15, 25, 40):
        errors.append(ValidationError(
            iid, "anchors.offset", f"offset={offset} not in {{15, 25, 40}}"
        ))

    if len(spec.evidence_structured) == 0:
        errors.append(ValidationError(iid, "evidence_structured", "empty evidence"))

    missing_count = sum(1 for e in spec.evidence_structured if e.get("missing"))
    if spec.difficulty == "easy" and missing_count != 0:
        errors.append(ValidationError(
            iid, "evidence_structured",
            f"easy item has {missing_count} missing evidence (expected 0)",
        ))
    if spec.difficulty == "hard" and missing_count != 2:
        errors.append(ValidationError(
            iid, "evidence_structured",
            f"hard item has {missing_count} missing evidence (expected 2)",
        ))

    visible_vals = [
        e["value"] for e in spec.evidence_structured if not e.get("missing")
    ]
    if visible_vals:
        expected_y = round(sum(visible_vals) / len(visible_vals))
        if spec.y_star_evidence != expected_y:
            errors.append(ValidationError(
                iid, "y_star_evidence",
                f"y_star_evidence={spec.y_star_evidence} != "
                f"round(mean(visible))={expected_y}",
            ))

    return errors


def validate_tool_v2_promptview(pv: PromptView) -> List[ValidationError]:
    """Validate Tool v2 PromptView: anchor placement in correct tool output field."""
    errors = []
    iid = pv.item_id

    if pv.condition not in _TOOL_V2_CONDS:
        errors.append(ValidationError(
            iid, "condition", f"unexpected tool v2 condition {pv.condition!r}"
        ))

    if pv.condition == "control":
        if pv.anchor_relevance != "none":
            errors.append(ValidationError(
                iid, "anchor_relevance",
                "control should have anchor_relevance='none'",
            ))
        if pv.anchor_value is not None:
            errors.append(ValidationError(
                iid, "anchor_value", "control has non-None anchor_value"
            ))
        if pv.anchor_string is not None:
            errors.append(ValidationError(
                iid, "anchor_string", "control has non-None anchor_string"
            ))
    else:
        if pv.anchor_value is None:
            errors.append(ValidationError(
                iid, "anchor_value", f"{pv.condition} has None anchor_value"
            ))
        if pv.anchor_string is None:
            errors.append(ValidationError(
                iid, "anchor_string", f"{pv.condition} has None anchor_string"
            ))

        if "irrelevant" in pv.condition:
            if pv.anchor_relevance != "irrelevant":
                errors.append(ValidationError(
                    iid, "anchor_relevance",
                    f"{pv.condition} should have relevance='irrelevant'",
                ))
            if pv.anchor_string and f'"request_id": {pv.anchor_string}' not in pv.prompt_text:
                errors.append(ValidationError(
                    iid, "prompt_text",
                    f"irrelevant anchor 'request_id: {pv.anchor_string}' not in prompt",
                ))
        elif "plausible" in pv.condition:
            if pv.anchor_relevance != "plausible":
                errors.append(ValidationError(
                    iid, "anchor_relevance",
                    f"{pv.condition} should have relevance='plausible'",
                ))
            if pv.anchor_string and f'"reference_value": {pv.anchor_string}' not in pv.prompt_text:
                errors.append(ValidationError(
                    iid, "prompt_text",
                    f"plausible anchor 'reference_value: {pv.anchor_string}' not in prompt",
                ))

    prov = pv.provenance or {}
    if prov.get("tool_name") != "check_external_reference" and pv.condition != "control":
        errors.append(ValidationError(
            iid, "provenance",
            f"expected tool_name='check_external_reference', got {prov.get('tool_name')!r}",
        ))

    comp = pv.prompt_components or {}
    if "evidence_summary" not in comp:
        errors.append(ValidationError(
            iid, "prompt_components", "missing evidence_summary"
        ))
    if "reference_result" not in comp:
        errors.append(ValidationError(
            iid, "prompt_components", "missing reference_result"
        ))

    if not pv.prompt_hash:
        errors.append(ValidationError(iid, "prompt_hash", "missing prompt_hash"))

    return errors


def validate_tool_v2_pairing(views: List[PromptView]) -> List[ValidationError]:
    """Check that Tool v2 matched conditions share the same stem.

    For each item: evidence, scenario, question, get_evidence_summary output
    must be identical across all 5 conditions. Only check_external_reference
    output differs.
    """
    errors = []
    by_item: Dict[str, Dict[str, PromptView]] = {}
    for pv in views:
        if pv.suite == "tool_v2":
            by_item.setdefault(pv.item_id, {})[pv.condition] = pv

    for iid, conds in by_item.items():
        cond_keys = set(conds.keys())
        if cond_keys != _TOOL_V2_CONDS:
            errors.append(ValidationError(
                iid, "pairing",
                f"expected {sorted(_TOOL_V2_CONDS)} conditions, got {sorted(cond_keys)}",
            ))
            continue

        ctrl = conds["control"]
        ctrl_evidence = ctrl.prompt_components.get("evidence", "")
        ctrl_scenario = ctrl.prompt_components.get("scenario", "")
        ctrl_question = ctrl.prompt_components.get("question", "")
        ctrl_summary = ctrl.prompt_components.get("evidence_summary", "")

        for cname in sorted(_TOOL_V2_CONDS - {"control"}):
            anch = conds[cname]
            if anch.prompt_components.get("evidence", "") != ctrl_evidence:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} evidence differs from control"
                ))
            if anch.prompt_components.get("scenario", "") != ctrl_scenario:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} scenario differs from control"
                ))
            if anch.prompt_components.get("question", "") != ctrl_question:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} question differs from control"
                ))
            if anch.prompt_components.get("evidence_summary", "") != ctrl_summary:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} evidence_summary differs from control"
                ))

        hashes = [conds[c].prompt_hash for c in sorted(conds)]
        if len(set(hashes)) != len(hashes):
            errors.append(ValidationError(
                iid, "pairing", "duplicate prompt hashes among conditions"
            ))

    return errors


# ── ICL v2 validators ──────────────────────────────────────────────────

_ICL_V2_CONDS = {
    "control", "irrelevant_low", "irrelevant_high",
    "plausible_low", "plausible_high",
}


def validate_icl_v2_itemspec(spec: ItemSpec) -> List[ValidationError]:
    """Validate ICL v2 ItemSpec: evidence, anchors, difficulty, icl_demos tag."""
    errors = []
    iid = spec.item_id

    if spec.suite != "icl_v2":
        errors.append(ValidationError(iid, "suite", "expected suite icl_v2"))

    if not (30 <= spec.theta <= 70):
        errors.append(ValidationError(iid, "theta", f"theta={spec.theta} not in [30,70]"))

    if spec.difficulty not in ("easy", "hard"):
        errors.append(ValidationError(
            iid, "difficulty", f"difficulty={spec.difficulty!r} not in (easy, hard)"
        ))

    if not (0 <= spec.y_star_evidence <= 100):
        errors.append(ValidationError(
            iid, "y_star_evidence", f"y_star_evidence={spec.y_star_evidence} out of [0,100]"
        ))
    if spec.y_star != spec.y_star_evidence:
        errors.append(ValidationError(
            iid, "y_star",
            f"y_star={spec.y_star} != y_star_evidence={spec.y_star_evidence}",
        ))

    if not (0 <= spec.anchors.get("low", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.low", "low anchor out of [0,100]"))
    if not (0 <= spec.anchors.get("high", -1) <= 100):
        errors.append(ValidationError(iid, "anchors.high", "high anchor out of [0,100]"))

    offset = spec.anchors.get("offset", 0)
    if offset not in (15, 25, 40):
        errors.append(ValidationError(
            iid, "anchors.offset", f"offset={offset} not in {{15, 25, 40}}"
        ))

    if len(spec.evidence_structured) == 0:
        errors.append(ValidationError(iid, "evidence_structured", "empty evidence"))

    missing_count = sum(1 for e in spec.evidence_structured if e.get("missing"))
    if spec.difficulty == "easy" and missing_count != 0:
        errors.append(ValidationError(
            iid, "evidence_structured",
            f"easy item has {missing_count} missing evidence (expected 0)",
        ))
    if spec.difficulty == "hard" and missing_count != 2:
        errors.append(ValidationError(
            iid, "evidence_structured",
            f"hard item has {missing_count} missing evidence (expected 2)",
        ))

    visible_vals = [
        e["value"] for e in spec.evidence_structured if not e.get("missing")
    ]
    if visible_vals:
        expected_y = round(sum(visible_vals) / len(visible_vals))
        if spec.y_star_evidence != expected_y:
            errors.append(ValidationError(
                iid, "y_star_evidence",
                f"y_star_evidence={spec.y_star_evidence} != "
                f"round(mean(visible))={expected_y}",
            ))

    icl_demos = spec.tags.get("icl_demos")
    if not icl_demos:
        errors.append(ValidationError(iid, "tags.icl_demos", "missing icl_demos in tags"))
    elif not isinstance(icl_demos, list) or len(icl_demos) != 3:
        errors.append(ValidationError(
            iid, "tags.icl_demos",
            f"expected list of 3 demos, got {type(icl_demos).__name__} len={len(icl_demos) if isinstance(icl_demos, list) else '?'}",
        ))
    else:
        for di, demo in enumerate(icl_demos):
            if "evidence" not in demo or "answer" not in demo:
                errors.append(ValidationError(
                    iid, f"tags.icl_demos[{di}]", "demo missing 'evidence' or 'answer'"
                ))
            elif not isinstance(demo["evidence"], list) or len(demo["evidence"]) != 3:
                errors.append(ValidationError(
                    iid, f"tags.icl_demos[{di}]",
                    f"demo evidence should have 3 entries, got {len(demo['evidence']) if isinstance(demo['evidence'], list) else '?'}",
                ))
            if not (0 <= demo.get("answer", -1) <= 100):
                errors.append(ValidationError(
                    iid, f"tags.icl_demos[{di}]",
                    f"demo answer {demo.get('answer')} out of [0,100]",
                ))

    return errors


def validate_icl_v2_promptview(pv: PromptView) -> List[ValidationError]:
    """Validate ICL v2 PromptView: anchor placement in demo headers only."""
    errors = []
    iid = pv.item_id

    if pv.condition not in _ICL_V2_CONDS:
        errors.append(ValidationError(
            iid, "condition", f"unexpected ICL v2 condition {pv.condition!r}"
        ))

    if pv.condition == "control":
        if pv.anchor_relevance != "none":
            errors.append(ValidationError(
                iid, "anchor_relevance",
                "control should have anchor_relevance='none'",
            ))
        if pv.anchor_value is not None:
            errors.append(ValidationError(
                iid, "anchor_value", "control has non-None anchor_value"
            ))
        if pv.anchor_string is not None:
            errors.append(ValidationError(
                iid, "anchor_string", "control has non-None anchor_string"
            ))
    else:
        if pv.anchor_value is None:
            errors.append(ValidationError(
                iid, "anchor_value", f"{pv.condition} has None anchor_value"
            ))
        if pv.anchor_string is None:
            errors.append(ValidationError(
                iid, "anchor_string", f"{pv.condition} has None anchor_string"
            ))

        if "irrelevant" in pv.condition:
            if pv.anchor_relevance != "irrelevant":
                errors.append(ValidationError(
                    iid, "anchor_relevance",
                    f"{pv.condition} should have relevance='irrelevant'",
                ))
            if pv.anchor_string and f"Case #{pv.anchor_string}" not in pv.prompt_text:
                errors.append(ValidationError(
                    iid, "prompt_text",
                    f"irrelevant anchor 'Case #{pv.anchor_string}' not in prompt",
                ))
        elif "plausible" in pv.condition:
            if pv.anchor_relevance != "plausible":
                errors.append(ValidationError(
                    iid, "anchor_relevance",
                    f"{pv.condition} should have relevance='plausible'",
                ))
            if pv.anchor_string and f"similar cases: {pv.anchor_string}" not in pv.prompt_text:
                errors.append(ValidationError(
                    iid, "prompt_text",
                    f"plausible anchor 'similar cases: {pv.anchor_string}' not in prompt",
                ))

    comp = pv.prompt_components or {}
    if "demo_answers" not in comp:
        errors.append(ValidationError(
            iid, "prompt_components", "missing demo_answers"
        ))
    if "demo_evidence" not in comp:
        errors.append(ValidationError(
            iid, "prompt_components", "missing demo_evidence"
        ))
    if "demo_headers" not in comp:
        errors.append(ValidationError(
            iid, "prompt_components", "missing demo_headers"
        ))

    if not pv.prompt_hash:
        errors.append(ValidationError(iid, "prompt_hash", "missing prompt_hash"))

    return errors


def validate_icl_v2_pairing(views: List[PromptView]) -> List[ValidationError]:
    """Check that ICL v2 matched conditions share the same stem.

    For each item: target evidence, scenario, question, demo evidence,
    and demo answers must be identical across all 5 conditions.
    Only the demo headers differ.
    """
    errors = []
    by_item: Dict[str, Dict[str, PromptView]] = {}
    for pv in views:
        if pv.suite == "icl_v2":
            by_item.setdefault(pv.item_id, {})[pv.condition] = pv

    for iid, conds in by_item.items():
        cond_keys = set(conds.keys())
        if cond_keys != _ICL_V2_CONDS:
            errors.append(ValidationError(
                iid, "pairing",
                f"expected {sorted(_ICL_V2_CONDS)} conditions, got {sorted(cond_keys)}",
            ))
            continue

        ctrl = conds["control"]
        ctrl_evidence = ctrl.prompt_components.get("evidence", "")
        ctrl_scenario = ctrl.prompt_components.get("scenario", "")
        ctrl_question = ctrl.prompt_components.get("question", "")
        ctrl_demo_answers = ctrl.prompt_components.get("demo_answers", [])
        ctrl_demo_evidence = ctrl.prompt_components.get("demo_evidence", [])

        for cname in sorted(_ICL_V2_CONDS - {"control"}):
            anch = conds[cname]
            if anch.prompt_components.get("evidence", "") != ctrl_evidence:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} evidence differs from control"
                ))
            if anch.prompt_components.get("scenario", "") != ctrl_scenario:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} scenario differs from control"
                ))
            if anch.prompt_components.get("question", "") != ctrl_question:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} question differs from control"
                ))
            if anch.prompt_components.get("demo_answers", []) != ctrl_demo_answers:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} demo_answers differ from control"
                ))
            if anch.prompt_components.get("demo_evidence", []) != ctrl_demo_evidence:
                errors.append(ValidationError(
                    iid, "pairing", f"{cname} demo_evidence differs from control"
                ))

        hashes = [conds[c].prompt_hash for c in sorted(conds)]
        if len(set(hashes)) != len(hashes):
            errors.append(ValidationError(
                iid, "pairing", "duplicate prompt hashes among conditions"
            ))

    return errors


def validate_external_v2_leakage(
    specs: List[ItemSpec], views: List[PromptView]
) -> List[ValidationError]:
    """Check that prompts do not leak theta or y_star values."""
    errors = []
    spec_map = {s.item_id: s for s in specs if _is_v2(s)}

    for pv in views:
        spec = spec_map.get(pv.item_id)
        if spec is None:
            continue
        text = pv.prompt_text

        theta_str = str(spec.theta)
        y_theta_str = str(spec.y_star_theta)
        y_ev_str = str(spec.y_star_evidence)

        for label, val_str in [
            ("theta", theta_str),
            ("y_star_theta", y_theta_str),
            ("y_star_evidence", y_ev_str),
        ]:
            if pv.condition == "control":
                forbidden_patterns = [
                    f"theta {val_str}",
                    f"y_star {val_str}",
                    f"gold answer {val_str}",
                    f"correct answer is {val_str}",
                ]
            else:
                forbidden_patterns = [
                    f"theta {val_str}",
                    f"y_star {val_str}",
                    f"gold answer {val_str}",
                    f"correct answer is {val_str}",
                ]
            for pat in forbidden_patterns:
                if pat.lower() in text.lower():
                    errors.append(ValidationError(
                        pv.item_id, "leakage",
                        f"prompt contains '{pat}' (possible {label} leak)",
                    ))

    return errors


def validate_external_v2_difficulty(
    specs: List[ItemSpec],
) -> List[ValidationError]:
    """Check structural differences between easy and hard items."""
    errors = []
    v2_specs = [s for s in specs if _is_v2(s)]
    if not v2_specs:
        return errors

    easy = [s for s in v2_specs if s.difficulty == "easy"]
    hard = [s for s in v2_specs if s.difficulty == "hard"]

    if not easy:
        errors.append(ValidationError("dataset", "difficulty", "no easy items found"))
    if not hard:
        errors.append(ValidationError("dataset", "difficulty", "no hard items found"))

    for s in easy:
        missing = sum(1 for e in s.evidence_structured if e.get("missing"))
        if missing > 0:
            errors.append(ValidationError(
                s.item_id, "difficulty",
                f"easy item has {missing} missing evidence slots",
            ))

    for s in hard:
        missing = sum(1 for e in s.evidence_structured if e.get("missing"))
        if missing == 0:
            errors.append(ValidationError(
                s.item_id, "difficulty",
                "hard item has no missing evidence slots",
            ))

    return errors


def validate_all(
    specs: List[ItemSpec],
    views: List[PromptView],
) -> List[ValidationError]:
    """Run all deterministic validators. Returns list of errors (empty = OK)."""
    errors = []
    for spec in specs:
        errors.extend(validate_itemspec(spec))
    for pv in views:
        errors.extend(validate_promptview(pv))
    errors.extend(validate_external_pairing(views))

    v2_specs = [s for s in specs if _is_v2(s)]
    if v2_specs:
        errors.extend(validate_external_v2_leakage(specs, views))
        errors.extend(validate_external_v2_difficulty(specs))

    rag_v2_views = [pv for pv in views if getattr(pv, "suite", None) == "rag_v2"]
    if rag_v2_views:
        errors.extend(validate_rag_v2_pairing(rag_v2_views))

    tool_v2_views = [pv for pv in views if getattr(pv, "suite", None) == "tool_v2"]
    if tool_v2_views:
        errors.extend(validate_tool_v2_pairing(tool_v2_views))

    icl_v2_views = [pv for pv in views if getattr(pv, "suite", None) == "icl_v2"]
    if icl_v2_views:
        errors.extend(validate_icl_v2_pairing(icl_v2_views))

    return errors


# ── Optional LLM spot-check ──────────────────────────────────────────

def llm_spot_check(
    specs: List[ItemSpec],
    views: List[PromptView],
    sample_fraction: float = 0.05,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run LLM-based spot-check validation on a sample.

    Checks:
      (i)  "no leakage of gold/theta/y_star"
      (ii) "anchor is normatively irrelevant"

    Returns list of flagged items (empty = all OK).
    Requires OPENROUTER_API_KEY to be set.
    """
    rng = random.Random(seed)
    sample_size = max(1, int(len(views) * sample_fraction))
    sample = rng.sample(views, min(sample_size, len(views)))

    spec_map = {s.item_id: s for s in specs}
    flags = []

    try:
        from .llm_generate import generate
    except Exception:
        return [{"error": "Could not import llm_generate; skipping LLM spot-check"}]

    for pv in sample:
        spec = spec_map.get(pv.item_id)
        if not spec:
            continue

        check_prompt = (
            f"You are a dataset quality auditor. Given the following prompt "
            f"that will be shown to an LLM:\n\n---\n{pv.prompt_text}\n---\n\n"
            f"Check two things:\n"
            f"1. Does the prompt leak the gold answer ({spec.y_star}), "
            f"the latent theta ({spec.theta}), or y_star explicitly?\n"
            f"2. Is any anchor value presented as normatively relevant "
            f"(i.e., as the correct answer) rather than as incidental context?\n\n"
            f"Respond with a JSON object: "
            f'{{\"leakage\": true/false, \"anchor_relevant\": true/false, \"reason\": \"...\"}}'
        )

        try:
            text, _ = generate("validator", check_prompt, temperature=0.0)
            import json
            result = json.loads(text)
            if result.get("leakage") or result.get("anchor_relevant"):
                flags.append({
                    "item_id": pv.item_id,
                    "condition": pv.condition,
                    **result,
                })
        except Exception as exc:
            flags.append({
                "item_id": pv.item_id,
                "condition": pv.condition,
                "error": str(exc),
            })

    return flags
