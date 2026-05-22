"""Deterministic validators for AnchorBench datasets.

Checks run by ``validate_all``:
  - no duplicate item_ids or (item_id, condition) pairs
  - per-ItemSpec: theta, difficulty, anchors, evidence, gold answer
  - per-PromptView: answer format instruction, relevance, anchor, hash
  - paired-condition completeness: expected conditions present per item
  - paired-condition stem identity: evidence/scenario/question invariant
  - domain/difficulty balance within each suite
  - manifest count consistency (optional)
  - suite-specific metadata (rag dict, tool dict, icl demos, history warmups)
  - template diversity (evidence label families, scenario/question templates)
"""

from __future__ import annotations

import statistics
from typing import Any, Counter, Dict, List, Optional, Set

from .schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView

BASE_CONDITIONS = {
    "control", "irrelevant_low", "irrelevant_high",
    "plausible_low", "plausible_high",
}

EXTERNAL_CONDITIONS = BASE_CONDITIONS | {
    "placebo_low", "placebo_high",
    "authority_low", "authority_high",
}

HISTORY_CONDITIONS = BASE_CONDITIONS | {"control_twostage"}

ICL_CONDITIONS = BASE_CONDITIONS | {"neutral_low", "neutral_high"}

RAG_BASE_CONDITIONS = BASE_CONDITIONS
RAG_ABLATION_CONDITIONS = {
    "irrelevant_low_order_first", "irrelevant_low_order_last",
    "irrelevant_high_order_first", "irrelevant_high_order_last",
    "plausible_low_order_first", "plausible_low_order_last",
    "plausible_high_order_first", "plausible_high_order_last",
    "irrelevant_low_nodiscl", "irrelevant_high_nodiscl",
    "plausible_low_authority", "plausible_high_authority",
}
RAG_ALL_CONDITIONS = RAG_BASE_CONDITIONS | RAG_ABLATION_CONDITIONS

TOOL_CONDITIONS = BASE_CONDITIONS

ALL_VALID_CONDITIONS = (
    EXTERNAL_CONDITIONS | HISTORY_CONDITIONS | ICL_CONDITIONS
    | RAG_ALL_CONDITIONS | TOOL_CONDITIONS
)

ALL_VALID_RELEVANCE = {"none", "irrelevant", "plausible", "neutral", "placebo", "authority"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

SUITE_CONDITIONS: Dict[str, Set[str]] = {
    "external": EXTERNAL_CONDITIONS,
    "history": HISTORY_CONDITIONS,
    "icl": ICL_CONDITIONS,
    "icl_dist": BASE_CONDITIONS,
    "rag": RAG_ALL_CONDITIONS,
    "tool": TOOL_CONDITIONS,
    "tool_agentic": TOOL_CONDITIONS,
    "tool_read": TOOL_CONDITIONS,
}

_OFFSET_SUITES = {
    "external", "rag", "tool", "tool_agentic", "tool_read", "icl", "icl_dist",
}


def _compute_expected_gold(
    visible: List[int],
    scoring_function: str = "mean",
    scoring_weights: Optional[List[float]] = None,
) -> int:
    """Recompute expected gold answer to validate ItemSpec.y_star_evidence."""
    if not visible:
        return 0
    if scoring_function == "mean":
        return round(sum(visible) / len(visible))
    if scoring_function == "weighted_mean":
        if scoring_weights:
            w = scoring_weights[:len(visible)]
            total = sum(v * wi for v, wi in zip(visible, w))
            return round(total / sum(w))
        return round(sum(visible) / len(visible))
    if scoring_function == "median":
        return round(statistics.median(visible))
    return round(sum(visible) / len(visible))


# ── ItemSpec checks ──────────────────────────────────────────────────

def _check_itemspec(spec: ItemSpec) -> List[str]:
    errs: List[str] = []
    iid = spec.item_id

    if not (30 <= spec.theta <= 70):
        errs.append(f"{iid}: theta={spec.theta} not in [30,70]")
    if spec.difficulty not in VALID_DIFFICULTIES:
        errs.append(f"{iid}: difficulty={spec.difficulty!r} not in {VALID_DIFFICULTIES}")
    if not (0 <= spec.y_star_evidence <= 100):
        errs.append(f"{iid}: y_star_evidence={spec.y_star_evidence} out of [0,100]")
    if spec.y_star != spec.y_star_evidence:
        errs.append(f"{iid}: y_star={spec.y_star} != y_star_evidence={spec.y_star_evidence}")

    if not (0 <= spec.anchors.get("low", -1) <= 100):
        errs.append(f"{iid}: anchors.low out of [0,100]")
    if not (0 <= spec.anchors.get("high", -1) <= 100):
        errs.append(f"{iid}: anchors.high out of [0,100]")

    if spec.suite in _OFFSET_SUITES:
        offset = spec.anchors.get("offset", 0)
        if offset not in (15, 25, 40):
            errs.append(f"{iid}: offset={offset} not in {{15, 25, 40}}")

    if not spec.evidence_structured:
        errs.append(f"{iid}: empty evidence")
    else:
        missing = sum(1 for e in spec.evidence_structured if e.get("missing"))
        expected_missing = {"easy": 0, "medium": 1, "hard": 2}
        exp = expected_missing.get(spec.difficulty)
        if exp is not None and missing != exp:
            errs.append(f"{iid}: {spec.difficulty} item has {missing} missing (expected {exp})")

        for e in spec.evidence_structured:
            if not e.get("missing"):
                v = e.get("value")
                if v is None or not (0 <= v <= 100):
                    errs.append(f"{iid}: visible evidence value {v} out of [0,100]")

        visible = [e["value"] for e in spec.evidence_structured if not e.get("missing")]
        if visible:
            scoring_fn = getattr(spec, "scoring_function", "mean")
            scoring_wts = getattr(spec, "scoring_weights", None)
            expected = _compute_expected_gold(visible, scoring_fn, scoring_wts)
            if spec.y_star_evidence != expected:
                errs.append(
                    f"{iid}: y_star_evidence={spec.y_star_evidence} != "
                    f"expected({scoring_fn})={expected}"
                )

    return errs


def _check_suite_extras(spec: ItemSpec) -> List[str]:
    errs: List[str] = []
    iid = spec.item_id

    if spec.suite == "external":
        if spec.y_star_theta != spec.theta:
            errs.append(f"{iid}: y_star_theta={spec.y_star_theta} != theta={spec.theta}")

    elif spec.suite == "rag":
        if not spec.rag:
            errs.append(f"{iid}: missing rag dict")
        elif spec.rag.get("corpus_size") != 3:
            errs.append(f"{iid}: rag.corpus_size={spec.rag.get('corpus_size')}, expected 3")

    elif spec.suite in ("tool", "tool_agentic", "tool_read"):
        if not spec.tool:
            errs.append(f"{iid}: missing tool dict")
        else:
            expected = {"get_evidence_summary", "check_external_reference"}
            actual = set(spec.tool.get("available_tools", []))
            if actual != expected:
                errs.append(f"{iid}: tool.available_tools mismatch")
            if spec.tool.get("tool_schema_version") != "v2":
                errs.append(f"{iid}: tool_schema_version should be 'v2'")

    elif spec.suite == "icl":
        demos = spec.tags.get("icl_demos")
        if not demos:
            errs.append(f"{iid}: missing icl_demos in tags")
        elif not isinstance(demos, list) or len(demos) != 3:
            errs.append(f"{iid}: icl_demos should be list of 3")
        else:
            for di, d in enumerate(demos):
                if "evidence" not in d or "answer" not in d:
                    errs.append(f"{iid}: icl_demos[{di}] missing evidence or answer")
                elif not (0 <= d.get("answer", -1) <= 100):
                    errs.append(f"{iid}: icl_demos[{di}].answer out of [0,100]")

    elif spec.suite == "icl_dist":
        for key in (
            "icl_dist_demos_control",
            "icl_dist_demos_low",
            "icl_dist_demos_high",
        ):
            demos = spec.tags.get(key)
            if not demos:
                errs.append(f"{iid}: missing tags.{key}")
            elif not isinstance(demos, list) or len(demos) != 3:
                errs.append(f"{iid}: {key} should be list of 3")
            else:
                for di, d in enumerate(demos):
                    if "evidence" not in d or "answer" not in d:
                        errs.append(f"{iid}: {key}[{di}] missing evidence or answer")
                    elif not (0 <= d.get("answer", -1) <= 100):
                        errs.append(f"{iid}: {key}[{di}].answer out of [0,100]")

    elif spec.suite == "history":
        if not spec.history:
            errs.append(f"{iid}: missing history dict")
        else:
            h = spec.history
            for key in ("subset_indices_low", "subset_indices_high"):
                idx = h.get(key, [])
                if not isinstance(idx, list) or len(idx) != 2:
                    errs.append(f"{iid}: history.{key} should be list of 2")
                elif any(not (0 <= i < 5) for i in idx):
                    errs.append(f"{iid}: history.{key} indices out of [0,4]")
            for key in ("warmup_low", "warmup_high"):
                w = h.get(key)
                if not w or "evidence_structured" not in w or "y_star_evidence" not in w:
                    errs.append(f"{iid}: history.{key} missing or incomplete")

    return errs


# ── PromptView checks ────────────────────────────────────────────────

def _infer_expected_relevance(condition: str) -> Optional[str]:
    if condition in ("control", "control_twostage"):
        return "none"
    if "authority" in condition:
        return "authority"
    if "placebo" in condition:
        return "placebo"
    if "neutral" in condition:
        return "neutral"
    if "irrelevant" in condition:
        return "irrelevant"
    if "plausible" in condition:
        return "plausible"
    return None


def _check_promptview(pv: PromptView) -> List[str]:
    errs: List[str] = []
    tag = f"{pv.item_id}/{pv.condition}"

    if ANSWER_FORMAT_INSTRUCTION not in pv.prompt_text:
        errs.append(f"{tag}: missing answer format instruction")
    if not pv.prompt_hash:
        errs.append(f"{tag}: missing prompt_hash")

    if pv.anchor_relevance not in ALL_VALID_RELEVANCE:
        errs.append(f"{tag}: anchor_relevance={pv.anchor_relevance!r} invalid")

    if pv.condition in ("control", "control_twostage"):
        if pv.anchor_relevance != "none":
            errs.append(f"{tag}: control anchor_relevance should be 'none'")
        if pv.anchor_value is not None:
            errs.append(f"{tag}: control should have anchor_value=None")
    elif pv.condition in ALL_VALID_CONDITIONS:
        expected_rel = _infer_expected_relevance(pv.condition)
        if expected_rel and pv.anchor_relevance != expected_rel:
            errs.append(f"{tag}: anchor_relevance={pv.anchor_relevance!r}, expected {expected_rel!r}")
        if pv.anchor_value is None and pv.suite != "history":
            errs.append(f"{tag}: anchor_value is None")

    return errs


# ── Pairing checks ───────────────────────────────────────────────────

_INVARIANT_KEYS = ("evidence", "scenario", "question")

_SUITE_EXTRA_INVARIANT_KEYS: Dict[str, tuple] = {
    "icl": ("demo_answers", "demo_evidence"),
    "icl_dist": (),
    "tool": ("evidence_summary",),
    "tool_agentic": ("evidence_summary",),
    "tool_read": ("evidence_summary",),
}


def _check_pairing(views: List[PromptView]) -> List[str]:
    errs: List[str] = []
    by_item: Dict[str, Dict[str, PromptView]] = {}
    for pv in views:
        by_item.setdefault(pv.item_id, {})[pv.condition] = pv

    for iid, conds in by_item.items():
        ctrl = conds.get("control")
        if ctrl is None:
            errs.append(f"{iid}: missing control condition")
            continue

        suite = ctrl.suite
        expected = SUITE_CONDITIONS.get(suite, BASE_CONDITIONS)
        actual = set(conds.keys())
        if not expected.issuperset(actual):
            unexpected = actual - expected
            errs.append(f"{iid}: unexpected conditions {sorted(unexpected)}")
        if not actual.issuperset(BASE_CONDITIONS):
            missing_conds = BASE_CONDITIONS - actual
            errs.append(f"{iid}: missing base conditions {sorted(missing_conds)}")
            continue

        keys_to_check = list(_INVARIANT_KEYS) + list(
            _SUITE_EXTRA_INVARIANT_KEYS.get(suite, ())
        )

        for cname in sorted(actual - {"control", "control_twostage"}):
            for key in keys_to_check:
                if conds[cname].prompt_components.get(key, "") != ctrl.prompt_components.get(key, ""):
                    errs.append(f"{iid}: {cname} '{key}' differs from control")

        if suite == "icl_dist":
            for a, b in (("plausible_low", "irrelevant_low"), ("plausible_high", "irrelevant_high")):
                pa, pb = conds.get(a), conds.get(b)
                if pa and pb:
                    if pa.prompt_components.get("demos") != pb.prompt_components.get("demos"):
                        errs.append(f"{iid}: icl_dist {a} vs {b}: demos must match")
                    if pa.prompt_components.get("framing_intro") == pb.prompt_components.get("framing_intro"):
                        errs.append(f"{iid}: icl_dist {a} vs {b}: framing_intro must differ")

        hashes = [conds[c].prompt_hash for c in sorted(conds)]
        if len(set(hashes)) != len(hashes):
            errs.append(f"{iid}: duplicate prompt hashes among conditions")

    return errs


# ── Cross-dataset checks ─────────────────────────────────────────────

def _check_no_duplicates(specs: List[ItemSpec], views: List[PromptView]) -> List[str]:
    errs: List[str] = []
    seen: Dict[str, int] = {}
    for s in specs:
        seen[s.item_id] = seen.get(s.item_id, 0) + 1
    for iid, n in seen.items():
        if n > 1:
            errs.append(f"{iid}: duplicate ItemSpec ({n} copies)")

    seen = {}
    for pv in views:
        key = f"{pv.item_id}|{pv.condition}"
        seen[key] = seen.get(key, 0) + 1
    for key, n in seen.items():
        if n > 1:
            errs.append(f"{key}: duplicate PromptView ({n} copies)")

    return errs


def _check_balance(specs: List[ItemSpec]) -> List[str]:
    errs: List[str] = []
    by_suite: Dict[str, List[ItemSpec]] = {}
    for s in specs:
        by_suite.setdefault(s.suite, []).append(s)

    for suite, ss in by_suite.items():
        domains: Dict[str, int] = {}
        diffs: Dict[str, int] = {}
        for s in ss:
            domains[s.domain] = domains.get(s.domain, 0) + 1
            diffs[s.difficulty] = diffs.get(s.difficulty, 0) + 1

        if len(domains) > 1 and max(domains.values()) > 2 * min(domains.values()):
            errs.append(f"suite:{suite}: domain imbalance {dict(sorted(domains.items()))}")

        difficulty_groups = [k for k in ("easy", "hard") if k in diffs]
        if len(difficulty_groups) == 2 and diffs["easy"] != diffs["hard"]:
            errs.append(f"suite:{suite}: difficulty imbalance easy={diffs['easy']}, hard={diffs['hard']}")

    return errs


def _check_template_diversity(specs: List[ItemSpec]) -> List[str]:
    """Warn if template indices show no variation (all items use same template)."""
    errs: List[str] = []
    by_suite: Dict[str, List[ItemSpec]] = {}
    for s in specs:
        by_suite.setdefault(s.suite, []).append(s)

    for suite, ss in by_suite.items():
        if len(ss) < 4:
            continue
        scenario_indices = Counter(s.scenario_template_idx for s in ss)
        question_indices = Counter(s.question_template_idx for s in ss)
        label_indices = Counter(s.evidence_label_family_idx for s in ss)
        phrasing_indices = Counter(s.anchor_phrasing_idx for s in ss)

        if len(scenario_indices) == 1:
            errs.append(f"suite:{suite}: all items use scenario_template_idx={list(scenario_indices.keys())[0]}")
        if len(question_indices) == 1:
            errs.append(f"suite:{suite}: all items use question_template_idx={list(question_indices.keys())[0]}")
        if len(label_indices) == 1 and len(ss) >= 6:
            errs.append(f"suite:{suite}: all items use evidence_label_family_idx={list(label_indices.keys())[0]}")
        if len(phrasing_indices) == 1 and len(ss) >= 8:
            errs.append(f"suite:{suite}: all items use anchor_phrasing_idx={list(phrasing_indices.keys())[0]}")

    return errs


def _check_manifest(
    specs: List[ItemSpec], views: List[PromptView], manifest: Dict[str, Any],
) -> List[str]:
    errs: List[str] = []
    counts = manifest.get("counts", {})

    n = counts.get("itemspecs")
    if n is not None and n != len(specs):
        errs.append(f"manifest: itemspecs count={n}, actual={len(specs)}")
    n = counts.get("promptviews")
    if n is not None and n != len(views):
        errs.append(f"manifest: promptviews count={n}, actual={len(views)}")

    m_domains = set(counts.get("domains", []))
    a_domains = set(s.domain for s in specs)
    if m_domains and m_domains != a_domains:
        errs.append(f"manifest: domains mismatch")

    return errs


# ── Public entry point ────────────────────────────────────────────────

def validate_all(
    specs: List[ItemSpec],
    views: List[PromptView],
    manifest: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Run all deterministic validators.  Empty list = all passed."""
    errs: List[str] = []

    errs.extend(_check_no_duplicates(specs, views))

    for spec in specs:
        errs.extend(_check_itemspec(spec))
        errs.extend(_check_suite_extras(spec))
    for pv in views:
        errs.extend(_check_promptview(pv))

    errs.extend(_check_pairing(views))
    errs.extend(_check_balance(specs))
    errs.extend(_check_template_diversity(specs))

    if manifest:
        errs.extend(_check_manifest(specs, views, manifest))

    return errs
