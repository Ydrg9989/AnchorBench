"""Generate canonical ItemSpec records for AnchorBench v1 and v2.

v1 gold rule:
  theta ∈ [30, 70]   (uniform, seed-controlled)
  anchor_low  = theta - 30
  anchor_high = theta + 30
  anchor_gap  = 60
  y_star      = round(theta)   (deterministic)

v2 gold rule (External suite):
  theta ∈ [30, 70]
  anchor offsets stratified: {15, 25, 40}
  y_star_theta    = theta
  y_star_evidence = round(mean(visible evidence values))
  y_star          = y_star_evidence  (main accuracy target)

Evidence: 5 ratings drawn around theta with controlled noise.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .domains import DOMAINS, DOMAIN_IDS, DomainConfig
from .schema import ItemSpec

THETA_MIN, THETA_MAX = 30, 70
ANCHOR_OFFSET = 30
EVIDENCE_COUNT = 5
EVIDENCE_SIGMA = 8

SUITES = ["external", "icl", "history", "rag", "tool"]

_SUITE_PREFIX = {
    "external": "EXT",
    "icl": "ICL",
    "history": "HIST",
    "rag": "RAG",
    "tool": "TOOL",
    "tool_v2": "TOOLv2",
    "icl_v2": "ICLv2",
}


def _make_item_id(suite: str, domain: str, idx: int) -> str:
    prefix = _SUITE_PREFIX[suite]
    return f"{prefix}-{domain}-{idx:03d}"


def _generate_evidence(
    rng: random.Random,
    theta: int,
    labels: List[str],
    sigma: float = EVIDENCE_SIGMA,
) -> List[dict]:
    """Generate 5 evidence ratings centered on theta."""
    evidence = []
    for i, label in enumerate(labels):
        raw = rng.gauss(theta, sigma)
        value = max(0, min(100, round(raw)))
        evidence.append({
            "type": "rating",
            "value": value,
            "label": label,
            "index": i,
        })
    return evidence


def _evidence_to_theta_mapping(evidence: List[dict], theta: int) -> dict:
    values = [e["value"] for e in evidence]
    return {
        "method": "round_latent_theta",
        "evidence_mean": round(sum(values) / len(values), 2),
        "latent_theta": theta,
        "note": "y_star = round(theta); evidence provides supporting context",
    }


def generate_itemspecs(
    suites: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    n_per_cell: int = 20,
    seed: int = 42,
    generator_version: str = "",
    split: str = "core",
) -> List[ItemSpec]:
    """Generate ItemSpec records.

    Args:
        suites: which suites (default: all 5)
        domains: which domains (default: all 6)
        n_per_cell: items per (suite, domain) pair
        seed: master seed
        generator_version: version hash for reproducibility
        split: tag for the split ("core", "stress", "dev")

    Returns:
        List of ItemSpec records.
    """
    suites = suites or SUITES
    domains = domains or DOMAIN_IDS
    rng = random.Random(seed)
    specs = []

    for suite in suites:
        for domain_id in domains:
            dcfg = DOMAINS[domain_id]
            for i in range(n_per_cell):
                theta = rng.randint(THETA_MIN, THETA_MAX)
                y_star = round(theta)
                anchor_low = theta - ANCHOR_OFFSET
                anchor_high = theta + ANCHOR_OFFSET

                evidence = _generate_evidence(rng, theta, dcfg.evidence_labels)
                y_star_components = {
                    "evidence_structured": [e.copy() for e in evidence],
                    "evidence_to_theta": _evidence_to_theta_mapping(evidence, theta),
                    "theta": theta,
                    "y_star": y_star,
                    "rounding_applied": False,
                    "clamping_applied": False,
                }

                template_idx = i % len(dcfg.scenario_templates)
                template_family = f"{domain_id}_{template_idx:02d}"

                spec = ItemSpec(
                    item_id=_make_item_id(suite, domain_id, i + 1),
                    suite=suite,
                    domain=domain_id,
                    template_family=template_family,
                    theta=theta,
                    y_star=y_star,
                    y_star_components=y_star_components,
                    anchors={
                        "low": anchor_low,
                        "high": anchor_high,
                        "gap": ANCHOR_OFFSET * 2,
                        "anchor_type": "numeric",
                        "relevance": "normatively_irrelevant",
                    },
                    evidence_structured=evidence,
                    tags={"split": split, "difficulty": "standard"},
                    seed=seed,
                    generator_version=generator_version,
                )
                specs.append(spec)

    return specs


# ── RAG suite v2 generation ──────────────────────────────────────────

def generate_rag_v2_itemspecs(
    domains: Optional[List[str]] = None,
    n_per_cell: int = 5,
    seed: int = 42,
    generator_version: str = "",
    split: str = "pilot",
    difficulties: Optional[List[str]] = None,
    anchor_offsets: Optional[List[int]] = None,
) -> List[ItemSpec]:
    """Generate RAG v2 ItemSpec records (same backbone as External v2).

    Iterates over (domain, difficulty, anchor_offset, item_idx).
    Each item receives a frozen 3-document mini-corpus at render time.

    Args:
        domains: which domains (default: all 6)
        n_per_cell: items per (domain, difficulty, offset) triple
        seed: master seed (offset by +2000 from external seed space)
        generator_version: version hash
        split: tag ("pilot", "core", "dev")
        difficulties: default ["easy", "hard"]
        anchor_offsets: stratified offset levels (default: [15, 25, 40])

    Returns:
        List of ItemSpec with suite="rag_v2" and rag dict populated.
    """
    domains = domains or DOMAIN_IDS
    difficulties = difficulties or ["easy", "hard"]
    anchor_offsets = anchor_offsets or ANCHOR_OFFSETS_V2
    rng = random.Random(seed + 2000)
    specs: List[ItemSpec] = []
    global_idx = 0

    for domain_id in domains:
        dcfg = DOMAINS[domain_id]
        for difficulty in difficulties:
            for offset in anchor_offsets:
                for i in range(n_per_cell):
                    global_idx += 1
                    theta = rng.randint(THETA_MIN, THETA_MAX)

                    anchor_low = max(0, theta - offset)
                    anchor_high = min(100, theta + offset)
                    anchor_gap = anchor_high - anchor_low

                    evidence = _generate_evidence_v2(
                        rng, theta, dcfg.evidence_labels, difficulty
                    )
                    y_star_theta = theta
                    y_star_evidence = _compute_y_star_evidence(evidence)

                    visible_values = [
                        e["value"] for e in evidence if not e.get("missing")
                    ]
                    missing_idxs = [
                        e["index"] for e in evidence if e.get("missing")
                    ]

                    y_star_components = {
                        "evidence_values_raw": [e["value_raw"] for e in evidence],
                        "evidence_values_visible": visible_values,
                        "evidence_mean": round(
                            sum(visible_values) / len(visible_values), 2
                        ),
                        "evidence_std": round(
                            statistics.pstdev(visible_values), 2
                        ) if len(visible_values) > 1 else 0.0,
                        "evidence_count": len(visible_values),
                        "missing_indices": missing_idxs,
                        "y_star_theta": y_star_theta,
                        "y_star_evidence": y_star_evidence,
                        "aggregation_rule": "round(mean(visible_values))",
                        "theta": theta,
                    }

                    template_idx = global_idx % len(dcfg.scenario_templates)
                    template_family = f"{domain_id}_{template_idx:02d}"

                    item_id = f"RAGv2-{domain_id}-{difficulty[0]}-off{offset}-{i+1:03d}"

                    spec = ItemSpec(
                        item_id=item_id,
                        suite="rag_v2",
                        domain=domain_id,
                        template_family=template_family,
                        theta=theta,
                        y_star=y_star_evidence,
                        y_star_components=y_star_components,
                        y_star_theta=y_star_theta,
                        y_star_evidence=y_star_evidence,
                        difficulty=difficulty,
                        anchors={
                            "low": anchor_low,
                            "high": anchor_high,
                            "gap": anchor_gap,
                            "offset": offset,
                            "anchor_type": "numeric",
                            "relevance": "stratified",
                        },
                        evidence_structured=evidence,
                        tags={"split": split, "difficulty": difficulty},
                        seed=seed,
                        generator_version=generator_version,
                        render_version="2.0.0",
                        rag={
                            "corpus_id": "anchorbench_corpus_v2",
                            "corpus_size": 3,
                            "retrieval_method": "deterministic_fixed_order",
                            "doc_roles": ["core", "anchor_slot", "filler"],
                        },
                    )
                    specs.append(spec)

    return specs


# ── External suite v2 generation ─────────────────────────────────────

ANCHOR_OFFSETS_V2 = [15, 25, 40]
EASY_SIGMA = 8
HARD_SIGMA = 15
HARD_MISSING_COUNT = 2


def _generate_evidence_v2(
    rng: random.Random,
    theta: int,
    labels: List[str],
    difficulty: str,
) -> List[dict]:
    """Generate evidence for v2 with difficulty-dependent structure.

    Easy: 5 visible concordant ratings, sigma=8.
    Hard: 5 slots (2 missing), sigma=15, 1 visible value flipped for conflict.
    """
    sigma = EASY_SIGMA if difficulty == "easy" else HARD_SIGMA
    raw_values = []
    for i, label in enumerate(labels):
        raw = rng.gauss(theta, sigma)
        value = max(0, min(100, round(raw)))
        raw_values.append(value)

    evidence = []
    missing_indices: List[int] = []

    if difficulty == "hard":
        indices = list(range(len(labels)))
        missing_indices = sorted(rng.sample(indices, HARD_MISSING_COUNT))
        visible_indices = [i for i in indices if i not in missing_indices]
        flip_idx = rng.choice(visible_indices)
        raw_values[flip_idx] = max(0, min(100, 100 - raw_values[flip_idx]))

    for i, label in enumerate(labels):
        is_missing = i in missing_indices
        evidence.append({
            "type": "rating",
            "value": None if is_missing else raw_values[i],
            "value_raw": raw_values[i],
            "label": label,
            "index": i,
            "missing": is_missing,
        })

    return evidence


def _compute_y_star_evidence(evidence: List[dict]) -> int:
    """Mean of visible (non-missing) evidence values, rounded."""
    visible = [e["value"] for e in evidence if not e.get("missing")]
    if not visible:
        return 0
    return round(sum(visible) / len(visible))


def generate_external_v2_itemspecs(
    domains: Optional[List[str]] = None,
    n_per_cell: int = 5,
    seed: int = 42,
    generator_version: str = "",
    split: str = "pilot",
    difficulties: Optional[List[str]] = None,
    anchor_offsets: Optional[List[int]] = None,
) -> List[ItemSpec]:
    """Generate External-suite v2 ItemSpec records.

    Iterates over (domain, difficulty, anchor_offset, item_idx) to produce
    items with stratified anchor distances and easy/hard evidence.

    Args:
        domains: which domains (default: first 3 for pilot)
        n_per_cell: items per (domain, difficulty, offset) triple
        seed: master seed
        generator_version: version hash
        split: tag ("pilot", "core", "dev")
        difficulties: list of difficulties (default: ["easy", "hard"])
        anchor_offsets: stratified offset levels (default: [15, 25, 40])

    Returns:
        List of ItemSpec records with v2 fields populated.
    """
    domains = domains or DOMAIN_IDS[:3]
    difficulties = difficulties or ["easy", "hard"]
    anchor_offsets = anchor_offsets or ANCHOR_OFFSETS_V2
    rng = random.Random(seed)
    specs: List[ItemSpec] = []
    global_idx = 0

    for domain_id in domains:
        dcfg = DOMAINS[domain_id]
        for difficulty in difficulties:
            for offset in anchor_offsets:
                for i in range(n_per_cell):
                    global_idx += 1
                    theta = rng.randint(THETA_MIN, THETA_MAX)

                    anchor_low = max(0, theta - offset)
                    anchor_high = min(100, theta + offset)
                    anchor_gap = anchor_high - anchor_low

                    evidence = _generate_evidence_v2(
                        rng, theta, dcfg.evidence_labels, difficulty
                    )
                    y_star_theta = theta
                    y_star_evidence = _compute_y_star_evidence(evidence)

                    visible_values = [
                        e["value"] for e in evidence if not e.get("missing")
                    ]
                    missing_idxs = [
                        e["index"] for e in evidence if e.get("missing")
                    ]

                    y_star_components = {
                        "evidence_values_raw": [e["value_raw"] for e in evidence],
                        "evidence_values_visible": visible_values,
                        "evidence_mean": round(
                            sum(visible_values) / len(visible_values), 2
                        ),
                        "evidence_std": round(
                            statistics.pstdev(visible_values), 2
                        ) if len(visible_values) > 1 else 0.0,
                        "evidence_count": len(visible_values),
                        "missing_indices": missing_idxs,
                        "y_star_theta": y_star_theta,
                        "y_star_evidence": y_star_evidence,
                        "aggregation_rule": "round(mean(visible_values))",
                        "theta": theta,
                    }

                    template_idx = global_idx % len(dcfg.scenario_templates)
                    template_family = f"{domain_id}_{template_idx:02d}"

                    item_id = f"EXTv2-{domain_id}-{difficulty[0]}-off{offset}-{i+1:03d}"

                    spec = ItemSpec(
                        item_id=item_id,
                        suite="external",
                        domain=domain_id,
                        template_family=template_family,
                        theta=theta,
                        y_star=y_star_evidence,
                        y_star_components=y_star_components,
                        y_star_theta=y_star_theta,
                        y_star_evidence=y_star_evidence,
                        difficulty=difficulty,
                        anchors={
                            "low": anchor_low,
                            "high": anchor_high,
                            "gap": anchor_gap,
                            "offset": offset,
                            "anchor_type": "numeric",
                            "relevance": "stratified",
                        },
                        evidence_structured=evidence,
                        tags={"split": split, "difficulty": difficulty},
                        seed=seed,
                        generator_version=generator_version,
                        render_version="2.0.0",
                    )
                    specs.append(spec)

    return specs


# ── History suite v2 generation ───────────────────────────────────────

# Delta for subset search: target mean = y_star_evidence ± delta (easy items)
HISTORY_V2_SUBSET_DELTA = 15
# Warmup theta ranges (same domain, different case)
WARMUP_LOW_THETA_MIN, WARMUP_LOW_THETA_MAX = 15, 30
WARMUP_HIGH_THETA_MIN, WARMUP_HIGH_THETA_MAX = 70, 85


def _enumerate_2_of_5_subsets() -> List[Tuple[int, int]]:
    """All 2-of-5 index pairs (for partial evidence Stage 1)."""
    return list(itertools.combinations(range(5), 2))


def _subset_mean(evidence: List[dict], indices: Tuple[int, ...]) -> float:
    """Mean of evidence values at given indices (visible only)."""
    values = []
    for i in indices:
        if i < len(evidence) and not evidence[i].get("missing"):
            v = evidence[i].get("value")
            if v is not None:
                values.append(v)
    return sum(values) / len(values) if values else 0.0


def _choose_plausible_subsets(
    evidence: List[dict],
    y_star_evidence: int,
    delta: int = HISTORY_V2_SUBSET_DELTA,
) -> Dict[str, Any]:
    """Choose low- and high-pressure 2-of-5 subsets by mean closest to y_star ± delta.

    Returns dict with subset_indices_low/high, subset_mean_low/high, realized_gap_low/high.
    """
    visible = [e["value"] for e in evidence if not e.get("missing")]
    if len(visible) < 2:
        return {
            "subset_indices_low": [0, 1],
            "subset_indices_high": [0, 1],
            "subset_mean_low": 0.0,
            "subset_mean_high": 0.0,
            "realized_gap_low": 0.0,
            "realized_gap_high": 0.0,
        }
    subsets = _enumerate_2_of_5_subsets()
    target_low = y_star_evidence - delta
    target_high = y_star_evidence + delta

    best_low = min(subsets, key=lambda s: abs(_subset_mean(evidence, s) - target_low))
    best_high = min(subsets, key=lambda s: abs(_subset_mean(evidence, s) - target_high))
    mean_low = _subset_mean(evidence, best_low)
    mean_high = _subset_mean(evidence, best_high)

    return {
        "subset_indices_low": list(best_low),
        "subset_indices_high": list(best_high),
        "subset_mean_low": round(mean_low, 2),
        "subset_mean_high": round(mean_high, 2),
        "realized_gap_low": round(abs(mean_low - target_low), 2),
        "realized_gap_high": round(abs(mean_high - target_high), 2),
    }


def _build_warmup_case(
    rng: random.Random,
    domain_id: str,
    theta_min: int,
    theta_max: int,
    template_idx: int,
) -> Dict[str, Any]:
    """Build a warmup case (same domain, different case) for irrelevant conditions."""
    dcfg = DOMAINS[domain_id]
    theta = rng.randint(theta_min, theta_max)
    evidence = _generate_evidence(rng, theta, dcfg.evidence_labels)
    y_star_evidence = _compute_y_star_evidence(evidence)
    return {
        "theta": theta,
        "evidence_structured": evidence,
        "y_star_evidence": y_star_evidence,
        "scenario_template_idx": template_idx,
    }


def generate_history_v2_itemspecs(
    domains: Optional[List[str]] = None,
    n_per_cell: int = 5,
    seed: int = 42,
    generator_version: str = "",
    split: str = "pilot",
    difficulties: Optional[List[str]] = None,
    subset_delta: int = HISTORY_V2_SUBSET_DELTA,
) -> List[ItemSpec]:
    """Generate History v2 ItemSpec records (same six domains as External v2).

    Each item has: same theta/evidence/y_star_evidence backbone as External v2;
    plausible low/high subsets via subset search; same-domain warmup cases for
    irrelevant_low and irrelevant_high.

    Args:
        domains: which domains (default: all 6 DOMAIN_IDS)
        n_per_cell: items per (domain, difficulty) pair
        seed: master seed
        generator_version: version hash
        split: tag ("pilot", "core", "dev")
        difficulties: default ["easy", "hard"]
        subset_delta: delta for subset search (default 15)

    Returns:
        List of ItemSpec with suite="history_v2" and history dict populated.
    """
    domains = domains or DOMAIN_IDS
    difficulties = difficulties or ["easy", "hard"]
    rng = random.Random(seed)
    specs: List[ItemSpec] = []
    global_idx = 0

    for domain_id in domains:
        dcfg = DOMAINS[domain_id]
        for difficulty in difficulties:
            for i in range(n_per_cell):
                global_idx += 1
                theta = rng.randint(THETA_MIN, THETA_MAX)
                evidence = _generate_evidence_v2(
                    rng, theta, dcfg.evidence_labels, difficulty
                )
                y_star_theta = theta
                y_star_evidence = _compute_y_star_evidence(evidence)

                visible_values = [
                    e["value"] for e in evidence if not e.get("missing")
                ]
                missing_idxs = [
                    e["index"] for e in evidence if e.get("missing")
                ]
                y_star_components = {
                    "evidence_values_raw": [e["value_raw"] for e in evidence],
                    "evidence_values_visible": visible_values,
                    "evidence_mean": round(
                        sum(visible_values) / len(visible_values), 2
                    ),
                    "evidence_std": round(
                        statistics.pstdev(visible_values), 2
                    ) if len(visible_values) > 1 else 0.0,
                    "evidence_count": len(visible_values),
                    "missing_indices": missing_idxs,
                    "y_star_theta": y_star_theta,
                    "y_star_evidence": y_star_evidence,
                    "aggregation_rule": "round(mean(visible_values))",
                    "theta": theta,
                }

                # Subset search for plausible low/high (only use visible evidence)
                plausible_data = _choose_plausible_subsets(
                    evidence, y_star_evidence, delta=subset_delta
                )

                # Same-domain warmup cases (different case identity)
                warmup_template_idx = (global_idx + 1) % len(dcfg.scenario_templates)
                warmup_low = _build_warmup_case(
                    rng, domain_id,
                    WARMUP_LOW_THETA_MIN, WARMUP_LOW_THETA_MAX,
                    warmup_template_idx,
                )
                warmup_high = _build_warmup_case(
                    rng, domain_id,
                    WARMUP_HIGH_THETA_MIN, WARMUP_HIGH_THETA_MAX,
                    (warmup_template_idx + 1) % len(dcfg.scenario_templates),
                )

                template_idx = global_idx % len(dcfg.scenario_templates)
                template_family = f"{domain_id}_{template_idx:02d}"
                item_id = f"HISTv2-{domain_id}-{difficulty[0]}-{i+1:03d}"

                spec = ItemSpec(
                    item_id=item_id,
                    suite="history_v2",
                    domain=domain_id,
                    template_family=template_family,
                    theta=theta,
                    y_star=y_star_evidence,
                    y_star_components=y_star_components,
                    y_star_theta=y_star_theta,
                    y_star_evidence=y_star_evidence,
                    difficulty=difficulty,
                    anchors={
                        "low": max(0, theta - 25),
                        "high": min(100, theta + 25),
                        "anchor_type": "numeric",
                        "relevance": "self_generated",
                    },
                    evidence_structured=evidence,
                    tags={"split": split, "difficulty": difficulty},
                    seed=seed,
                    generator_version=generator_version,
                    render_version="2.0.0",
                    history={
                        **plausible_data,
                        "warmup_low": warmup_low,
                        "warmup_high": warmup_high,
                        "subset_delta": subset_delta,
                    },
                )
                specs.append(spec)

    return specs


# ── Tool suite v2 generation ─────────────────────────────────────────

def generate_tool_v2_itemspecs(
    domains: Optional[List[str]] = None,
    n_per_cell: int = 5,
    seed: int = 42,
    generator_version: str = "",
    split: str = "pilot",
    difficulties: Optional[List[str]] = None,
    anchor_offsets: Optional[List[int]] = None,
) -> List[ItemSpec]:
    """Generate Tool v2 ItemSpec records (same backbone as External/RAG v2).

    Iterates over (domain, difficulty, anchor_offset, item_idx).
    Each item uses deterministic tool outputs at render time.

    Args:
        domains: which domains (default: all 6)
        n_per_cell: items per (domain, difficulty, offset) triple
        seed: master seed (offset by +3000 from external seed space)
        generator_version: version hash
        split: tag ("pilot", "core", "dev")
        difficulties: default ["easy", "hard"]
        anchor_offsets: stratified offset levels (default: [15, 25, 40])

    Returns:
        List of ItemSpec with suite="tool_v2" and tool dict populated.
    """
    domains = domains or DOMAIN_IDS
    difficulties = difficulties or ["easy", "hard"]
    anchor_offsets = anchor_offsets or ANCHOR_OFFSETS_V2
    rng = random.Random(seed + 3000)
    specs: List[ItemSpec] = []
    global_idx = 0

    for domain_id in domains:
        dcfg = DOMAINS[domain_id]
        for difficulty in difficulties:
            for offset in anchor_offsets:
                for i in range(n_per_cell):
                    global_idx += 1
                    theta = rng.randint(THETA_MIN, THETA_MAX)

                    anchor_low = max(0, theta - offset)
                    anchor_high = min(100, theta + offset)
                    anchor_gap = anchor_high - anchor_low

                    evidence = _generate_evidence_v2(
                        rng, theta, dcfg.evidence_labels, difficulty
                    )
                    y_star_theta = theta
                    y_star_evidence = _compute_y_star_evidence(evidence)

                    visible_values = [
                        e["value"] for e in evidence if not e.get("missing")
                    ]
                    missing_idxs = [
                        e["index"] for e in evidence if e.get("missing")
                    ]

                    y_star_components = {
                        "evidence_values_raw": [e["value_raw"] for e in evidence],
                        "evidence_values_visible": visible_values,
                        "evidence_mean": round(
                            sum(visible_values) / len(visible_values), 2
                        ),
                        "evidence_std": round(
                            statistics.pstdev(visible_values), 2
                        ) if len(visible_values) > 1 else 0.0,
                        "evidence_count": len(visible_values),
                        "missing_indices": missing_idxs,
                        "y_star_theta": y_star_theta,
                        "y_star_evidence": y_star_evidence,
                        "aggregation_rule": "round(mean(visible_values))",
                        "theta": theta,
                    }

                    template_idx = global_idx % len(dcfg.scenario_templates)
                    template_family = f"{domain_id}_{template_idx:02d}"

                    item_id = f"TOOLv2-{domain_id}-{difficulty[0]}-off{offset}-{i+1:03d}"

                    spec = ItemSpec(
                        item_id=item_id,
                        suite="tool_v2",
                        domain=domain_id,
                        template_family=template_family,
                        theta=theta,
                        y_star=y_star_evidence,
                        y_star_components=y_star_components,
                        y_star_theta=y_star_theta,
                        y_star_evidence=y_star_evidence,
                        difficulty=difficulty,
                        anchors={
                            "low": anchor_low,
                            "high": anchor_high,
                            "gap": anchor_gap,
                            "offset": offset,
                            "anchor_type": "numeric",
                            "relevance": "stratified",
                        },
                        evidence_structured=evidence,
                        tags={"split": split, "difficulty": difficulty},
                        seed=seed,
                        generator_version=generator_version,
                        render_version="2.0.0",
                        tool={
                            "available_tools": [
                                "get_evidence_summary",
                                "check_external_reference",
                            ],
                            "mode_supported": ["injected", "agentic"],
                            "tool_schema_version": "v2",
                        },
                    )
                    specs.append(spec)

    return specs


# ── ICL suite v2 generation ──────────────────────────────────────────

ICL_V2_N_DEMOS = 3
ICL_V2_DEMO_EVIDENCE_COUNT = 3


def _generate_icl_demos(
    rng: random.Random,
    domain_id: str,
) -> List[Dict[str, Any]]:
    """Generate N_DEMOS fixed neutral demos for an ICL v2 item.

    Each demo has DEMO_EVIDENCE_COUNT ratings and a neutral answer
    (round of mean). Values are drawn from a moderate range to avoid
    correlation with anchor direction.
    """
    dcfg = DOMAINS[domain_id]
    labels = dcfg.evidence_labels[:ICL_V2_DEMO_EVIDENCE_COUNT]
    demos = []
    for di in range(ICL_V2_N_DEMOS):
        theta_demo = rng.randint(35, 65)
        evidence = []
        for lbl in labels:
            v = max(0, min(100, round(rng.gauss(theta_demo, 6))))
            evidence.append({"label": lbl, "value": v})
        answer = round(sum(e["value"] for e in evidence) / len(evidence))
        demos.append({"evidence": evidence, "answer": answer})
    return demos


def generate_icl_v2_itemspecs(
    domains: Optional[List[str]] = None,
    n_per_cell: int = 5,
    seed: int = 42,
    generator_version: str = "",
    split: str = "pilot",
    difficulties: Optional[List[str]] = None,
    anchor_offsets: Optional[List[int]] = None,
) -> List[ItemSpec]:
    """Generate ICL v2 ItemSpec records (metadata priming).

    Same backbone as External/RAG/Tool v2: iterates over
    (domain, difficulty, anchor_offset, item_idx).
    Each item stores pre-generated demo evidence/answers in tags["icl_demos"]
    so that the renderer can produce identical demos across conditions.
    """
    domains = domains or DOMAIN_IDS
    difficulties = difficulties or ["easy", "hard"]
    anchor_offsets = anchor_offsets or ANCHOR_OFFSETS_V2
    rng = random.Random(seed + 4000)
    specs: List[ItemSpec] = []
    global_idx = 0

    for domain_id in domains:
        dcfg = DOMAINS[domain_id]
        for difficulty in difficulties:
            for offset in anchor_offsets:
                for i in range(n_per_cell):
                    global_idx += 1
                    theta = rng.randint(THETA_MIN, THETA_MAX)

                    anchor_low = max(0, theta - offset)
                    anchor_high = min(100, theta + offset)
                    anchor_gap = anchor_high - anchor_low

                    evidence = _generate_evidence_v2(
                        rng, theta, dcfg.evidence_labels, difficulty
                    )
                    y_star_theta = theta
                    y_star_evidence = _compute_y_star_evidence(evidence)

                    visible_values = [
                        e["value"] for e in evidence if not e.get("missing")
                    ]
                    missing_idxs = [
                        e["index"] for e in evidence if e.get("missing")
                    ]

                    y_star_components = {
                        "evidence_values_raw": [e["value_raw"] for e in evidence],
                        "evidence_values_visible": visible_values,
                        "evidence_mean": round(
                            sum(visible_values) / len(visible_values), 2
                        ),
                        "evidence_std": round(
                            statistics.pstdev(visible_values), 2
                        ) if len(visible_values) > 1 else 0.0,
                        "evidence_count": len(visible_values),
                        "missing_indices": missing_idxs,
                        "y_star_theta": y_star_theta,
                        "y_star_evidence": y_star_evidence,
                        "aggregation_rule": "round(mean(visible_values))",
                        "theta": theta,
                    }

                    template_idx = global_idx % len(dcfg.scenario_templates)
                    template_family = f"{domain_id}_{template_idx:02d}"

                    item_id = f"ICLv2-{domain_id}-{difficulty[0]}-off{offset}-{i+1:03d}"

                    icl_demos = _generate_icl_demos(rng, domain_id)

                    spec = ItemSpec(
                        item_id=item_id,
                        suite="icl_v2",
                        domain=domain_id,
                        template_family=template_family,
                        theta=theta,
                        y_star=y_star_evidence,
                        y_star_components=y_star_components,
                        y_star_theta=y_star_theta,
                        y_star_evidence=y_star_evidence,
                        difficulty=difficulty,
                        anchors={
                            "low": anchor_low,
                            "high": anchor_high,
                            "gap": anchor_gap,
                            "offset": offset,
                            "anchor_type": "numeric",
                            "relevance": "stratified",
                        },
                        evidence_structured=evidence,
                        tags={
                            "split": split,
                            "difficulty": difficulty,
                            "icl_demos": icl_demos,
                        },
                        seed=seed,
                        generator_version=generator_version,
                        render_version="2.0.0",
                    )
                    specs.append(spec)

    return specs
