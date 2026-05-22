"""Generate canonical ItemSpec records for AnchorBench.

Gold rule (all suites):
  theta ∈ [30, 70]
  anchor offsets stratified: {15, 25, 40}
  y_star_evidence = round(aggregation(visible evidence values))
  y_star          = y_star_evidence

Evidence: 5 ratings drawn around theta with difficulty-dependent noise.
  easy:   sigma=8,  0 missing
  medium: sigma=12, 1 missing
  hard:   sigma=15, 2 missing, 1 conflicting value

Scoring functions (core = "mean"):
  mean:          round(mean(visible))
  weighted_mean: round(weighted_mean(visible, weights))
  median:        round(median(visible))
"""

from __future__ import annotations

import itertools
import random
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from .domains import DOMAINS, DOMAIN_IDS, DomainConfig
from .schema import ItemSpec

# ── Shared constants ─────────────────────────────────────────────────

THETA_MIN, THETA_MAX = 30, 70
EVIDENCE_COUNT = 5

ANCHOR_OFFSETS = [15, 25, 40]
EASY_SIGMA = 8
MEDIUM_SIGMA = 12
HARD_SIGMA = 15
HARD_MISSING_COUNT = 2
MEDIUM_MISSING_COUNT = 1


# ── Scoring functions ────────────────────────────────────────────────

def compute_gold_answer(
    visible_values: List[int],
    scoring_function: str = "mean",
    weights: Optional[List[float]] = None,
) -> int:
    """Compute deterministic gold answer from visible evidence values."""
    if not visible_values:
        return 0
    if scoring_function == "mean":
        return round(sum(visible_values) / len(visible_values))
    if scoring_function == "weighted_mean":
        if weights is None:
            return round(sum(visible_values) / len(visible_values))
        w = weights[:len(visible_values)]
        total = sum(v * wi for v, wi in zip(visible_values, w))
        return round(total / sum(w))
    if scoring_function == "median":
        return round(statistics.median(visible_values))
    raise ValueError(f"Unknown scoring function: {scoring_function!r}")


# ── Evidence generation ──────────────────────────────────────────────

def _generate_evidence(
    rng: random.Random,
    theta: int,
    labels: List[str],
    difficulty: str,
) -> List[dict]:
    """Generate 5 evidence ratings with difficulty-dependent structure.

    Easy:   5 visible concordant ratings, sigma=8.
    Medium: 5 slots (1 missing), sigma=12, no conflict.
    Hard:   5 slots (2 missing), sigma=15, 1 visible value flipped for conflict.
    """
    if difficulty == "easy":
        sigma = EASY_SIGMA
        n_missing = 0
    elif difficulty == "medium":
        sigma = MEDIUM_SIGMA
        n_missing = MEDIUM_MISSING_COUNT
    else:
        sigma = HARD_SIGMA
        n_missing = HARD_MISSING_COUNT

    raw_values = [max(0, min(100, round(rng.gauss(theta, sigma)))) for _ in labels]

    missing_indices: List[int] = []
    if n_missing > 0:
        indices = list(range(len(labels)))
        missing_indices = sorted(rng.sample(indices, n_missing))

    if difficulty == "hard" and missing_indices:
        visible_indices = [i for i in range(len(labels)) if i not in missing_indices]
        flip_idx = rng.choice(visible_indices)
        raw_values[flip_idx] = max(0, min(100, 100 - raw_values[flip_idx]))

    return [
        {
            "type": "rating",
            "value": None if i in missing_indices else raw_values[i],
            "value_raw": raw_values[i],
            "label": label,
            "index": i,
            "missing": i in missing_indices,
        }
        for i, label in enumerate(labels)
    ]


def _visible_values(evidence: List[dict]) -> List[int]:
    """Extract non-missing evidence values."""
    return [e["value"] for e in evidence if not e.get("missing")]


def _build_y_star_components(
    evidence: List[dict],
    theta: int,
    y_star_evidence: int,
    scoring_function: str = "mean",
) -> Dict[str, Any]:
    """Construct the y_star derivation record."""
    visible = _visible_values(evidence)
    return {
        "evidence_values_raw": [e["value_raw"] for e in evidence],
        "evidence_values_visible": visible,
        "evidence_mean": round(sum(visible) / len(visible), 2) if visible else 0.0,
        "evidence_std": round(statistics.pstdev(visible), 2) if len(visible) > 1 else 0.0,
        "evidence_count": len(visible),
        "missing_indices": [e["index"] for e in evidence if e.get("missing")],
        "y_star_theta": theta,
        "y_star_evidence": y_star_evidence,
        "aggregation_rule": scoring_function,
        "theta": theta,
    }


# ── Shared backbone for offset-stratified suites ─────────────────────

def _generate_offset_grid_specs(
    suite: str,
    id_prefix: str,
    seed_offset: int,
    *,
    domains: Optional[List[str]] = None,
    n_per_cell: int = 5,
    seed: int = 42,
    generator_version: str = "",
    split: str = "pilot",
    difficulties: Optional[List[str]] = None,
    anchor_offsets: Optional[List[int]] = None,
    scoring_function: str = "mean",
    scoring_weights: Optional[List[float]] = None,
    extra_fn: Optional[Callable] = None,
) -> List[ItemSpec]:
    """Generate ItemSpecs on the (domain × difficulty × offset × idx) grid.

    Suite-specific metadata is injected via extra_fn(rng, spec_kwargs, dcfg).
    """
    domains = domains or DOMAIN_IDS
    difficulties = difficulties or ["easy", "hard"]
    anchor_offsets = anchor_offsets or ANCHOR_OFFSETS
    rng = random.Random(seed + seed_offset)
    specs: List[ItemSpec] = []
    global_idx = 0

    for domain_id in domains:
        dcfg = DOMAINS[domain_id]
        for difficulty in difficulties:
            for offset in anchor_offsets:
                for i in range(n_per_cell):
                    global_idx += 1
                    theta = rng.randint(THETA_MIN, THETA_MAX)

                    label_family_idx = global_idx % dcfg.n_label_families
                    labels = dcfg.evidence_label_families[label_family_idx]

                    evidence = _generate_evidence(rng, theta, labels, difficulty)
                    visible = _visible_values(evidence)
                    y_star_evidence = compute_gold_answer(
                        visible, scoring_function, scoring_weights,
                    )

                    s_idx = global_idx % len(dcfg.scenario_templates)
                    q_idx = global_idx % len(dcfg.question_templates)
                    a_idx = global_idx % len(dcfg.anchor_preambles.get("irrelevant", [""]))

                    item_id = f"{id_prefix}-{domain_id}-{difficulty[0]}-off{offset}-{i+1:03d}"

                    spec_kwargs: Dict[str, Any] = dict(
                        item_id=item_id,
                        suite=suite,
                        domain=domain_id,
                        template_family=f"{domain_id}_{s_idx:02d}",
                        theta=theta,
                        y_star=y_star_evidence,
                        y_star_components=_build_y_star_components(
                            evidence, theta, y_star_evidence, scoring_function,
                        ),
                        y_star_theta=theta,
                        y_star_evidence=y_star_evidence,
                        difficulty=difficulty,
                        anchors={
                            "low": max(0, theta - offset),
                            "high": min(100, theta + offset),
                            "gap": min(100, theta + offset) - max(0, theta - offset),
                            "offset": offset,
                            "anchor_type": "numeric",
                            "relevance": "stratified",
                        },
                        evidence_structured=evidence,
                        tags={"split": split, "difficulty": difficulty},
                        seed=seed,
                        generator_version=generator_version,
                        render_version="2.1.0",
                        evidence_label_family_idx=label_family_idx,
                        scenario_template_idx=s_idx,
                        question_template_idx=q_idx,
                        anchor_phrasing_idx=a_idx,
                        scoring_function=scoring_function,
                        scoring_weights=scoring_weights,
                    )

                    if extra_fn is not None:
                        extra_fn(rng, spec_kwargs, dcfg)

                    specs.append(ItemSpec(**spec_kwargs))

    return specs


# ── Per-suite wrappers ───────────────────────────────────────────────

def generate_external_itemspecs(**kwargs) -> List[ItemSpec]:
    """External-suite ItemSpecs (no suite-specific metadata)."""
    return _generate_offset_grid_specs("external", "EXT", seed_offset=0, **kwargs)


def _rag_extras(rng, kw, dcfg):
    kw["rag"] = {
        "corpus_id": "anchorbench_corpus",
        "corpus_size": 3,
        "retrieval_method": "deterministic_fixed_order",
        "doc_roles": ["core", "anchor_slot", "filler"],
    }


def generate_rag_itemspecs(**kwargs) -> List[ItemSpec]:
    """RAG ItemSpecs (adds frozen 3-doc mini-corpus metadata)."""
    return _generate_offset_grid_specs("rag", "RAG", seed_offset=2000,
                                      extra_fn=_rag_extras, **kwargs)


def _tool_extras(rng, kw, dcfg):
    kw["tool"] = {
        "available_tools": ["get_evidence_summary", "check_external_reference"],
        "mode_supported": ["injected", "agentic"],
        "tool_schema_version": "v2",
    }


def generate_tool_itemspecs(**kwargs) -> List[ItemSpec]:
    """Tool ItemSpecs (adds tool-calling metadata)."""
    return _generate_offset_grid_specs("tool", "TOOL", seed_offset=3000,
                                      extra_fn=_tool_extras, **kwargs)


def generate_tool_agentic_itemspecs(**kwargs) -> List[ItemSpec]:
    """Tool-Agentic ItemSpecs (same grid as tool)."""
    return _generate_offset_grid_specs("tool_agentic", "TOOL-A", seed_offset=3000,
                                      extra_fn=_tool_extras, **kwargs)


def _tool_read_extras(rng, kw, dcfg):
    kw["tool"] = {
        "available_tools": ["get_evidence_summary", "check_external_reference"],
        "mode_supported": ["read_only"],
        "tool_schema_version": "v2",
    }


def generate_tool_read_itemspecs(**kwargs) -> List[ItemSpec]:
    """Tool-Read ItemSpecs (same grid, plain JSON rendering)."""
    return _generate_offset_grid_specs("tool_read", "TOOL-R", seed_offset=3000,
                                      extra_fn=_tool_read_extras, **kwargs)


ICL_N_DEMOS = 3
ICL_DEMO_EVIDENCE_COUNT = 3


def _generate_icl_demos(rng: random.Random, domain_id: str) -> List[Dict[str, Any]]:
    """Generate N neutral demos for an ICL item."""
    labels = DOMAINS[domain_id].evidence_labels[:ICL_DEMO_EVIDENCE_COUNT]
    demos = []
    for _ in range(ICL_N_DEMOS):
        theta_demo = rng.randint(35, 65)
        evidence = [{"label": lbl, "value": max(0, min(100, round(rng.gauss(theta_demo, 6))))}
                    for lbl in labels]
        demos.append({"evidence": evidence, "answer": round(sum(e["value"] for e in evidence) / len(evidence))})
    return demos


def _icl_extras(rng, kw, dcfg):
    kw["tags"]["icl_demos"] = _generate_icl_demos(rng, kw["domain"])


def generate_icl_itemspecs(**kwargs) -> List[ItemSpec]:
    """ICL ItemSpecs (adds pre-generated demo examples)."""
    return _generate_offset_grid_specs("icl", "ICL", seed_offset=4000,
                                      extra_fn=_icl_extras, **kwargs)


def _generate_icl_dist_demos(
    rng: random.Random,
    domain_id: str,
    anchor_low: int,
    anchor_high: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Pre-generate three demo lists: neutral mid-band, low-clustered, high-clustered."""
    labels = DOMAINS[domain_id].evidence_labels[:ICL_DEMO_EVIDENCE_COUNT]

    def pack(center: int, spread: int) -> List[Dict[str, Any]]:
        demos: List[Dict[str, Any]] = []
        for _ in range(ICL_N_DEMOS):
            theta_d = max(0, min(100, center + rng.randint(-spread, spread)))
            evidence = [
                {"label": lbl, "value": max(0, min(100, round(rng.gauss(theta_d, 6))))}
                for lbl in labels
            ]
            demos.append(
                {
                    "evidence": evidence,
                    "answer": round(sum(e["value"] for e in evidence) / len(evidence)),
                }
            )
        return demos

    neutral_center = rng.randint(44, 56)
    return {
        "icl_dist_demos_control": pack(neutral_center, spread=8),
        "icl_dist_demos_low": pack(anchor_low, spread=4),
        "icl_dist_demos_high": pack(anchor_high, spread=4),
    }


def _icl_dist_extras(rng, kw, dcfg):
    low = kw["anchors"]["low"]
    high = kw["anchors"]["high"]
    for key, demos in _generate_icl_dist_demos(rng, kw["domain"], low, high).items():
        kw["tags"][key] = demos


def generate_icl_dist_itemspecs(**kwargs) -> List[ItemSpec]:
    """ICL-distribution ItemSpecs (neutral vs low vs high demo bands + framing)."""
    return _generate_offset_grid_specs("icl_dist", "ICLDIST", seed_offset=4500,
                                      extra_fn=_icl_dist_extras, **kwargs)


# ── History suite (different grid: domain × difficulty, no offset) ───

HISTORY_SUBSET_DELTA = 15
WARMUP_LOW_THETA_MIN, WARMUP_LOW_THETA_MAX = 15, 30
WARMUP_HIGH_THETA_MIN, WARMUP_HIGH_THETA_MAX = 70, 85


def _enumerate_2_of_5_subsets() -> List[Tuple[int, int]]:
    return list(itertools.combinations(range(5), 2))


def _subset_mean(evidence: List[dict], indices: Tuple[int, ...]) -> float:
    vals = [evidence[i]["value"] for i in indices
            if i < len(evidence) and not evidence[i].get("missing") and evidence[i].get("value") is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _choose_plausible_subsets(
    evidence: List[dict], y_star_evidence: int, delta: int = HISTORY_SUBSET_DELTA,
) -> Dict[str, Any]:
    """Choose low- and high-pressure 2-of-5 subsets by mean closest to y_star ± delta."""
    visible = [e["value"] for e in evidence if not e.get("missing")]
    if len(visible) < 2:
        return {
            "subset_indices_low": [0, 1], "subset_indices_high": [0, 1],
            "subset_mean_low": 0.0, "subset_mean_high": 0.0,
            "realized_gap_low": 0.0, "realized_gap_high": 0.0,
        }
    subsets = _enumerate_2_of_5_subsets()
    target_low = y_star_evidence - delta
    target_high = y_star_evidence + delta
    best_low = min(subsets, key=lambda s: abs(_subset_mean(evidence, s) - target_low))
    best_high = min(subsets, key=lambda s: abs(_subset_mean(evidence, s) - target_high))
    mean_low = _subset_mean(evidence, best_low)
    mean_high = _subset_mean(evidence, best_high)
    return {
        "subset_indices_low": list(best_low), "subset_indices_high": list(best_high),
        "subset_mean_low": round(mean_low, 2), "subset_mean_high": round(mean_high, 2),
        "realized_gap_low": round(abs(mean_low - target_low), 2),
        "realized_gap_high": round(abs(mean_high - target_high), 2),
    }


def _build_warmup_case(
    rng: random.Random, domain_id: str, theta_min: int, theta_max: int, template_idx: int,
) -> Dict[str, Any]:
    """Build a warmup case (same domain, different case) for irrelevant conditions."""
    dcfg = DOMAINS[domain_id]
    labels = dcfg.evidence_labels
    theta = rng.randint(theta_min, theta_max)
    evidence = _generate_evidence(rng, theta, labels, "easy")
    return {
        "theta": theta,
        "evidence_structured": evidence,
        "y_star_evidence": compute_gold_answer(_visible_values(evidence)),
        "scenario_template_idx": template_idx,
    }


def generate_history_itemspecs(
    domains: Optional[List[str]] = None,
    n_per_cell: int = 5,
    seed: int = 42,
    generator_version: str = "",
    split: str = "pilot",
    difficulties: Optional[List[str]] = None,
    subset_delta: int = HISTORY_SUBSET_DELTA,
    scoring_function: str = "mean",
    scoring_weights: Optional[List[float]] = None,
) -> List[ItemSpec]:
    """Generate History ItemSpecs (domain × difficulty grid, no offset dimension)."""
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

                label_family_idx = global_idx % dcfg.n_label_families
                labels = dcfg.evidence_label_families[label_family_idx]

                evidence = _generate_evidence(rng, theta, labels, difficulty)
                visible = _visible_values(evidence)
                y_star_evidence = compute_gold_answer(
                    visible, scoring_function, scoring_weights,
                )

                plausible_data = _choose_plausible_subsets(evidence, y_star_evidence, delta=subset_delta)

                s_idx = global_idx % len(dcfg.scenario_templates)
                q_idx = global_idx % len(dcfg.question_templates)
                a_idx = global_idx % len(dcfg.anchor_preambles.get("irrelevant", [""]))

                warmup_tidx = (s_idx + 1) % len(dcfg.scenario_templates)
                warmup_low = _build_warmup_case(rng, domain_id, WARMUP_LOW_THETA_MIN, WARMUP_LOW_THETA_MAX, warmup_tidx)
                warmup_high = _build_warmup_case(rng, domain_id, WARMUP_HIGH_THETA_MIN, WARMUP_HIGH_THETA_MAX,
                                                 (warmup_tidx + 1) % len(dcfg.scenario_templates))

                item_id = f"HIST-{domain_id}-{difficulty[0]}-{i+1:03d}"

                specs.append(ItemSpec(
                    item_id=item_id,
                    suite="history",
                    domain=domain_id,
                    template_family=f"{domain_id}_{s_idx:02d}",
                    theta=theta,
                    y_star=y_star_evidence,
                    y_star_components=_build_y_star_components(
                        evidence, theta, y_star_evidence, scoring_function,
                    ),
                    y_star_theta=theta,
                    y_star_evidence=y_star_evidence,
                    difficulty=difficulty,
                    anchors={"low": max(0, theta - 25), "high": min(100, theta + 25),
                             "anchor_type": "numeric", "relevance": "self_generated"},
                    evidence_structured=evidence,
                    tags={"split": split, "difficulty": difficulty},
                    seed=seed,
                    generator_version=generator_version,
                    render_version="2.1.0",
                    evidence_label_family_idx=label_family_idx,
                    scenario_template_idx=s_idx,
                    question_template_idx=q_idx,
                    anchor_phrasing_idx=a_idx,
                    scoring_function=scoring_function,
                    scoring_weights=scoring_weights,
                    history={**plausible_data, "warmup_low": warmup_low,
                             "warmup_high": warmup_high, "subset_delta": subset_delta},
                ))

    return specs
