"""Generate canonical ItemSpec records for AnchorBench v1.

Gold rule:
  theta ∈ [30, 70]   (uniform, seed-controlled)
  anchor_low  = theta - 30
  anchor_high = theta + 30
  anchor_gap  = 60
  y_star      = round(theta)   (deterministic)

Evidence: 5 ratings drawn around theta with controlled noise.
"""

from __future__ import annotations

import hashlib
import random
from typing import List, Optional

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
