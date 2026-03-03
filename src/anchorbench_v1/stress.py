"""Stress-split generator: 200 items tagged with stress factors.

Stress factors:
  - anchor_position: early / late  (where anchor appears in prompt)
  - anchor_format:   digits / words (numeric vs written-out anchor)
  - anchor_magnitude: mild / extreme
  - rag_order:       anchor_doc_first / anchor_doc_last (RAG-suite only)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from .domains import DOMAINS, DOMAIN_IDS
from .itemspec_gen import (
    ANCHOR_OFFSET,
    EVIDENCE_COUNT,
    EVIDENCE_SIGMA,
    THETA_MAX,
    THETA_MIN,
    _generate_evidence,
    _evidence_to_theta_mapping,
    _make_item_id,
)
from .schema import ItemSpec

STRESS_FACTORS = {
    "anchor_position": ["early", "late"],
    "anchor_format": ["digits", "words"],
    "anchor_magnitude": ["mild", "extreme"],
    "rag_order": ["anchor_doc_first", "anchor_doc_last"],
}

_NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty",
}

MILD_OFFSET = 15
EXTREME_OFFSET = 45


def anchor_to_words(n: int) -> str:
    """Best-effort number-to-words for 0-100."""
    if n in _NUMBER_WORDS:
        return _NUMBER_WORDS[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        tens_words = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
                      60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
        base = tens_words.get(tens, str(tens))
        if ones == 0:
            return base
        return f"{base}-{_NUMBER_WORDS.get(ones, str(ones))}"
    return "one hundred" if n == 100 else str(n)


def generate_stress_specs(
    n_items: int = 200,
    seed: int = 99,
    generator_version: str = "",
) -> List[ItemSpec]:
    """Generate stress-split ItemSpec records with tagged perturbations."""
    rng = random.Random(seed)
    suites = ["external", "icl", "history", "rag", "tool"]
    specs = []

    items_per_suite = n_items // len(suites)

    for suite in suites:
        for i in range(items_per_suite):
            domain_id = DOMAIN_IDS[i % len(DOMAIN_IDS)]
            dcfg = DOMAINS[domain_id]

            theta = rng.randint(THETA_MIN, THETA_MAX)
            y_star = round(theta)

            magnitude = rng.choice(["mild", "extreme"])
            offset = MILD_OFFSET if magnitude == "mild" else EXTREME_OFFSET
            anchor_low = max(0, theta - offset)
            anchor_high = min(100, theta + offset)
            gap = anchor_high - anchor_low

            evidence = _generate_evidence(rng, theta, dcfg.evidence_labels, sigma=12)

            stress_tags = {
                "anchor_position": rng.choice(STRESS_FACTORS["anchor_position"]),
                "anchor_format": rng.choice(STRESS_FACTORS["anchor_format"]),
                "anchor_magnitude": magnitude,
            }
            if suite == "rag":
                stress_tags["rag_order"] = rng.choice(STRESS_FACTORS["rag_order"])

            template_idx = i % len(dcfg.scenario_templates)
            spec = ItemSpec(
                item_id=f"STRESS-{suite[:3].upper()}-{domain_id}-{i+1:03d}",
                suite=suite,
                domain=domain_id,
                template_family=f"{domain_id}_{template_idx:02d}",
                theta=theta,
                y_star=y_star,
                y_star_components={
                    "evidence_structured": [e.copy() for e in evidence],
                    "evidence_to_theta": _evidence_to_theta_mapping(evidence, theta),
                    "theta": theta,
                    "y_star": y_star,
                    "rounding_applied": False,
                    "clamping_applied": False,
                },
                anchors={
                    "low": anchor_low,
                    "high": anchor_high,
                    "gap": gap,
                    "anchor_type": "numeric",
                    "relevance": "normatively_irrelevant",
                },
                evidence_structured=evidence,
                tags={
                    "split": "stress",
                    "difficulty": "stress",
                    **stress_tags,
                },
                seed=seed,
                generator_version=generator_version,
            )
            specs.append(spec)

    return specs
