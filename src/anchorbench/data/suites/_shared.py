"""Shared constants and helpers for all suite renderers.

Every suite shares:
  - the 5 matched conditions (control + 2 relevance × 2 direction)
  - evidence formatting (with missing-value handling)
  - template resolution from spec metadata indices
"""

from __future__ import annotations

from ..domains import ALL_DOMAINS as DOMAINS
from ..domains import DomainConfig
from ..schema import ItemSpec

MISSING_VALUE_DISPLAY = "[data not available]"

CONDITIONS: list[tuple[str, str, str]] = [
    # (condition_name, relevance_type, anchor_direction)
    ("control",         "none",       ""),
    ("irrelevant_low",  "irrelevant", "low"),
    ("irrelevant_high", "irrelevant", "high"),
    ("plausible_low",   "plausible",  "low"),
    ("plausible_high",  "plausible",  "high"),
]

EXTENDED_CONDITIONS: list[tuple[str, str, str]] = CONDITIONS + [
    ("placebo_low",    "placebo",   "low"),
    ("placebo_high",   "placebo",   "high"),
    ("authority_low",  "authority", "low"),
    ("authority_high", "authority", "high"),
]

# Rebuttal-only D1 conditions for the plausible-intensity probe (Cluster H).
# Used only by the dedicated D1 dataset generator; the core External dataset
# is unchanged.
INTENSITY_CONDITIONS: list[tuple[str, str, str]] = CONDITIONS + [
    ("plausible_mild_low",    "plausible_mild",   "low"),
    ("plausible_mild_high",   "plausible_mild",   "high"),
    ("plausible_strong_low",  "plausible_strong", "low"),
    ("plausible_strong_high", "plausible_strong", "high"),
]

HISTORY_CONDITIONS: list[tuple[str, str, str]] = CONDITIONS + [
    ("control_twostage", "none", ""),
]

ICL_CONDITIONS: list[tuple[str, str, str]] = CONDITIONS + [
    ("neutral_low",  "neutral", "low"),
    ("neutral_high", "neutral", "high"),
]


def format_evidence(
    evidence: list[dict],
    show_missing: bool = True,
    indices: list[int] | None = None,
) -> str:
    """Format evidence ratings as indented bullet lines."""
    if indices is not None:
        evidence = [e for i, e in enumerate(evidence) if i in indices]
    lines = []
    for e in evidence:
        if show_missing and e.get("missing"):
            lines.append(f"  - {e['label']}: {MISSING_VALUE_DISPLAY}")
        else:
            val = e.get("value")
            lines.append(
                f"  - {e['label']}: {val}"
                if val is not None
                else f"  - {e['label']}: {MISSING_VALUE_DISPLAY}"
            )
    return "\n".join(lines)


def resolve_templates(spec: ItemSpec) -> tuple[str, str, int, DomainConfig]:
    """Resolve scenario text and question template from spec metadata.

    Uses the auditable indices stored in the spec to select templates.
    Returns (scenario, question, scenario_template_idx, domain_config).
    """
    dcfg = DOMAINS[spec.domain]
    s_idx = spec.scenario_template_idx % len(dcfg.scenario_templates)
    q_idx = spec.question_template_idx % len(dcfg.question_templates)
    scenario = spec.scenario_text or dcfg.scenario_templates[s_idx]
    question = dcfg.question_templates[q_idx]
    return scenario, question, s_idx, dcfg


def resolve_anchor_preamble(
    dcfg: DomainConfig,
    relevance: str,
    phrasing_idx: int,
    anchor_value: int | None,
) -> str | None:
    """Select and format the anchor preamble for a given relevance type."""
    if relevance == "none" or anchor_value is None:
        return None
    preamble_pool = dcfg.anchor_preambles.get(relevance, [])
    if not preamble_pool:
        return None
    preamble_template = preamble_pool[phrasing_idx % len(preamble_pool)]
    return preamble_template.format(anchor=anchor_value)
