"""Shared constants and helpers for all suite renderers.

Every suite shares:
  - the 5 matched conditions (control + 2 relevance × 2 direction)
  - evidence formatting (with missing-value handling)
  - template resolution from spec metadata indices
"""

from __future__ import annotations

from ..domains import ALL_DOMAINS as DOMAINS
from ..domains import DomainConfig
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec

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

# Extension-only conditions for the plausible-intensity probe. Used only by
# the dedicated intensity dataset generator; the core External dataset is
# unchanged.
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
    anchor_value: int,
) -> str:
    """The anchor sentence for one relevance framing (Sec. 3.2).

    Every domain declares a pool per framing; a missing pool is a config
    error, not a control prompt in disguise, so it raises.
    """
    pool = dcfg.anchor_preambles[relevance]
    return pool[phrasing_idx % len(pool)].format(anchor=anchor_value)


def render_demos(
    demos: list[tuple[list[dict], int]], headers: list[str],
) -> tuple[str, list[int], list[str]]:
    """The few-shot block: one header, evidence and answer per demo.

    Returns (demos_text, answers, evidence_strings) for the prompt and its
    components.
    """
    blocks, answers, evidence_strs = [], [], []
    for header, (demo_ev, demo_ans) in zip(headers, demos):
        ev_str = format_evidence(demo_ev, show_missing=False)
        blocks.append(f"{header}\n{ev_str}\nAnswer: {demo_ans}")
        answers.append(demo_ans)
        evidence_strs.append(ev_str)
    return "\n\n".join(blocks), answers, evidence_strs


def anchored_prompt(
    base_text: str,
    question: str,
    dcfg: DomainConfig,
    relevance: str,
    phrasing_idx: int,
    anchor_value: int | None,
) -> tuple[str, str | None, str | None, list[int] | None]:
    """Finish a prompt: the question and answer instruction after ``base_text``,
    with the anchor sentence inserted before the question in an anchored
    condition. Returns (prompt_text, anchor_sentence, anchor_string,
    anchor_span); the last three are None for the control.
    """
    tail = f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    if relevance == "none" or anchor_value is None:
        return f"{base_text}{tail}", None, None, None
    sentence = resolve_anchor_preamble(dcfg, relevance, phrasing_idx, anchor_value)
    anchor_string = str(anchor_value)
    prompt_text = f"{base_text}{sentence} {tail}"
    start = prompt_text.find(anchor_string, len(base_text))
    span = [start, start + len(anchor_string)] if start >= 0 else None
    return prompt_text, sentence, anchor_string, span
