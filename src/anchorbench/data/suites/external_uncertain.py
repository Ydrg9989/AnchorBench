"""External-uncertain suite.

Identical to the External suite except evidence is *partially withheld*,
creating genuine epistemic uncertainty: the gold answer remains the
arithmetic mean of the full (hidden) 5 ratings, but the model only sees
k of them (k in {1, 2, 3}). This makes the plausible-anchor shift
analyzable against an information-theoretic lower bound on rational
Bayesian updating.

Conditions per item (15 = 3 k-levels x 5 anchor conditions):
  uncertain_p1_{control,irrelevant_low/high,plausible_low/high}
  uncertain_p2_{control,irrelevant_low/high,plausible_low/high}
  uncertain_p3_{control,irrelevant_low/high,plausible_low/high}

The visible-rating subsets are deterministic per item to keep pairing
invariants across conditions. For k=1 we use index [0]; for k=2 we use
[0,2]; for k=3 we use [0,2,4]. These are spread across the evidence
to avoid privileging adjacent measurements.
"""

from __future__ import annotations

from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView
from ._shared import (
    CONDITIONS,
    format_evidence,
    resolve_anchor_preamble,
    resolve_templates,
)

# Deterministic visible-index subsets per k-level (5 evidence items total).
VISIBLE_INDICES: dict[int, list[int]] = {
    1: [0],
    2: [0, 2],
    3: [0, 2, 4],
}

K_LEVELS = (1, 2, 3)


def _build_prompt(
    spec: ItemSpec,
    k: int,
    condition: str,
    relevance: str,
    anchor_value: int | None,
) -> PromptView:
    scenario, question, _, dcfg = resolve_templates(spec)
    indices = VISIBLE_INDICES[k]
    partial_evidence = format_evidence(
        spec.evidence_structured, indices=indices,
    )
    n_total = len(spec.evidence_structured)
    n_shown = len(indices)
    notice = (
        f"(Note: only {n_shown} of {n_total} ratings are available "
        f"for this assessment.)"
    )
    base_text = (
        f"{scenario}\n\n"
        f"Partial evidence:\n{partial_evidence}\n\n"
        f"{notice}\n\n"
    )

    components = {
        "scenario": scenario,
        "evidence": partial_evidence,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
        "k_visible": n_shown,
        "n_total": n_total,
        "visible_indices": ",".join(str(i) for i in indices),
    }

    anchor_string: str | None = None
    anchor_span: list[int] | None = None

    if relevance == "none" or anchor_value is None:
        prompt_text = f"{base_text}{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    else:
        anchor_sentence = resolve_anchor_preamble(
            dcfg, relevance, spec.anchor_phrasing_idx, anchor_value,
        )
        if anchor_sentence is None:
            prompt_text = f"{base_text}{question}\n{ANSWER_FORMAT_INSTRUCTION}"
        else:
            components["anchor_sentence"] = anchor_sentence
            anchor_string = str(anchor_value)
            prompt_text = (
                f"{base_text}{anchor_sentence} "
                f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
            )
            start = prompt_text.find(anchor_string, len(base_text))
            if start >= 0:
                anchor_span = [start, start + len(anchor_string)]

    return PromptView(
        item_id=spec.item_id,
        suite="external_uncertain",
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text,
        prompt_components=components,
        anchor_string=anchor_string,
        anchor_span=anchor_span,
        anchor_relevance=relevance,
        anchor_value=anchor_value,
        provenance={"k_visible": n_shown, "n_total": n_total},
    )


def _conditions_for_k(k: int) -> list[tuple[str, str, str, str]]:
    """Return list of (full_condition_name, base_cond, relevance, direction)."""
    out = []
    for cond, rel, direction in CONDITIONS:
        # cond is one of: control, irrelevant_low/high, plausible_low/high
        full = f"uncertain_p{k}_{cond}"
        out.append((full, cond, rel, direction))
    return out


def render_external_uncertain(spec: ItemSpec) -> list[PromptView]:
    """Render 15 conditions per External item (3 k-levels x 5 conditions)."""
    if spec.suite != "external":
        return []
    views: list[PromptView] = []
    for k in K_LEVELS:
        for full, _base, rel, direction in _conditions_for_k(k):
            anchor = spec.anchors.get(direction) if direction else None
            views.append(_build_prompt(spec, k, full, rel, anchor))
    return views


# Helper: the visible-mean of an item at level k. Used by the analyzer
# to construct rational-Bayesian baselines.
def visible_mean(spec_dict: dict, k: int) -> float | None:
    """Compute the arithmetic mean of the visible ratings at k for an
    itemspec dict (as loaded from JSONL).
    """
    evidence = spec_dict.get("evidence_structured", [])
    indices = VISIBLE_INDICES[k]
    vals = []
    for i in indices:
        if i >= len(evidence):
            continue
        e = evidence[i]
        if e.get("missing"):
            continue
        v = e.get("value")
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)
