"""Render final dataset items from templates + specs with ICL conditions."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Any

from .render_utils import (
    build_demo_text,
    build_icl_block,
    plain_2dp,
    render_display_values,
    sample_variables,
)
from .safe_eval import safe_eval

ICL_CONDITIONS: dict[str, tuple[float, float] | None] = {
    "no_demo": None,
    "low_anchor_demo": (1.0, 50.0),
    "high_anchor_demo": (5_000.0, 50_000.0),
    "outlier_anchor_demo": (10_000_000.0, 1_000_000_000.0),
    "placeholder_demo": None,
}


def render_items(
    template: dict[str, Any],
    spec: dict[str, Any],
    *,
    n_items: int,
    rng: random.Random,
    demo_shots: int = 2,
    anchor_ranges: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Produce *n_items* dataset rows from one template, across all ICL conditions."""
    items: list[dict[str, Any]] = []
    ranges = spec["variable_ranges"]
    derived_exprs = spec.get("derived_variables", {})
    truth_expr = spec["truth_expression"]

    for _ in range(n_items):
        seed = rng.randint(0, 2**31)
        item_rng = random.Random(seed)

        raw_vars = sample_variables(ranges, item_rng)
        derived_vars = _compute_derived(derived_exprs, raw_vars)
        all_vars = {**raw_vars, **derived_vars}

        truth_value = safe_eval(truth_expr, all_vars)
        truth_formatted = plain_2dp(truth_value)

        display = render_display_values(
            all_vars, spec.get("format_spec_id", "currency_us")
        )
        fill = {**{k: str(v) for k, v in all_vars.items()}, **display}

        ctx = template["context_template"].format_map(_SafeDict(fill))
        question = template["question_template"].format_map(_SafeDict(fill))

        for condition in ICL_CONDITIONS:
            ar = _resolve_anchor_range(condition, anchor_ranges)
            anchor_val, prompt = _build_prompt(
                ctx, question, condition, item_rng, demo_shots, ar
            )
            items.append({
                "item_id": str(uuid.uuid4()),
                "template_id": template["template_id"],
                "domain": template["domain"],
                "language": template.get("language", "en"),
                "variables_raw": raw_vars,
                "variables_derived": derived_vars,
                "rendered_context": ctx,
                "rendered_question": question,
                "truth_value": truth_value,
                "truth_formatted": truth_formatted,
                "icl_condition": condition,
                "anchor_value": anchor_val,
                "final_prompt": prompt,
                "format_spec_id": template.get("format_spec_id", "currency_us"),
                "metadata": {
                    "seed": seed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            })
    return items


def _compute_derived(
    exprs: dict[str, str], raw: dict[str, float]
) -> dict[str, float]:
    env = dict(raw)
    for name, expr in exprs.items():
        env[name] = safe_eval(expr, env)
    return {k: v for k, v in env.items() if k not in raw}


def _resolve_anchor_range(
    condition: str, overrides: dict[str, list[float]] | None
) -> tuple[float, float] | None:
    default = ICL_CONDITIONS[condition]
    if overrides and condition in overrides:
        lo, hi = overrides[condition]
        return (lo, hi)
    return default


def _build_prompt(
    ctx: str,
    question: str,
    condition: str,
    rng: random.Random,
    shots: int,
    anchor_range: tuple[float, float] | None,
) -> tuple[float | None, str]:
    if condition == "no_demo":
        return None, f"{ctx}\n\n{question}"

    placeholder_mode = condition == "placeholder_demo"
    anchor_val: float | None = None
    demos: list[tuple[str, str]] = []

    for _ in range(shots):
        if placeholder_mode:
            demos.append(build_demo_text(None, placeholder_mode=True))
        else:
            assert anchor_range is not None
            a = round(rng.uniform(*anchor_range), 2)
            anchor_val = a
            demos.append(build_demo_text(a))

    icl_block = build_icl_block(demos)
    return anchor_val, f"{icl_block}\n\nNow answer:\n{ctx}\n\n{question}"


class _SafeDict(dict):
    """dict subclass that returns ``{key}`` for missing keys during format_map."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
