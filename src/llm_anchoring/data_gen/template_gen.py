"""Generate digit-free NL templates via OpenRouter and validate them."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from ..openrouter_client import chat, extract_content
from .prompt_text import SYSTEM_PROMPT, retry_prompt, template_gen_prompt
from .validators import validate_template

logger = logging.getLogger(__name__)
MAX_RETRIES = 2


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from the LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed: %.120s…", text)
        return None


def generate_template(
    spec: dict[str, Any],
    *,
    model: str,
    cache_dir: Any,
) -> dict[str, Any] | None:
    """Generate and validate one template from a spec dict.

    Returns a template record ready for JSONL, or None on failure.
    """
    required = spec["required_placeholders"]
    allow_extra = spec.get("allow_extra_placeholders", False)

    user_msg = template_gen_prompt(
        domain=spec["domain"],
        description=spec["description"],
        required_placeholders=required,
        allow_extra=allow_extra,
        format_hint=spec.get("format_hint", "plain digits with two decimals"),
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(1 + MAX_RETRIES):
        resp = chat(messages, model=model, cache_dir=cache_dir)
        text = extract_content(resp)
        parsed = _parse_json_response(text)
        if parsed is None:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Return valid JSON only."})
            continue

        errors = validate_template(parsed, required, allow_extra)
        if not errors:
            return _build_record(spec, parsed, model)

        logger.info("attempt %d errors: %s", attempt, errors)
        if attempt < MAX_RETRIES:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": retry_prompt(errors)})

    logger.error("skipping spec %s after %d attempts", spec.get("id"), 1 + MAX_RETRIES)
    return None


def _build_record(
    spec: dict[str, Any], tpl: dict[str, Any], model: str
) -> dict[str, Any]:
    return {
        "template_id": spec["id"],
        "domain": spec["domain"],
        "language": spec.get("language", "en"),
        "format_spec_id": spec.get("format_spec_id", "currency_us"),
        "context_template": tpl["context_template"],
        "question_template": tpl["question_template"],
        "placeholders": tpl.get("placeholders", []),
        "model_used": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
