"""Prompt templates sent to OpenRouter for template generation."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a dataset-template author for cognitive-bias research. "
    "You produce STRICT JSON with no markdown fences, no commentary."
)


def template_gen_prompt(
    domain: str,
    description: str,
    required_placeholders: list[str],
    allow_extra: bool,
    format_hint: str,
) -> str:
    """Build the user prompt for generating one template."""
    ph_list = ", ".join(f"{{{p}}}" for p in required_placeholders)
    extra_rule = (
        "You MAY add extra placeholders if they improve realism."
        if allow_extra
        else "Do NOT introduce any placeholders beyond the required ones."
    )
    return (
        f"Domain: {domain}\n"
        f"Scenario description: {description}\n\n"
        f"Required placeholders (use exactly these): {ph_list}\n"
        f"{extra_rule}\n\n"
        "Return STRICT JSON (no markdown, no code fences) with keys:\n"
        '  "context_template": a realistic paragraph using the placeholders. '
        "Display values should use the _DISPLAY suffix placeholder.\n"
        '  "question_template": a question asking the reader to produce the '
        f"answer in {format_hint} format.\n"
        '  "placeholders": list of all placeholder names you used.\n\n'
        "CRITICAL RULES:\n"
        "- context_template and question_template must contain ZERO digits "
        "(0-9). Express any numbers in words (e.g. 'two' not '2').\n"
        "- Use curly-brace placeholders like {SUBTOTAL_DISPLAY} for values.\n"
    )


def retry_prompt(errors: list[str]) -> str:
    """Build a follow-up prompt asking the LLM to fix validation errors."""
    joined = "; ".join(errors)
    return (
        f"Your previous template was invalid: {joined}.\n"
        "Please regenerate the JSON, ensuring:\n"
        "- Absolutely NO digits (0-9) in context_template or question_template.\n"
        "- All required placeholders are present.\n"
        "- No extra placeholders unless allowed.\n"
        "Return only the corrected JSON."
    )
