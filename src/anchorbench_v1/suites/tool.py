"""Backward-compatibility shim — re-exports from tool_agentic.py.

All implementation has moved to tool_agentic.py and tool_read.py.
"""

from .tool_agentic import (  # noqa: F401
    TOOL_SCHEMAS,
    TOOL_EXECUTORS,
    execute_get_evidence_summary,
    execute_check_external_reference,
    render_tool,
    get_tool_messages,
    _build_messages,
    _build_prompt,
    _get_visible_ratings,
    _render_plaintext_fallback,
    _SYSTEM_PROMPT,
)
