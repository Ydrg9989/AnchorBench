"""Suite renderers -- each transforms an ItemSpec into PromptView records.

Suites:
    external   -- anchor sentence before the question
    history    -- warmup case (irrelevant) or partial-evidence subset (plausible)
    icl        -- anchor in demo header metadata
    icl_dist   -- demo answer bands (low/mid/high) x plausible vs irrelevant
    rag        -- anchor document in the retrieved corpus
    tool       -- anchor field in tool output JSON (agentic format)
"""

from __future__ import annotations

from .external import render_external
from .history import render_history
from .icl import render_icl
from .icl_dist import render_icl_dist
from .rag import build_full_corpus, render_rag
from .tool import render_tool

SUITE_RENDERERS = {
    "external": render_external,
    "history": render_history,
    "icl": render_icl,
    "icl_dist": render_icl_dist,
    "rag": render_rag,
    "tool": render_tool,
    "tool_agentic": render_tool,
}

__all__ = [
    "SUITE_RENDERERS",
    "render_external",
    "render_history",
    "render_icl",
    "render_icl_dist",
    "render_rag",
    "render_tool",
    "build_full_corpus",
]
