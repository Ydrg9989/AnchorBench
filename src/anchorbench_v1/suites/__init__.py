"""Suite renderers — each transforms an ItemSpec into PromptView records.

Suites:
  external      → anchor sentence before the question
  history       → warmup case (irrelevant) or partial-evidence subset (plausible)
  icl           → anchor in demo header metadata
  icl_dist      → demo answer bands (low/mid/high) × plausible vs irrelevant framing
  rag           → anchor document in the retrieved corpus
  tool          → (alias for tool_agentic) anchor field in tool output JSON
  tool_agentic  → anchor in tool-calling format (role:"tool" messages)
  tool_read     → anchor in plain structured JSON blocks (no tool-calling format)
"""

from .external import render_external
from .history import render_history
from .icl import render_icl
from .icl_dist import render_icl_dist
from .rag import render_rag, build_full_corpus
from .tool_agentic import render_tool
from .tool_read import render_tool_read

SUITE_RENDERERS = {
    "external": render_external,
    "history": render_history,
    "icl": render_icl,
    "icl_dist": render_icl_dist,
    "rag": render_rag,
    "tool": render_tool,
    "tool_agentic": render_tool,
    "tool_read": render_tool_read,
}
