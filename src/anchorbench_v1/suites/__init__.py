"""Suite renderers — each transforms an ItemSpec into PromptView records."""

from .external import render_external
from .icl import render_icl
from .history import render_history
from .rag import render_rag
from .tools import render_tool

SUITE_RENDERERS = {
    "external": render_external,
    "icl": render_icl,
    "history": render_history,
    "rag": render_rag,
    "tool": render_tool,
}
