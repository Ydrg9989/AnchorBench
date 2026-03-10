"""Suite renderers — each transforms an ItemSpec into PromptView records."""

from .external import render_external, render_external_v2
from .icl import render_icl, render_icl_legacy, render_icl_v2
from .history import render_history, render_history_v2
from .rag import render_rag
from .rag_v2 import render_rag_v2
from .tools import render_tool
from .tool_v2 import render_tool_v2

SUITE_RENDERERS = {
    "external": render_external,
    "external_v2": render_external_v2,
    "icl": render_icl_legacy,
    "icl_v2": render_icl_v2,
    "history": render_history,
    "history_v2": render_history_v2,
    "rag": render_rag,
    "rag_v2": render_rag_v2,
    "tool": render_tool,
    "tool_v2": render_tool_v2,
}
