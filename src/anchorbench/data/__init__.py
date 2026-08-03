"""Dataset generation for AnchorBench.

Submodules:
    schema, domains, itemspec_gen   -- ItemSpec construction
    suites/                          -- per-suite renderers (ItemSpec -> PromptView)
    generate, validate, validators   -- end-to-end pipeline
    llm_enhance, openrouter_client   -- optional LLM-augmented item generation
"""

from __future__ import annotations

from anchorbench import __version__

__all__ = ["__version__"]
