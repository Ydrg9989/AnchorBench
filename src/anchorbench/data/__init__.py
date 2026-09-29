"""Dataset generation for AnchorBench.

Submodules:
    schema, domains, itemspec_gen   -- ItemSpec construction
    suites/                          -- per-suite renderers (ItemSpec -> PromptView)
    generate, validate, validators   -- end-to-end pipeline
"""

from __future__ import annotations

from anchorbench import __version__

__all__ = ["__version__"]
