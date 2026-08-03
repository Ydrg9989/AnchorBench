"""Inference helpers for AnchorBench.

Provides a thin async wrapper around OpenRouter's chat-completions API
used by the API benchmark runner and the mitigation runners.

Submodules:
    async_api -- ``AsyncOpenRouterClient`` with rate limiting + retries
"""

from __future__ import annotations
