"""Deprecation shim: see :mod:`anchorbench.data.suites`."""

from __future__ import annotations

from anchorbench.data.suites import (  # noqa: F401
    SUITE_RENDERERS,
    build_full_corpus,
    render_external,
    render_history,
    render_icl,
    render_icl_dist,
    render_rag,
    render_tool,
)
