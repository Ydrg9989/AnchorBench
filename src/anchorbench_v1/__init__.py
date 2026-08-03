"""Deprecation shim: ``anchorbench_v1`` was renamed to :mod:`anchorbench.data`.

This module re-exports the new package and emits a ``DeprecationWarning``
on import. It will be removed in AnchorBench v2.1. Update your imports
to ``from anchorbench.data import ...``.
"""

from __future__ import annotations

import warnings

import anchorbench.data as _new

warnings.warn(
    "anchorbench_v1 has been renamed to anchorbench.data; the alias will "
    "be removed in v2.1. Update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__path__ = _new.__path__  # type: ignore[attr-defined]
__all__ = list(getattr(_new, "__all__", []))
