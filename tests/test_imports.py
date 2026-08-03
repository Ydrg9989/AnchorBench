"""Smoke test: every public submodule of the anchorbench package imports.

This catches regressions when modules are moved or renamed (e.g. during
the v2.0 consolidation of three separate packages into a single
anchorbench/ package). See RELEASE_NOTES.md for the import migration map.
"""

from __future__ import annotations

import importlib

import pytest

PUBLIC_MODULES = [
    "anchorbench",
    "anchorbench.data",
    "anchorbench.data.domains",
    "anchorbench.data.generate",
    "anchorbench.data.itemspec_gen",
    "anchorbench.data.schema",
    "anchorbench.data.validators",
    "anchorbench.data.suites",
    "anchorbench.data.suites.external",
    "anchorbench.data.suites.history",
    "anchorbench.data.suites.icl",
    "anchorbench.data.suites.icl_dist",
    "anchorbench.data.suites.rag",
    "anchorbench.data.suites.tool",
    "anchorbench.eval",
    "anchorbench.eval.backends",
    "anchorbench.eval.constants",
    "anchorbench.eval.evaluator",
    "anchorbench.eval.io",
    "anchorbench.eval.metrics",
    "anchorbench.eval.parsing",
    "anchorbench.eval.runner_utils",
    "anchorbench.inference",
    "anchorbench.runners",
    "anchorbench.runners.base",
    "anchorbench.runners.external",
    "anchorbench.runners.history",
    "anchorbench.runners.icl",
    "anchorbench.runners.rag",
    "anchorbench.runners.tool",
    "anchorbench.analysis",
    "anchorbench.analysis.unified",
    "anchorbench.paper",
    "anchorbench.paper._common",
    "anchorbench.paper.tables_main",
    "anchorbench.paper.tables_appendix",
    "anchorbench.paper.fig4_dose_response",
    "anchorbench.paper.fig5_acc_vs_disc",
    "anchorbench.paper.verify",
    "anchorbench.cli",
    "anchorbench.cli.main",
    "anchorbench.cli.eval",
    "anchorbench.cli.experiment",
    "anchorbench.cli.tables",
    "anchorbench.cli.generate",
    "anchorbench.cli.add_model",
]


# Modules that need an optional dependency group. Importing them on a bare
# `pip install anchorbench` raises ModuleNotFoundError, so they are checked
# separately and skipped when the extra is absent.
OPTIONAL_MODULES = [
    ("anchorbench.inference.async_api", "aiohttp", "api"),
    ("anchorbench.runners.api", "aiohttp", "api"),
]


@pytest.mark.parametrize("mod", PUBLIC_MODULES)
def test_module_imports(mod: str) -> None:
    importlib.import_module(mod)


@pytest.mark.parametrize("mod,dep,extra", OPTIONAL_MODULES)
def test_optional_module_imports(mod: str, dep: str, extra: str) -> None:
    pytest.importorskip(dep, reason=f"{mod} needs the '{extra}' extra")
    importlib.import_module(mod)


def test_version_attr() -> None:
    import anchorbench

    assert isinstance(anchorbench.__version__, str)
    assert anchorbench.__version__.count(".") >= 1
