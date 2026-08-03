"""Smoke test: every public submodule of the anchorbench package imports.

This catches regressions when modules are moved or renamed (e.g. during
the v2.0 consolidation of anchorbench_v1, anchorbench_eval, and
mitigation_eval into a single anchorbench/ package).
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
    "anchorbench.inference.async_api",
    "anchorbench.runners",
    "anchorbench.runners.base",
    "anchorbench.runners.external",
    "anchorbench.runners.history",
    "anchorbench.runners.icl",
    "anchorbench.runners.rag",
    "anchorbench.runners.tool",
    "anchorbench.runners.api",
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


@pytest.mark.parametrize("mod", PUBLIC_MODULES)
def test_module_imports(mod: str) -> None:
    importlib.import_module(mod)


def test_version_attr() -> None:
    import anchorbench

    assert isinstance(anchorbench.__version__, str)
    assert anchorbench.__version__.count(".") >= 1


def test_deprecation_shims_redirect() -> None:
    """Old package names still import and point at the new modules."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import anchorbench_v1
        import anchorbench_eval
        import mitigation_eval

    import anchorbench.data
    import anchorbench.eval
    import anchorbench.inference

    assert anchorbench_v1.__path__ == anchorbench.data.__path__
    assert anchorbench_eval.__path__ == anchorbench.eval.__path__
    assert mitigation_eval.__path__ == anchorbench.inference.__path__
