"""Regression tests for appendix table builders that need no results/ tree.

`build_model_details` used to read a `configs/benchmark.yaml` that was deleted
in the v2.0 refactor, via an unguarded `open()`. That crashed
`anchorbench tables --paper` (and so `scripts/reproduce_paper.sh` phases 4-5)
without any test noticing, because every other builder in the module needs the
multi-GB results/ tree and therefore is not exercised in CI.

These tests deliberately depend on nothing but the installed package.
"""

from __future__ import annotations

from anchorbench.eval.constants import API_MODEL_IDS, OW_MODEL_IDS
from anchorbench.paper.tables_appendix import (
    FAMILY_MACRO,
    PARAM_BY_SLUG,
    build_model_details,
)


def test_build_model_details_needs_no_config_file() -> None:
    """The default call takes no arguments and touches no filesystem path."""
    tex = build_model_details()
    assert tex.startswith("%")
    assert "\\begin{table}" in tex and "\\bottomrule" in tex
    assert "\\label{tab:model-details}" in tex


def test_build_model_details_lists_the_full_panel() -> None:
    tex = build_model_details()
    for slug in OW_MODEL_IDS:
        assert slug in tex, f"open-weight model missing from table: {slug}"
    for slug in API_MODEL_IDS:
        assert slug in tex, f"API model missing from table: {slug}"
    assert tex.count("\\path{") == len(OW_MODEL_IDS) + len(API_MODEL_IDS) == 14


def test_open_weight_and_api_rows_are_labelled() -> None:
    tex = build_model_details()
    assert tex.count("& Open-weight & Local") == len(OW_MODEL_IDS)
    assert tex.count("& API & OpenRouter") == len(API_MODEL_IDS)


def test_param_and_family_maps_cover_the_panel() -> None:
    """Guard against a model being added to constants but not to the maps.

    A missing entry silently renders as an empty family macro or a '---'
    parameter count rather than failing, so it has to be asserted.
    """
    for slug in OW_MODEL_IDS:
        assert slug in PARAM_BY_SLUG, f"no parameter count for {slug}"
    for slug in OW_MODEL_IDS + API_MODEL_IDS:
        assert slug in FAMILY_MACRO, f"no LaTeX family macro for {slug}"


def test_explicit_panel_overrides_the_default() -> None:
    tex = build_model_details(ow=["Qwen/Qwen2.5-7B-Instruct"], api=[])
    assert tex.count("\\path{") == 1
    assert "Qwen/Qwen2.5-7B-Instruct" in tex
    assert "allenai/OLMo-2-1124-13B-Instruct" not in tex
