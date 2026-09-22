"""The registry must reproduce, exactly, the literals it replaced.

Every list below is the value the corresponding module carried before the
registry existed. Table order, slugs, short names and LaTeX macros all feed
generated tables that are pinned by golden hashes, so any drift here is a
drift in a paper artifact.
"""

from __future__ import annotations

import yaml

from anchorbench import registry
from anchorbench.eval import constants
from anchorbench.paper import _common, tables_appendix, verify
from anchorbench.paths import CONF_DIR

OW_ORDER = ["Qwen-1.5B", "Qwen-3B", "Qwen-7B", "Llama-1B", "Llama-3B", "Llama-8B",
            "Gemma-1B", "Gemma-4B", "OLMo-13B", "OLMo-32B"]
API_ORDER = ["GPT-5.4-mini", "Claude-H4.5", "Gemini-2.5-Flash", "Grok-3-mini"]
OW_SLUGS = ["Qwen_Qwen2.5-1.5B-Instruct", "Qwen_Qwen2.5-3B-Instruct", "Qwen_Qwen2.5-7B-Instruct",
            "meta-llama_Llama-3.2-1B-Instruct", "meta-llama_Llama-3.2-3B-Instruct",
            "meta-llama_Llama-3.1-8B-Instruct", "google_gemma-3-1b-it", "google_gemma-3-4b-it",
            "allenai_OLMo-2-1124-13B-Instruct", "allenai_OLMo-2-0325-32B-Instruct"]
API_SLUGS = ["openai_gpt-5.4-mini", "anthropic_claude-haiku-4.5", "google_gemini-2.5-flash",
             "x-ai_grok-3-mini-beta"]
OW_IDS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
          "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-1b-it", "google/gemma-3-4b-it",
          "allenai/OLMo-2-1124-13B-Instruct", "allenai/OLMo-2-0325-32B-Instruct"]
API_IDS = ["openai/gpt-5.4-mini", "anthropic/claude-haiku-4.5", "google/gemini-2.5-flash",
           "x-ai/grok-3-mini-beta"]
MODEL_LATEX = {
    "Qwen-1.5B": r"\qwenonefive", "Qwen-3B": r"\qwenthree", "Qwen-7B": r"\qwenseven",
    "Llama-1B": r"\llamaone", "Llama-3B": r"\llamathree", "Llama-8B": r"\llamaeight",
    "Gemma-1B": r"\gemmaone", "Gemma-4B": r"\gemmafour",
    "OLMo-13B": r"\olmothirteen", "OLMo-32B": r"\olmonthirtytwo",
    "GPT-5.4-mini": r"\gptfivefourmini", "Claude-H4.5": r"\claudehaiku",
    "Gemini-2.5-Flash": r"\geminiflash", "Grok-3-mini": r"\grokthreemini",
}
PARAMS = dict(zip(OW_IDS, ["1.5B", "3B", "7B", "1B", "3B", "8B", "1B", "4B", "13B", "32B"]))
FAMILY = dict(zip(OW_IDS + API_IDS, [r"\qwen"] * 3 + [r"\llama"] * 3 + [r"\gemma"] * 2
                  + [r"\aitwo"] * 2 + [r"\openai", r"\claude", r"\gemini", r"\grok"]))
SUITE_LATEX = {"External": r"\externalsuite", "History": r"\historysuite", "Icl": r"\iclsuite",
               "Rag": r"\ragsuite", "Tool": r"\toolsuite"}


def test_panel_is_the_published_fourteen_in_table_order():
    assert [m.short for m in registry.open_weight_models()] == OW_ORDER
    assert [m.short for m in registry.api_models()] == API_ORDER
    assert [m.slug for m in registry.models()] == OW_SLUGS + API_SLUGS
    assert [m.hf_id for m in registry.models()] == OW_IDS + API_IDS
    assert all(not m.is_api for m in registry.open_weight_models())
    assert all(m.is_api for m in registry.api_models())


def test_every_model_config_loads_and_slugs_are_unique():
    keys = registry.all_model_keys()
    loaded = [registry.load_model(k) for k in keys]
    assert len({m.slug for m in loaded}) == len(loaded)
    assert set(m.key for m in registry.models()) <= set(keys)
    for m in loaded:
        assert m.slug == m.hf_id.replace("/", "_"), m.key


def test_metadata_lives_in_the_yaml_files():
    for m in registry.models():
        raw = yaml.safe_load(open(CONF_DIR / "model" / f"{m.key}.yaml"))
        assert raw["latex"] == MODEL_LATEX[m.short]
        assert raw["family_latex"] == FAMILY[m.hf_id]


def test_suites_in_paper_order_with_their_spellings():
    s = registry.suites()
    assert [x.key for x in s] == ["external", "history", "icl", "rag", "tool"]
    assert [x.unified_key for x in s] == ["External", "History", "Icl", "Rag", "Tool"]
    assert [x.label for x in s] == ["External", "History", "ICL", "RAG", "Tool"]
    assert {x.unified_key: x.latex for x in s} == SUITE_LATEX
    (v,) = registry.variants()
    assert v.key == "icl_dist" and v.suite == "icl" and v.variant == "icl_dist"
    assert registry.RESULTS_DISCOVERY_ORDER == ("external", "icl", "rag", "tool", "history")


def test_constants_are_the_old_literals():
    assert constants.MODEL_SHORT == dict(zip(OW_SLUGS + API_SLUGS, OW_ORDER + API_ORDER))
    assert constants.API_MODEL_IDS == API_IDS and constants.OW_MODEL_IDS == OW_IDS
    assert constants.SUITES == ("external", "icl", "rag", "tool", "history")
    assert constants.SUITE_DATASETS == {
        "external": "datasets/anchorbench_external_core",
        "history": "datasets/anchorbench_history_core",
        "icl": "datasets/anchorbench_icl_core",
        "rag": "datasets/anchorbench_rag_core",
        "tool": "datasets/anchorbench_tool_core",
    }
    assert constants.VARIANT_DATASETS == {"icl_dist": "datasets/anchorbench_icl_dist_core"}


def test_paper_helpers_are_the_old_literals():
    assert _common.OW_MODELS_ORDER == OW_ORDER and _common.API_MODELS_ORDER == API_ORDER
    assert _common.ALL_MODELS_ORDER == OW_ORDER + API_ORDER
    assert _common.OW_SLUGS == OW_SLUGS and _common.API_SLUGS == API_SLUGS
    assert _common.SUITES == ["External", "History", "Icl", "Rag", "Tool"]
    assert _common.SUITE_LATEX == SUITE_LATEX and _common.MODEL_LATEX == MODEL_LATEX
    assert _common.slug_to_short("Qwen_Qwen2.5-7B-Instruct") == "Qwen-7B"
    assert _common.slug_to_short("unknown_slug") == "unknown_slug"
    assert tables_appendix.PARAM_BY_SLUG == PARAMS and tables_appendix.FAMILY_MACRO == FAMILY
    assert verify.MODEL_ORDER == list(zip(OW_ORDER + API_ORDER, OW_SLUGS + API_SLUGS))
    assert verify.SUITE_NAME_MAP["ICL"] == "Icl" and verify.SUITE_NAME_MAP["RAG"] == "Rag"
    assert verify.SUITES == ["External", "History", "ICL", "RAG", "Tool"]
