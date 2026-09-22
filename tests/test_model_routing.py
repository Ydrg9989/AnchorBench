"""Open-weight models must not be dispatched to the hosted-API runner.

Routing used to prefix-match hf_ids against ("openai/", "anthropic/",
"google/", "x-ai/"). That swept up Gemma: google/gemma-3-1b-it and
google/gemma-3-4b-it are open-weight and declare backend: vllm, but share a
namespace with google/gemini-2.5-flash. They were sent to OpenRouter, which
answered 429 and produced 1800 records with parse_rate 0.0 and every metric
null -- a summary that looks complete and contains no data.

The fix first landed in ``anchorbench experiment`` only; ``anchorbench eval``
kept its own copy of the rule and still mis-routed Gemma. Both commands now
go through :mod:`anchorbench.cli.cells`, and this file checks the shared rule
*and* each command's use of it.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

from anchorbench.cli import eval as eval_cli
from anchorbench.cli.cells import ROOT, build_cell_cmd, is_api_model, runner_module

EXTERNAL = {
    "suite": "external",
    "dataset_dir": "datasets/anchorbench_external_core",
    "promptviews_file": "promptviews_core.jsonl",
    "itemspecs_file": "itemspecs.jsonl",
}
GREEDY = {"max_tokens": 512}


def _models():
    for path in sorted(glob.glob(str(ROOT / "conf" / "model" / "*.yaml"))):
        with open(path) as fh:
            yield Path(path).stem, yaml.safe_load(fh)


def test_routing_follows_declared_backend():
    wrong = [
        f"{m['short']} (backend={m.get('backend')}, hf_id={m['hf_id']}) -> "
        f"{'API' if is_api_model(m) else 'local'}"
        for _, m in _models()
        if is_api_model(m) != (m.get("backend") == "openrouter")
    ]
    assert not wrong, "models routed against their declared backend:\n  " + "\n  ".join(wrong)


def test_gemma_is_local_despite_the_google_namespace():
    """The specific collision that caused the failure, pinned by name."""
    for _, m in _models():
        if m["hf_id"].startswith("google/gemma"):
            assert not is_api_model(m), f"{m['hf_id']} must run locally, not via the API"


def test_build_cell_cmd_picks_the_runner_from_the_backend(tmp_path):
    for _, m in _models():
        cmd = build_cell_cmd(m, EXTERNAL, GREEDY, tmp_path)
        module = cmd[cmd.index("-m") + 1]
        assert module == runner_module(m, EXTERNAL)
        if m.get("backend") == "openrouter":
            assert module == "anchorbench.runners.api"
        else:
            assert module == "anchorbench.runners.external"
            assert "--promptviews" in cmd


def test_icl_dist_reuses_the_icl_runner(tmp_path):
    data = dict(EXTERNAL, suite="icl_dist", variant="icl",
                dataset_dir="datasets/anchorbench_icl_dist_core")
    qwen = {"hf_id": "Qwen/Qwen2.5-7B-Instruct", "backend": "vllm"}
    assert runner_module(qwen, data) == "anchorbench.runners.icl"


@pytest.mark.parametrize("model_name", ["gemma_4b", "gemma_1b"])
def test_eval_command_routes_gemma_locally(model_name, tmp_path):
    """Compose the real Hydra config exactly as ``anchorbench eval`` does."""
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"model={model_name}", "data=external", f"out_dir={tmp_path}"],
        )
    cmd = eval_cli._build_cmd(cfg)
    assert "anchorbench.runners.external" in cmd
    assert "anchorbench.runners.api" not in cmd


def test_eval_command_routes_api_models_to_the_api_runner(tmp_path):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["model=gemini_2_5_flash", "data=external", f"out_dir={tmp_path}"],
        )
    cmd = eval_cli._build_cmd(cfg)
    assert "anchorbench.runners.api" in cmd
