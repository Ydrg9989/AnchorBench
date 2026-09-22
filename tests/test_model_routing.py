"""Every cell the CLI builds must reach the right runner with flags it accepts.

Routing used to prefix-match hf_ids against ("openai/", "anthropic/",
"google/", "x-ai/"). That swept up Gemma: google/gemma-3-1b-it and
google/gemma-3-4b-it are open-weight and declare backend: vllm, but share a
namespace with google/gemini-2.5-flash. They were sent to OpenRouter, which
answered 429 and produced 1800 records with parse_rate 0.0 and every metric
null -- a summary that looks complete and contains no data.

The fix first landed in ``anchorbench experiment`` only; ``anchorbench eval``
kept its own copy of the rule and still mis-routed Gemma. Both commands now
go through :mod:`anchorbench.cli.cells`. This file checks the shared rule,
each command's use of it on the real Hydra configs, and that every flag the
builder emits is one the target runner declares (the API branch used to emit
``--suite`` and ``--variant``, which ``runners.api`` does not define).
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

from anchorbench.cli import eval as eval_cli
from anchorbench.cli import experiment as experiment_cli
from anchorbench.cli.cells import API_RUNNER, ROOT, build_cell_cmd, is_api_model, runner_module

GREEDY = {"max_tokens": 512}
RUNNERS_DIR = ROOT / "src" / "anchorbench" / "runners"
COMMON_ARGS = ROOT / "src" / "anchorbench" / "eval" / "runner_utils.py"
_FLAG = re.compile(r'add_argument\(\s*"(--[a-z_]+)"')


def _models():
    for path in sorted(glob.glob(str(ROOT / "conf" / "model" / "*.yaml"))):
        with open(path) as fh:
            yield Path(path).stem, yaml.safe_load(fh)


def _data(name: str) -> dict:
    with open(ROOT / "conf" / "data" / f"{name}.yaml") as fh:
        return yaml.safe_load(fh)


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def _declared_flags(module: str) -> set[str]:
    """Flags the runner module's argparse accepts, read from its source."""
    name = module.rsplit(".", 1)[1]
    sources = [RUNNERS_DIR / f"{name}.py"]
    if name in ("external", "icl", "rag"):
        sources.append(RUNNERS_DIR / "_single_stage.py")
    if name != "api":
        sources.append(COMMON_ARGS)          # add_common_args
    flags: set[str] = set()
    for src in sources:
        flags.update(_FLAG.findall(src.read_text()))
    return flags


def _emitted_flags(cmd: list[str]) -> set[str]:
    return {tok for tok in cmd if tok.startswith("--")}


# --- the shared rule ---------------------------------------------------------

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


@pytest.mark.parametrize("suite", ["external", "history", "icl", "icl_dist", "rag", "tool"])
def test_local_runner_is_named_after_the_suite(suite):
    qwen = {"hf_id": "Qwen/Qwen2.5-7B-Instruct", "backend": "vllm"}
    data = _data(suite)
    # icl_dist keeps suite: icl and only swaps the dataset, so it reaches the icl runner.
    assert runner_module(qwen, data) == f"anchorbench.runners.{data['suite']}"


@pytest.mark.parametrize("suite", ["external", "history", "icl", "icl_dist", "rag", "tool"])
def test_every_emitted_flag_is_declared_by_the_target_runner(suite, tmp_path):
    data = _data(suite)
    for _, m in _models():
        cmd = build_cell_cmd(m, data, GREEDY, tmp_path, batch_size=32, seed=42,
                             baseline_condition="control", tool_plaintext=True)
        module = cmd[cmd.index("-m") + 1]
        unknown = _emitted_flags(cmd) - _declared_flags(module)
        assert not unknown, f"{m['short']} / {suite}: {module} does not accept {sorted(unknown)}"


def test_api_cells_pass_the_variant_as_a_suite_name(tmp_path):
    gemini = {"hf_id": "google/gemini-2.5-flash", "backend": "openrouter"}
    cmd = build_cell_cmd(gemini, _data("icl_dist"), GREEDY, tmp_path)
    assert cmd[cmd.index("-m") + 1] == API_RUNNER
    assert cmd[cmd.index("--suites") + 1] == "icl_dist"
    assert "--variant" not in cmd and "--suite" not in cmd


# --- anchorbench eval -----------------------------------------------------------

@pytest.mark.parametrize("model_name", ["gemma_4b", "gemma_1b"])
def test_eval_command_routes_gemma_locally(model_name, tmp_path):
    cfg = _compose(f"model={model_name}", "data=external", f"out_dir={tmp_path}")
    cmd = eval_cli._build_cmd(cfg)
    assert "anchorbench.runners.external" in cmd
    assert API_RUNNER not in cmd


def test_eval_command_routes_api_models_to_the_api_runner(tmp_path):
    cfg = _compose("model=gemini_2_5_flash", "data=external", f"out_dir={tmp_path}")
    assert API_RUNNER in eval_cli._build_cmd(cfg)


# --- anchorbench experiment ------------------------------------------------------

def test_paper_main_recipe_routes_every_cell_by_backend():
    cfg = _compose("+experiment=paper_main")
    cells = experiment_cli._resolve_cells(cfg)
    assert len(cells) == 70, "14 models x 5 suites"
    for model, data, _gpu, out_dir in cells:
        cmd = build_cell_cmd(model, data, GREEDY, out_dir,
                             baseline_condition=cfg.get("baseline_condition"))
        module = cmd[cmd.index("-m") + 1]
        if model.get("backend") == "openrouter":
            assert module == API_RUNNER, f"{model['short']} / {data['suite']}"
        else:
            assert module == f"anchorbench.runners.{data['suite']}", f"{model['short']} / {data['suite']}"
            assert not str(out_dir).endswith("api_benchmark")
    gemma_cells = [(m, d) for m, d, _, _ in cells if m["hf_id"].startswith("google/gemma")]
    assert len(gemma_cells) == 10 and all(not is_api_model(m) for m, _ in gemma_cells)
