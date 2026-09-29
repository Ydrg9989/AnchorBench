"""`anchorbench add-model` writes the backend it is told, never one guessed
from the identifier prefix: google/gemma-* runs locally and google/gemini-*
on OpenRouter, and cells.is_api_model routes by the declared backend alone.
"""

from __future__ import annotations

import sys

import yaml

from anchorbench.cli import add_model
from anchorbench.cli.cells import is_api_model


def _run(monkeypatch, tmp_path, *argv: str) -> dict:
    monkeypatch.setattr(add_model, "CONF_MODEL_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["anchorbench add-model", *argv])
    assert add_model.main() == 0
    files = list(tmp_path.glob("*.yaml"))
    assert len(files) == 1, files
    return yaml.safe_load(files[0].read_text())


def test_gemma_defaults_to_a_local_backend(monkeypatch, tmp_path):
    cfg = _run(monkeypatch, tmp_path, "google/gemma-3-12b-it")
    assert cfg["backend"] == "vllm"
    assert not is_api_model(cfg)
    assert cfg["hf_id"] == "google/gemma-3-12b-it"


def test_openrouter_backend_is_declared_explicitly(monkeypatch, tmp_path):
    cfg = _run(monkeypatch, tmp_path, "google/gemini-2.5-pro", "--backend", "openrouter")
    assert cfg["backend"] == "openrouter"
    assert is_api_model(cfg)
    assert "tensor_parallel_size" not in cfg


def test_hf_backend_keeps_the_local_settings(monkeypatch, tmp_path):
    cfg = _run(monkeypatch, tmp_path, "meta-llama/Llama-4-8B-Instruct", "--backend", "hf")
    assert cfg["backend"] == "hf"
    assert cfg["dtype"] == "bfloat16"


def test_an_existing_config_is_not_overwritten(monkeypatch, tmp_path):
    (tmp_path / "gemma_3_12b_it.yaml").write_text("name: keep me\n")
    monkeypatch.setattr(add_model, "CONF_MODEL_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["anchorbench add-model", "google/gemma-3-12b-it"])
    assert add_model.main() == 0
    assert (tmp_path / "gemma_3_12b_it.yaml").read_text() == "name: keep me\n"
