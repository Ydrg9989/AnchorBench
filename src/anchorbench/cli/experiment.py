"""Hydra-driven `anchorbench experiment` subcommand.

Composes ``conf/experiment/*.yaml`` (which references ``data``, ``decoding``,
``model``, and ``tier`` configs) and dispatches to a sequence of per-cell
runs. Each cell reuses the same logic as ``anchorbench eval``.

Usage::

    anchorbench experiment +experiment=paper_main
    anchorbench experiment +experiment=paper_sampling --dry_run
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from anchorbench.eval.constants import API_MODEL_IDS

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = ROOT / "conf"


def _load_yaml(rel: str) -> dict:
    import yaml
    with open(CONF_DIR / rel) as f:
        return yaml.safe_load(f) or {}


def _resolve_model(name: str) -> dict:
    return _load_yaml(f"model/{name}.yaml")


def _resolve_data(name: str) -> dict:
    return _load_yaml(f"data/{name}.yaml")


def _resolve_tier(name: str) -> dict:
    return _load_yaml(f"tier/{name}.yaml")


def _is_api(hf_id: str) -> bool:
    return hf_id in set(API_MODEL_IDS) or any(
        hf_id.startswith(p) for p in ("openai/", "anthropic/", "google/", "x-ai/")
    )


def _build_cell_cmd(model: dict, data: dict, decoding: dict,
                    out_dir: Path, baseline_condition: str | None,
                    tool_plaintext: bool = False) -> list[str]:
    suite = data["suite"]
    variant = data.get("variant")
    dataset_dir = ROOT / data["dataset_dir"]
    pv = dataset_dir / data["promptviews_file"]
    isp = dataset_dir / data["itemspecs_file"]

    if _is_api(model["hf_id"]):
        cmd = [
            sys.executable, "-m", "anchorbench.runners.api",
            "--model_id", model["hf_id"],
            "--suite", suite,
            "--out_dir", str(out_dir),
            "--max_tokens", str(decoding["max_tokens"]),
        ]
        if variant:
            cmd += ["--variant", variant]
        return cmd

    runner_suite = "icl" if (variant or suite) == "icl" else suite
    cmd = [
        sys.executable, "-m", f"anchorbench.runners.{runner_suite}",
        "--model_id", model["hf_id"],
        "--promptviews", str(pv),
        "--itemspecs", str(isp),
        "--out_dir", str(out_dir),
        "--backend", model.get("backend", "vllm"),
        "--max_tokens", str(decoding["max_tokens"]),
        "--gpu_memory_utilization", str(model.get("gpu_memory_utilization", 0.9)),
        "--max_model_len", str(model.get("max_model_len", 4096)),
        "--tensor_parallel_size", str(model.get("tensor_parallel_size", 1)),
        "--dtype", model.get("dtype", "bfloat16"),
    ]
    if suite == "history" and baseline_condition:
        cmd += ["--baseline_condition", baseline_condition]
    if suite == "tool" and tool_plaintext:
        # Forces the plaintext rendering for models that would otherwise
        # get native structured tool messages, which is what makes the
        # within-model comparison in tab:tool_plaintext possible.
        cmd += ["--tool_plaintext"]
    return cmd


def _resolve_cells(cfg: DictConfig) -> list[tuple[dict, dict, str | None, Path]]:
    """Return list of (model_dict, data_dict, gpu_ids, out_dir) cells."""
    suites: list[str] = list(cfg.get("suites", []))
    base_out = ROOT / cfg.get("out_dir", "results/experiment")
    api_out = ROOT / cfg.get("api_out_dir", str(base_out))

    cells: list[tuple[dict, dict, str | None, Path]] = []

    explicit_models = cfg.get("models")
    tiers: list[str] = list(cfg.get("tiers", []))

    suite_data = {s: _resolve_data(s) for s in suites}

    if explicit_models:
        for model_name in explicit_models:
            model = _resolve_model(model_name)
            for suite in suites:
                data = suite_data[suite]
                out = (api_out if _is_api(model["hf_id"]) else base_out / suite)
                cells.append((model, data, None, out))
        return cells

    for tier_name in tiers:
        tier = _resolve_tier(tier_name)
        gpu_slots = [str(g) for g in (tier.get("gpu_ids") or [None])]
        for i, model_name in enumerate(tier.get("models", [])):
            model = _resolve_model(model_name)
            gpu = gpu_slots[i % len(gpu_slots)] if gpu_slots else None
            for suite in suites:
                data = suite_data[suite]
                out = (api_out if _is_api(model["hf_id"]) else base_out / suite)
                cells.append((model, data, gpu, out))
    return cells


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def _hydra_main(cfg: DictConfig) -> int:
    log.info("Resolved experiment:\n%s", OmegaConf.to_yaml(cfg))
    cells = _resolve_cells(cfg)
    log.info("Plan: %d cells", len(cells))
    failures = 0
    decoding = OmegaConf.to_container(cfg.decoding, resolve=True)
    baseline = cfg.get("baseline_condition")
    tool_plaintext = bool(cfg.get("tool_plaintext", False))
    for i, (model, data, gpu, out_dir) in enumerate(cells, 1):
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_cell_cmd(model, data, decoding, out_dir, baseline,
                              tool_plaintext)
        tag = f"[{i}/{len(cells)}] {model['short']} / {data['suite']}"
        if cfg.get("dry_run"):
            print(tag, " ".join(cmd))
            continue
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        log.info("%s -> %s", tag, out_dir)
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            log.error("%s FAILED (rc=%d)", tag, rc)
            failures += 1
    return 1 if failures else 0


def main() -> int:
    return _hydra_main()


if __name__ == "__main__":
    sys.exit(main())
