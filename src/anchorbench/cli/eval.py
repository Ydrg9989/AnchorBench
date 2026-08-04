"""Hydra-driven `anchorbench eval` subcommand.

Composes ``conf/data/*.yaml`` + ``conf/model/*.yaml`` + ``conf/decoding/*.yaml``
and dispatches to the per-suite runner inside ``anchorbench.runners``.

Usage::

    anchorbench eval data=external model=qwen_7b
    anchorbench eval data=icl_dist model=llama_8b decoding=greedy
    anchorbench eval data=external model=qwen_7b model.batch_size=128
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


def _is_api_model(hf_id: str) -> bool:
    return hf_id in set(API_MODEL_IDS) or "/" in hf_id and any(
        hf_id.startswith(p) for p in ("openai/", "anthropic/", "google/", "x-ai/")
    )


def _build_cmd(cfg: DictConfig) -> list[str]:
    data = cfg.data
    model = cfg.model
    decoding = cfg.decoding
    suite = data.suite
    variant = data.get("variant")
    dataset_dir = ROOT / data.dataset_dir
    pv = dataset_dir / data.promptviews_file
    isp = dataset_dir / data.itemspecs_file

    # cfg.out_dir is relative, and hydra's version_base=None turns on
    # job.chdir, so a bare Path() resolved under outputs/hydra/<date>/<time>/
    # instead of results/. cli/experiment.py already anchors on ROOT.
    out_dir = ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if _is_api_model(model.hf_id):
        runner_module = "anchorbench.runners.api"
        cmd = [
            sys.executable, "-m", runner_module,
            "--model_id", model.hf_id,
            "--suite", suite,
            "--out_dir", str(out_dir),
            "--max_tokens", str(decoding.max_tokens),
        ]
        if variant:
            cmd += ["--variant", variant]
        return cmd

    runner_module = f"anchorbench.runners.{'icl' if (variant or suite) == 'icl' else suite}"
    cmd = [
        sys.executable, "-m", runner_module,
        "--model_id", model.hf_id,
        "--promptviews", str(pv),
        "--itemspecs", str(isp),
        "--out_dir", str(out_dir),
        "--backend", model.get("backend", "vllm"),
        "--batch_size", str(cfg.batch_size),
        "--max_tokens", str(decoding.max_tokens),
        "--seed", str(cfg.seed),
        "--gpu_memory_utilization", str(model.get("gpu_memory_utilization", 0.9)),
        "--max_model_len", str(model.get("max_model_len", 4096)),
        "--tensor_parallel_size", str(model.get("tensor_parallel_size", 1)),
        "--dtype", model.get("dtype", "bfloat16"),
    ]
    if suite == "history" and cfg.get("baseline_condition"):
        cmd += ["--baseline_condition", cfg.baseline_condition]
    return cmd


@hydra.main(version_base=None, config_path=str(ROOT / "conf"), config_name="config")
def _hydra_main(cfg: DictConfig) -> int:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    cmd = _build_cmd(cfg)
    log.info("Launching: %s", " ".join(cmd))
    if cfg.get("dry_run"):
        return 0
    env = os.environ.copy()
    if "gpu_ids" in cfg and cfg.gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_ids)
    return subprocess.run(cmd, env=env).returncode


def main() -> int:
    return _hydra_main()


if __name__ == "__main__":
    sys.exit(main())
