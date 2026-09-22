"""Hydra-driven `anchorbench eval` subcommand.

Composes ``conf/data/*.yaml`` + ``conf/model/*.yaml`` + ``conf/decoding/*.yaml``
and dispatches to the per-suite runner inside ``anchorbench.runners``. The
command line for the runner is built by :mod:`anchorbench.cli.cells`, which
``anchorbench experiment`` shares, so both commands route a model the same way.

Usage::

    anchorbench eval data=external model=qwen_7b
    anchorbench eval data=icl_dist model=llama_8b decoding=greedy
    anchorbench eval data=external model=qwen_7b batch_size=128 decoding.max_tokens=256
    anchorbench eval data=tool model=qwen_7b +tool_plaintext=true
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from anchorbench.cli.cells import build_cell_cmd
from anchorbench.paths import CONF_DIR, ROOT

log = logging.getLogger(__name__)


def _out_dir(cfg: DictConfig) -> Path:
    # cfg.out_dir is relative, and hydra's version_base=None turns on
    # job.chdir, so a bare Path() would resolve under outputs/hydra/<date>/
    # <time>/ instead of results/. Anchor it on the repository root.
    return ROOT / cfg.out_dir


def _build_cmd(cfg: DictConfig) -> list[str]:
    """Translate the composed Hydra config into a runner command."""
    plain = OmegaConf.to_container(cfg, resolve=True)
    return build_cell_cmd(
        plain["model"],
        plain["data"],
        plain["decoding"],
        _out_dir(cfg),
        batch_size=plain.get("batch_size"),
        seed=plain.get("seed"),
        baseline_condition=plain.get("baseline_condition"),
        tool_plaintext=bool(plain.get("tool_plaintext", False)),
    )


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def _hydra_main(cfg: DictConfig) -> int:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    cmd = _build_cmd(cfg)
    log.info("Launching: %s", " ".join(cmd))
    if cfg.get("dry_run"):
        return 0
    _out_dir(cfg).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if "gpu_ids" in cfg and cfg.gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_ids)
    return subprocess.run(cmd, env=env).returncode


def main() -> int:
    return _hydra_main()


if __name__ == "__main__":
    sys.exit(main())
