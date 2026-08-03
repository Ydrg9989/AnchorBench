"""Hydra-driven `anchorbench generate` subcommand.

Generates an AnchorBench dataset for the suite specified by the
composed Hydra config. Forwards to :func:`anchorbench.data.generate.generate_suite_dataset`.

Usage::

    anchorbench generate data=external
    anchorbench generate data=icl_dist size=core
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from anchorbench.data.generate import generate_suite_dataset

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = ROOT / "conf"


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def _hydra_main(cfg: DictConfig) -> int:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    suite = cfg.data.suite
    out_dir = cfg.data.dataset_dir
    size = cfg.get("size", "core")
    seed = cfg.get("seed", 42)
    info = generate_suite_dataset(
        suite=suite,
        size=size,
        seed=seed,
        out_dir=str(ROOT / out_dir),
    )
    log.info("Generated: %s", info)
    return 0


def main() -> int:
    return _hydra_main()


if __name__ == "__main__":
    sys.exit(main())
