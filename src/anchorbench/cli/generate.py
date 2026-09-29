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

import hydra
from omegaconf import DictConfig, OmegaConf

from anchorbench.data.generate import generate_suite_dataset
from anchorbench.paths import CONF_DIR, ROOT

log = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
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


def main() -> int:
    # hydra.main discards the task function's return value, so a failure is
    # signalled by raising SystemExit inside _hydra_main.
    _hydra_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
