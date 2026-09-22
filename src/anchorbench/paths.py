"""Where the repository lives, and where its configs, datasets, results and outputs are.

Thirteen modules used to climb ``Path(__file__).resolve().parents[N]`` on
their own. That tied the package to being imported from a git checkout,
repeated a fragile piece of arithmetic in every module, and gave no way to
point the code at a results tree that lives somewhere else (a downloaded
results bundle, a shared drive). The root is resolved once here, and each
directory can be overridden through the environment:

    ANCHORBENCH_ROOT      repository root (default: the checkout this package is imported from)
    ANCHORBENCH_CONF      Hydra config tree      (default: <root>/conf)
    ANCHORBENCH_DATASETS  benchmark datasets     (default: <root>/datasets)
    ANCHORBENCH_RESULTS   model outputs          (default: <root>/results)
    ANCHORBENCH_OUTPUTS   generated figures/tables (default: <root>/outputs)

Analysis modules that default to relative ``results/...`` paths still expect
to be run from the repository root; this module is for the code that needs
an absolute anchor.
"""

from __future__ import annotations

import os
from pathlib import Path


def _discover_root() -> Path:
    env = os.environ.get("ANCHORBENCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "conf").is_dir():
            return candidate
    # src/anchorbench/paths.py -> src/anchorbench -> src -> <root>
    return here.parents[2]


def _dir(env_var: str, default: Path) -> Path:
    env = os.environ.get(env_var)
    return Path(env).expanduser().resolve() if env else default


ROOT: Path = _discover_root()
CONF_DIR: Path = _dir("ANCHORBENCH_CONF", ROOT / "conf")
DATASETS_DIR: Path = _dir("ANCHORBENCH_DATASETS", ROOT / "datasets")
RESULTS_DIR: Path = _dir("ANCHORBENCH_RESULTS", ROOT / "results")
OUTPUTS_DIR: Path = _dir("ANCHORBENCH_OUTPUTS", ROOT / "outputs")

__all__ = ["CONF_DIR", "DATASETS_DIR", "OUTPUTS_DIR", "RESULTS_DIR", "ROOT"]
