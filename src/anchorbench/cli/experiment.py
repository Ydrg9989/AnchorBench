"""Hydra-driven `anchorbench experiment` subcommand.

Composes ``conf/experiment/*.yaml`` (which references ``data``, ``decoding``,
``model``, and ``tier`` configs) and dispatches to a sequence of per-cell
runs. Each cell's command comes from :mod:`anchorbench.cli.cells`, the
same builder ``anchorbench eval`` uses.

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

from anchorbench.cli.cells import ROOT, build_cell_cmd, is_api_model

log = logging.getLogger(__name__)

CONF_DIR = ROOT / "conf"


def _load_yaml(rel: str) -> dict:
    import yaml
    with open(CONF_DIR / rel) as f:
        return yaml.safe_load(f) or {}


def _resolve_model(name: str) -> dict:
    return _load_yaml(f"model/{name}.yaml")


def _resolve_data(name: str, cfg: DictConfig | None = None) -> dict:
    """Load conf/data/<name>.yaml, preferring the composed config.

    The composed cfg.data carries the recipe's overrides -- for example
    paper_history_matched sets promptviews_file: promptviews.jsonl, because
    control_twostage is absent from promptviews_core.jsonl. Reading the YAML
    straight off disk discarded those silently, so that recipe pointed the
    runner at a file with no control_twostage rows and it found 0 items.
    """
    if cfg is not None:
        composed = cfg.get("data")
        if composed is not None and composed.get("suite") == name:
            return OmegaConf.to_container(composed, resolve=True)
    return _load_yaml(f"data/{name}.yaml")


def _resolve_tier(name: str) -> dict:
    return _load_yaml(f"tier/{name}.yaml")


def _resolve_cells(cfg: DictConfig) -> list[tuple[dict, dict, str | None, Path]]:
    """Return list of (model_dict, data_dict, gpu_ids, out_dir) cells."""
    suites: list[str] = list(cfg.get("suites", []))
    base_out = ROOT / cfg.get("out_dir", "results/experiment")
    api_out = ROOT / cfg.get("api_out_dir", str(base_out))

    cells: list[tuple[dict, dict, str | None, Path]] = []

    explicit_models = cfg.get("models")
    tiers: list[str] = list(cfg.get("tiers", []))

    suite_data = {s: _resolve_data(s, cfg) for s in suites}

    if explicit_models:
        for model_name in explicit_models:
            model = _resolve_model(model_name)
            for suite in suites:
                data = suite_data[suite]
                out = (api_out if is_api_model(model) else base_out / suite)
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
                out = (api_out if is_api_model(model) else base_out / suite)
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
        cmd = build_cell_cmd(model, data, decoding, out_dir,
                             baseline_condition=baseline,
                             tool_plaintext=tool_plaintext)
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
