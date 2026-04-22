#!/usr/bin/env python3
"""Unified AnchorBench CLI.

Single entry point that dispatches to the existing per-suite runners
based on ``configs/benchmark.yaml``. The legacy shell scripts
(``run_full_benchmark.sh``, per-suite ``run_*_gpus.sh``) and per-suite
runners (``scripts/eval/run_*.py``) are kept intact and called as
sub-processes so old workflows still work.

Subcommands:
    eval        Run one (model, suite[, variant]) cell.
    experiment  Run a named recipe from `experiments:` in the YAML.
    tables      Regenerate paper figures and LaTeX tables.
    add-model   Append a new model to the YAML and print smoke-test command.

Examples:
    # Single cell
    python scripts/run.py eval --model Qwen/Qwen2.5-7B-Instruct --suite external

    # ICL-dist variant
    python scripts/run.py eval --model Qwen/Qwen2.5-7B-Instruct \
        --suite icl --variant icl_dist

    # Full paper recipe
    python scripts/run.py experiment --name paper_main

    # Regenerate every paper artifact
    python scripts/run.py tables --paper

    # Add a model and print smoke-test
    python scripts/run.py add-model openai/gpt-5o
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CONFIG_PATH = ROOT / "configs" / "benchmark.yaml"

sys.path.insert(0, str(SRC))

from anchorbench_eval.constants import (  # noqa: E402
    API_MODEL_IDS,
    DEFAULTS,
    EXPERIMENTS,
    MODEL_TIERS,
    OW_MODEL_IDS,
    PAPER_RUN,
    SUITE_DATASETS,
    VARIANT_DATASETS,
    get_experiment,
)


# ─── Suite runner mapping ────────────────────────────────────────

SUITE_RUNNERS = {
    "external": "scripts/eval/run_external.py",
    "history": "scripts/eval/run_history.py",
    "icl": "scripts/eval/run_icl.py",
    "rag": "scripts/eval/run_rag.py",
    "tool": "scripts/eval/run_tool.py",
}


def run_with_env() -> list[str]:
    """Wrap commands through the conda environment helper."""
    return ["bash", str(ROOT / "scripts" / "run_with_env.sh")]


def env_for_gpu(gpu_ids: str | None) -> dict[str, str]:
    e = os.environ.copy()
    if gpu_ids:
        e["CUDA_VISIBLE_DEVICES"] = gpu_ids
    return e


def dataset_dir(suite: str, variant: str | None) -> Path:
    """Resolve the dataset directory for a (suite, variant) pair."""
    if variant:
        if variant not in VARIANT_DATASETS:
            raise SystemExit(f"unknown variant: {variant} "
                             f"(known: {list(VARIANT_DATASETS)})")
        return ROOT / VARIANT_DATASETS[variant]
    if suite not in SUITE_DATASETS:
        raise SystemExit(f"unknown suite: {suite} "
                         f"(known: {list(SUITE_DATASETS)})")
    return ROOT / SUITE_DATASETS[suite]


def runner_for_suite(suite: str, variant: str | None) -> Path:
    """Resolve the per-suite runner script. Variants reuse the base runner."""
    base_suite = "icl" if variant and variant.startswith("icl") else suite
    if base_suite not in SUITE_RUNNERS:
        raise SystemExit(f"no runner for suite: {base_suite}")
    return ROOT / SUITE_RUNNERS[base_suite]


def is_api_model(model_id: str) -> bool:
    return model_id in set(API_MODEL_IDS)


# ─── Subcommand: eval ────────────────────────────────────────────

def cmd_eval(args: argparse.Namespace) -> int:
    suite = args.suite
    variant = args.variant
    model = args.model
    out_dir = Path(args.out_dir) if args.out_dir else None

    data_dir = dataset_dir(suite, variant)
    pv = data_dir / DEFAULTS.get("promptviews_file", "promptviews_core.jsonl")
    isp = data_dir / DEFAULTS.get("itemspecs_file", "itemspecs.jsonl")

    if is_api_model(model):
        # Use the API runner; it handles all suites internally.
        runner = ROOT / "scripts" / "eval" / "run_api_benchmark.py"
        out = out_dir or ROOT / "results" / "api_benchmark"
        cmd = run_with_env() + [
            "python", str(runner),
            "--model_id", model,
            "--suite", suite,
            "--out_dir", str(out),
        ]
        if variant:
            cmd += ["--variant", variant]
    else:
        runner = runner_for_suite(suite, variant)
        out = out_dir or ROOT / "results" / "full_benchmark" / suite
        cmd = run_with_env() + [
            "python", str(runner),
            "--model_id", model,
            "--promptviews", str(pv),
            "--itemspecs", str(isp),
            "--out_dir", str(out),
            "--backend", args.backend,
            "--batch_size", str(args.batch_size),
            "--gpu_memory_utilization", str(args.gpu_memory_utilization),
            "--max_model_len", str(args.max_model_len),
            "--tensor_parallel_size", str(args.tensor_parallel_size),
        ]
        if suite == "history":
            cmd += ["--baseline_condition",
                    args.baseline_condition or "control"]

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    env = env_for_gpu(args.gpu_ids)
    print(f"[eval] {model} / {suite}{f' ({variant})' if variant else ''}")
    return subprocess.run(cmd, env=env).returncode


# ─── Subcommand: experiment ──────────────────────────────────────

def cmd_experiment(args: argparse.Namespace) -> int:
    recipe = get_experiment(args.name)
    print(f"[experiment] {args.name}: {recipe.get('description', '')}")

    suites = recipe.get("suites", [])
    variant = recipe.get("variant")
    api_out = Path(recipe.get("api_out_dir", "results/api_benchmark"))
    base_out = Path(recipe.get("out_dir", "results"))

    # Collect (model, gpu_ids, tp) work items by tier.
    failures = 0
    if "models" in recipe:
        # Explicit model list — assign in round-robin to first GPU.
        for model in recipe["models"]:
            for suite in suites:
                rc = _run_cell(model, suite, variant,
                               base_out, api_out,
                               gpu_ids="0", tp=1, dry_run=args.dry_run)
                if rc != 0:
                    failures += 1
    else:
        for tier_name in recipe.get("model_tiers", []):
            tier = MODEL_TIERS[tier_name]
            print(f"\n=== Tier: {tier_name} ({tier.get('description', '')}) ===")
            tp = int(tier.get("tensor_parallel", 1))
            gpu_slots = [str(g) for g in tier.get("gpu_ids", []) or [None]]
            for i, model in enumerate(tier["models"]):
                gpu = gpu_slots[i % len(gpu_slots)] if gpu_slots else None
                for suite in suites:
                    rc = _run_cell(model, suite, variant,
                                   base_out, api_out,
                                   gpu_ids=gpu, tp=tp, dry_run=args.dry_run)
                    if rc != 0:
                        failures += 1
    return 1 if failures else 0


def _run_cell(model: str, suite: str, variant: str | None,
              base_out: Path, api_out: Path,
              gpu_ids: str | None, tp: int, dry_run: bool) -> int:
    args = argparse.Namespace(
        model=model, suite=suite, variant=variant,
        out_dir=str(api_out if is_api_model(model) else base_out / suite),
        backend="vllm", batch_size=64, max_model_len=4096,
        gpu_memory_utilization=0.9, tensor_parallel_size=tp,
        baseline_condition=None, gpu_ids=gpu_ids, dry_run=dry_run,
    )
    return cmd_eval(args)


# ─── Subcommand: tables ──────────────────────────────────────────

def cmd_tables(args: argparse.Namespace) -> int:
    umbrella = ROOT / "COLM" / "scripts" / "generate_paper_figures.py"
    cmd = [sys.executable, str(umbrella)]
    if args.figures:
        cmd.append("--figures")
    if args.tables_only:
        cmd.append("--tables")
    if args.skip_extensions:
        cmd.append("--skip_extensions")
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(SRC))
    return subprocess.run(cmd, env=env).returncode


# ─── Subcommand: add-model ───────────────────────────────────────

def cmd_add_model(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    model_id = args.model_id
    is_api = "/" in model_id and any(
        model_id.startswith(p) for p in
        ("openai/", "anthropic/", "google/", "x-ai/", "mistralai/")
    )
    key = "api_models" if is_api else "ow_models"
    if model_id in cfg.get(key, []):
        print(f"  Already present in {key}: {model_id}")
        return 0
    cfg.setdefault(key, []).append(model_id)
    slug = model_id.replace("/", "_")
    if "model_short_names" in cfg and slug not in cfg["model_short_names"]:
        suggested = model_id.split("/")[-1]
        cfg["model_short_names"][slug] = suggested
    CONFIG_PATH.write_text(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
    print(f"  Added to {key}: {model_id}")
    print(f"  Slug: {slug}")
    print()
    print("  Smoke test:")
    print(f"    python scripts/run.py eval --model {model_id} \\")
    print(f"        --suite external --gpu_ids 0 --batch_size 8")
    return 0


# ─── argparse wiring ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anchorbench",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="Run one (model, suite) cell.")
    e.add_argument("--model", required=True, help="HF or OpenRouter model id")
    e.add_argument("--suite", required=True, choices=list(SUITE_RUNNERS))
    e.add_argument("--variant", default=None,
                   help="e.g. icl_dist (uses VARIANT_DATASETS).")
    e.add_argument("--out_dir", default=None)
    e.add_argument("--gpu_ids", default=None,
                   help="CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1,2,3').")
    e.add_argument("--backend", default="vllm")
    e.add_argument("--batch_size", type=int, default=64)
    e.add_argument("--max_model_len", type=int, default=4096)
    e.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    e.add_argument("--tensor_parallel_size", type=int, default=1)
    e.add_argument("--baseline_condition", default=None,
                   help="History only: 'control' or 'control_twostage'.")
    e.add_argument("--dry_run", action="store_true",
                   help="Print the command without running it.")
    e.set_defaults(func=cmd_eval)

    x = sub.add_parser("experiment", help="Run a named recipe from YAML.")
    x.add_argument("--name", required=True,
                   help=f"Known: {', '.join(sorted(EXPERIMENTS)) or '<configure benchmark.yaml>'}")
    x.add_argument("--dry_run", action="store_true")
    x.set_defaults(func=cmd_experiment)

    t = sub.add_parser("tables", help="Regenerate paper figures and tables.")
    t.add_argument("--paper", action="store_true",
                   help="Run the full paper umbrella (default if no flags).")
    t.add_argument("--figures", action="store_true",
                   help="Only regenerate Figure 4 + Figure 5.")
    t.add_argument("--tables_only", action="store_true",
                   help="Only regenerate LaTeX tables.")
    t.add_argument("--skip_extensions", action="store_true",
                   help="Skip slow gold-shift / sampling / mitigation scripts.")
    t.add_argument("--dry_run", action="store_true")
    t.set_defaults(func=cmd_tables)

    a = sub.add_parser("add-model", help="Append a model to benchmark.yaml.")
    a.add_argument("model_id", help="HF Hub or OpenRouter model id")
    a.set_defaults(func=cmd_add_model)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
