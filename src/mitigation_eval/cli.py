"""CLI entrypoint for mitigation evaluation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .strategies import ALL_STRATEGIES, get_max_tokens
from .runner import (
    HFRunner,
    LLMFallbackExtractor,
    load_promptviews,
    load_itemspecs,
    run_mitigation_hf,
    run_mitigation_api,
)
from .evaluate import (
    load_results,
    compute_metrics,
    compute_reduction_rates,
    write_summary_csv,
    generate_latex_table,
)
from .plotting import plot_rr_vs_utility, plot_nai_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Mitigation evaluation pipeline")
    sub = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run_p = sub.add_parser("run", help="Run mitigation experiments")
    run_p.add_argument("--model_id", required=True)
    run_p.add_argument("--backend", choices=["hf", "openrouter"], default="hf")
    run_p.add_argument("--mitigation", default="ALL",
                       help="Comma-separated list or ALL")
    run_p.add_argument("--promptviews", type=Path, required=True)
    run_p.add_argument("--itemspecs", type=Path, required=True)
    run_p.add_argument("--max_items", type=int, default=None)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--out_dir", type=Path, default=Path("results/mitigation"))
    run_p.add_argument("--device", type=str, default="auto")
    run_p.add_argument("--device_map", type=str, default=None)
    run_p.add_argument("--dtype", type=str, default="bfloat16")
    run_p.add_argument("--batch_size", type=int, default=16)
    run_p.add_argument("--max_concurrent", type=int, default=20)
    run_p.add_argument("--llm_fallback", action="store_true", default=False,
                       help="Use a small HF model as fallback when regex parsing fails")
    run_p.add_argument("--fallback_model", type=str,
                       default="meta-llama/Llama-3.2-1B-Instruct",
                       help="HF model for fallback extraction")
    run_p.add_argument("--fallback_device", type=str, default="cpu",
                       help="Device for fallback model (cpu to avoid competing for GPU VRAM)")

    # --- eval subcommand ---
    eval_p = sub.add_parser("eval", help="Evaluate results and generate artifacts")
    eval_p.add_argument("--auto_discover", type=Path)
    eval_p.add_argument("--inputs", type=Path, nargs="*")
    eval_p.add_argument("--itemspecs", type=Path)
    eval_p.add_argument("--promptviews", type=Path)
    eval_p.add_argument("--out_dir", type=Path, default=Path("results/mitigation"))
    eval_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "eval":
        _cmd_eval(args)
    else:
        parser.print_help()


def _cmd_run(args):
    model_short = args.model_id.split("/")[-1]
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"{model_short}_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mitigation.upper() == "ALL":
        mitigations = ALL_STRATEGIES
    else:
        mitigations = [m.strip() for m in args.mitigation.split(",")]

    fallback = None
    if args.llm_fallback:
        log.info("Loading LLM fallback extractor: %s on %s",
                 args.fallback_model, args.fallback_device)
        fallback = LLMFallbackExtractor(
            model_id=args.fallback_model,
            device=args.fallback_device,
        )

    provenance = {
        "model_id": args.model_id,
        "backend": args.backend,
        "mitigations": mitigations,
        "git_commit": _git_hash(),
        "seed": args.seed,
        "max_items": args.max_items,
        "llm_fallback": args.llm_fallback,
        "fallback_model": args.fallback_model if args.llm_fallback else None,
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    log.info("Loading dataset...")
    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    log.info("Loaded %d items with prompt views", len(views))

    np.random.seed(args.seed)
    t0 = time.time()

    if args.backend == "hf":
        log.info("Loading HF model %s...", args.model_id)
        runner = HFRunner(
            args.model_id,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
        )

        for mit in mitigations:
            output_path = out_dir / f"{mit}_results.jsonl"
            log.info("Running %s...", mit)
            run_mitigation_hf(
                runner, views, specs, mit, output_path,
                max_items=args.max_items,
                batch_size=args.batch_size,
                seed=args.seed,
                fallback_extractor=fallback,
            )

    elif args.backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            env_path = Path(".env")
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break

        for mit in mitigations:
            output_path = out_dir / f"{mit}_results.jsonl"
            log.info("Running %s (API)...", mit)
            asyncio.run(run_mitigation_api(
                args.model_id, views, specs, mit, output_path,
                max_concurrent=args.max_concurrent,
                max_items=args.max_items,
                seed=args.seed,
                api_key=api_key,
                fallback_extractor=fallback,
            ))

    log.info("All mitigations done in %.1fs. Output: %s", time.time() - t0, out_dir)

    _generate_eval_artifacts(out_dir)


def _cmd_eval(args):
    """Evaluate results from existing JSONL files."""
    input_paths = []

    if args.auto_discover and args.auto_discover.exists():
        for sub in sorted(args.auto_discover.iterdir()):
            if sub.is_dir():
                for jsonl in sorted(sub.glob("*_results.jsonl")):
                    input_paths.append(jsonl)
    if args.inputs:
        input_paths.extend(args.inputs)

    if not input_paths:
        log.error("No input files found")
        return

    _generate_eval_artifacts_from_paths(input_paths, args.out_dir)


def _generate_eval_artifacts(out_dir: Path) -> None:
    """Generate eval artifacts from all JSONL files in a run directory."""
    jsonl_files = sorted(out_dir.glob("*_results.jsonl"))
    if not jsonl_files:
        return
    _generate_eval_artifacts_from_paths(jsonl_files, out_dir)


def _generate_eval_artifacts_from_paths(paths: list[Path], out_dir: Path) -> None:
    records = load_results(paths)
    if not records:
        return

    metrics = compute_metrics(records)
    metrics = compute_reduction_rates(metrics)

    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary_table.csv"
    write_summary_csv(metrics, csv_path)

    plot_rr_vs_utility(csv_path, out_dir / "figures" / "rr_vs_utility.pdf")
    plot_nai_heatmap(csv_path, out_dir / "figures" / "nai_heatmap.pdf")
    generate_latex_table(metrics, out_dir / "tables" / "mitigation_main.tex")


if __name__ == "__main__":
    main()
