#!/usr/bin/env python3
"""Parallel multi-GPU sweep: dispatches runs across GPUs via subprocess.

Main mode:   groups runs by model, launches up to N_GPU workers in parallel.
Worker mode: runs one (model, preset, seed) on the single visible GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
THIS_FILE = str(Path(__file__).resolve())


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full evaluation sweep")
    p.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--n_gpus", type=int, default=None,
                   help="Override GPU count (default: auto-detect)")
    # Worker mode (called internally by the dispatcher)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


# ── Run expansion ────────────────────────────────────────────────────

def _expand_runs(cfg: dict) -> list[dict]:
    """Expand (model, preset) × seeds into individual run specs."""
    presets = cfg["presets"]
    runs = []
    for entry in cfg["runs"]:
        p = presets[entry["preset"]]
        n_samp = p["n_samples"]
        if p.get("temperature", 0) == 0 and n_samp > 1:
            n_samp = 1
        model_short = entry["model"].split("/")[-1]
        for seed in p["seeds"]:
            runs.append({
                "model": entry["model"],
                "model_short": model_short,
                "preset": entry["preset"],
                "seed": seed,
                "n_samples": n_samp,
                "run_name": f"{model_short}_{entry['preset']}_s{seed}",
            })
    return runs


# ── Dispatcher (main mode) ──────────────────────────────────────────

def _detect_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 1


def _launch_worker(
    gpu_id: int, model: str, preset: str, seed: int,
    config_path: Path, no_wandb: bool,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, THIS_FILE, "--worker",
        "--config", str(config_path),
        "--model", model,
        "--preset", preset,
        "--seed", str(seed),
    ]
    if no_wandb:
        cmd.append("--no_wandb")
    return subprocess.run(cmd, env=env)


def _dispatch(args: argparse.Namespace) -> None:
    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    all_runs = _expand_runs(cfg)
    n_gpus = args.n_gpus or _detect_gpus()
    logger.info("sweep: %d runs across %d GPUs", len(all_runs), n_gpus)

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in all_runs:
        by_model[r["model"]].append(r)

    total_done = 0
    for model_id, model_runs in by_model.items():
        n_workers = min(n_gpus, len(model_runs))
        logger.info(
            "▶ %s — %d runs on %d GPUs",
            model_id.split("/")[-1], len(model_runs), n_workers,
        )
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for i, run in enumerate(model_runs):
                gpu = i % n_gpus
                fut = pool.submit(
                    _launch_worker, gpu,
                    run["model"], run["preset"], run["seed"],
                    args.config, args.no_wandb,
                )
                futures[fut] = (run["run_name"], gpu)
            for fut in as_completed(futures):
                name, gpu = futures[fut]
                result = fut.result()
                if result.returncode != 0:
                    logger.error("FAILED %s (GPU %d) exit=%d", name, gpu, result.returncode)
                else:
                    total_done += 1
                    logger.info("✓ %s (GPU %d)", name, gpu)

    logger.info("sweep complete: %d/%d runs succeeded", total_done, len(all_runs))


# ── Worker (single-run mode) ────────────────────────────────────────

def _worker_main(args: argparse.Namespace) -> None:
    import gc

    import torch
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from llm_anchoring.eval.inference import (
        build_gen_kwargs,
        generate_batch,
        load_model,
        prepare_prompt,
    )
    from llm_anchoring.eval.metrics import compute_paired_anchoring, condition_summary
    from llm_anchoring.eval.parse_utils import compute_base_id, diagnose_parse
    from llm_anchoring.eval.parsers import parse_hierarchical
    from llm_anchoring.eval.wandb_logger import (
        finish,
        init_run,
        log_condition_metrics,
        log_dataset_info,
        log_per_base_table,
        log_summary,
    )

    cfg = yaml.safe_load(args.config.read_text())
    preset_cfg = cfg["presets"][args.preset]
    n_samples = preset_cfg["n_samples"]
    if preset_cfg.get("temperature", 0) == 0 and n_samples > 1:
        n_samples = 1

    model_short = args.model.split("/")[-1]
    run_name = f"{model_short}_{args.preset}_s{args.seed}"
    out_dir = Path(cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in Path(cfg["dataset_path"]).read_text().strip().split("\n")]
    base_ids = [compute_base_id(it) for it in items]
    logger.info("[%s] %d items, %d samples", run_name, len(items), n_samples)

    gen_kwargs = build_gen_kwargs(preset_cfg, cfg["max_new_tokens"])
    model, tokenizer = load_model(args.model, dtype=cfg.get("dtype", "auto"))
    gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    use_chat = cfg.get("use_chat_template", True)
    prompts = [prepare_prompt(it["final_prompt"], tokenizer, use_chat) for it in items]
    batch_size = cfg.get("batch_size", 64)
    col_tol = cfg.get("anchor_collision_tol", 0.01)
    run_id = str(uuid.uuid4())[:8]

    torch.manual_seed(args.seed)
    records: list[dict] = []

    for si in range(n_samples):
        logger.info("[%s] sample %d/%d", run_name, si + 1, n_samples)
        for bs in range(0, len(items), batch_size):
            be = min(bs + batch_size, len(items))
            outputs = generate_batch(prompts[bs:be], model, tokenizer, gen_kwargs)
            for i, raw in enumerate(outputs):
                it = items[bs + i]
                trace = parse_hierarchical(raw, prefer_last=True)
                diag = diagnose_parse(
                    trace, it["truth_value"], it["anchor_value"],
                    collision_tol=col_tol,
                )
                pv = trace["parsed_value"]
                tv = it["truth_value"]
                chosen_raw = None
                if trace["chosen_idx"] is not None and trace["candidates"]:
                    chosen_raw = trace["candidates"][trace["chosen_idx"]]["raw"]

                records.append({
                    "run_id": run_id,
                    "model_id": args.model,
                    "seed": args.seed,
                    "sample_idx": si,
                    "item_id": it["item_id"],
                    "base_id": base_ids[bs + i],
                    "template_id": it["template_id"],
                    "domain": it["domain"],
                    "icl_condition": it["icl_condition"],
                    "anchor_value": it["anchor_value"],
                    "truth_value": tv,
                    "truth_formatted": it["truth_formatted"],
                    "prompt": it["final_prompt"],
                    "raw_output_text": raw,
                    "parsed_value": pv,
                    "parse_ok": trace["parse_ok"],
                    "format_ok": trace["format_ok"],
                    "abs_error": abs(pv - tv) if pv is not None else None,
                    "signed_error": (pv - tv) if pv is not None else None,
                    "parse_method": trace["strategy_used"],
                    "candidate_count": trace["candidate_count"],
                    "chosen_raw": chosen_raw,
                    "anchor_collision": diag["anchor_collision"],
                    "mismatch_suspect": diag["mismatch_suspect"],
                    "best_candidate_error": diag["best_candidate_error"],
                })

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    _save_and_report(records, run_id, run_name, args, cfg, out_dir)


def _save_and_report(
    records: list[dict],
    run_id: str,
    run_name: str,
    args: argparse.Namespace,
    cfg: dict,
    out_dir: Path,
) -> None:
    from llm_anchoring.eval.metrics import compute_paired_anchoring, condition_summary
    from llm_anchoring.eval.wandb_logger import (
        finish,
        init_run,
        log_condition_metrics,
        log_dataset_info,
        log_per_base_table,
        log_summary,
    )

    with (out_dir / "generations.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    run_cfg = {
        "run_id": run_id, "model": args.model, "preset": args.preset,
        "seed": args.seed, "n_samples": cfg["presets"][args.preset]["n_samples"],
        "n_items": len({r["item_id"] for r in records}),
        "total_records": len(records),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    cond_summ = condition_summary(records)
    anchoring = compute_paired_anchoring(records)

    summary = {
        "condition_summary": cond_summ,
        "ias_mean": anchoring.get("ias_mean"),
        "ias_median": anchoring.get("ias_median"),
        "ias_ci_95": anchoring.get("ias_ci_95"),
        "outlier_shift_mean": anchoring.get("outlier_shift_mean"),
        "outlier_shift_median": anchoring.get("outlier_shift_median"),
        "n_records": len(records),
        "n_base_ids": len(anchoring["per_base"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    per_base = anchoring["per_base"]
    if per_base:
        with (out_dir / "per_base.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_base[0].keys())
            w.writeheader()
            w.writerows(per_base)

    ias_str = f"mean={anchoring.get('ias_mean', 0) or 0:.4f}"
    ci = anchoring.get("ias_ci_95")
    if ci:
        ias_str += f"  95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]"
    logger.info("[%s] IAS %s  OutlierShift=%.4f", run_name, ias_str,
                anchoring.get("outlier_shift_mean", 0) or 0)

    for cond, m in cond_summ.items():
        acr = m.get("anchor_collision_rate", 0)
        if acr > 0:
            logger.info("[%s] %s anchor_collision=%.3f", run_name, cond, acr)

    if not args.no_wandb:
        wb = cfg.get("wandb", {})
        wb_run = init_run(
            config={**run_cfg, "anchor_collision_tol": cfg.get("anchor_collision_tol")},
            run_name=run_name,
            project=wb.get("project", "llm-anchoring"),
            entity=wb.get("entity"),
        )
        log_dataset_info(cfg.get("dataset_path", ""), wb_run)
        log_summary({
            "ias_mean": summary["ias_mean"],
            "ias_median": summary["ias_median"],
            "outlier_shift_mean": summary["outlier_shift_mean"],
        }, wb_run)
        log_condition_metrics(cond_summ, wb_run)
        log_per_base_table(per_base, wb_run)
        finish(wb_run)

    logger.info("[%s] saved → %s", run_name, out_dir)


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.worker:
        _worker_main(args)
    else:
        _dispatch(args)


if __name__ == "__main__":
    main()
