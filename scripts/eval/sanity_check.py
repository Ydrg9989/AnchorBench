#!/usr/bin/env python3
"""Sanity-check: run 50 items per experiment run with the updated parser.

Dispatcher mode:  groups runs by model, launches workers across GPUs.
Worker mode:      runs one (model, preset, seed) on 50 items, writes report.

Outputs per run go to results/sanity/<run_name>/:
  generations.jsonl     — full records with parse trace
  sanity_summary.json   — rates + method distribution
  mismatches.jsonl      — top mismatch suspects for manual inspection
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
THIS_FILE = str(Path(__file__).resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parser sanity check (50 items)")
    p.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--max_items", type=int, default=50)
    p.add_argument("--n_gpus", type=int, default=None)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


# ── Dispatcher ───────────────────────────────────────────────────────

def _expand_runs(cfg: dict) -> list[dict]:
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


def _detect_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 1


def _dispatch(args: argparse.Namespace) -> None:
    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    all_runs = _expand_runs(cfg)
    n_gpus = args.n_gpus or _detect_gpus()
    logger.info("sanity check: %d runs, %d items each, %d GPUs",
                len(all_runs), args.max_items, n_gpus)

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in all_runs:
        by_model[r["model"]].append(r)

    total_done = 0
    for model_id, model_runs in by_model.items():
        n_w = min(n_gpus, len(model_runs))
        logger.info("▶ %s — %d runs on %d GPUs",
                     model_id.split("/")[-1], len(model_runs), n_w)
        with ThreadPoolExecutor(max_workers=n_w) as pool:
            futures = {}
            for i, run in enumerate(model_runs):
                gpu = i % n_gpus
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                cmd = [
                    sys.executable, THIS_FILE, "--worker",
                    "--config", str(args.config),
                    "--model", run["model"],
                    "--preset", run["preset"],
                    "--seed", str(run["seed"]),
                    "--max_items", str(args.max_items),
                ]
                fut = pool.submit(subprocess.run, cmd, env=env)
                futures[fut] = (run["run_name"], gpu)
            for fut in as_completed(futures):
                name, gpu = futures[fut]
                res = fut.result()
                if res.returncode != 0:
                    logger.error("FAILED %s (GPU %d)", name, gpu)
                else:
                    total_done += 1
                    logger.info("✓ %s (GPU %d)", name, gpu)
    logger.info("sanity check done: %d/%d", total_done, len(all_runs))


# ── Worker ───────────────────────────────────────────────────────────

def _worker_main(args: argparse.Namespace) -> None:
    import gc

    import torch
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from llm_anchoring.eval.inference import (
        build_gen_kwargs, generate_batch, load_model, prepare_prompt,
    )
    from llm_anchoring.eval.parse_utils import compute_base_id, diagnose_parse
    from llm_anchoring.eval.parsers import parse_hierarchical

    cfg = yaml.safe_load(args.config.read_text())
    preset_cfg = cfg["presets"][args.preset]
    n_samples = preset_cfg["n_samples"]
    if preset_cfg.get("temperature", 0) == 0 and n_samples > 1:
        n_samples = 1

    model_short = args.model.split("/")[-1]
    run_name = f"{model_short}_{args.preset}_s{args.seed}"
    out_dir = Path("results/sanity") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in
             Path(cfg["dataset_path"]).read_text().strip().split("\n")]
    items = items[: args.max_items]
    base_ids = [compute_base_id(it) for it in items]
    logger.info("[%s] %d items, %d samples", run_name, len(items), n_samples)

    gen_kwargs = build_gen_kwargs(preset_cfg, cfg["max_new_tokens"])
    model, tokenizer = load_model(args.model, dtype=cfg.get("dtype", "auto"))
    gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    use_chat = cfg.get("use_chat_template", True)
    prompts = [prepare_prompt(it["final_prompt"], tokenizer, use_chat)
               for it in items]
    batch_size = cfg.get("batch_size", 128)
    col_tol = cfg.get("anchor_collision_tol", 0.01)
    run_id = str(uuid.uuid4())[:8]

    torch.manual_seed(args.seed)
    records: list[dict] = []

    for si in range(n_samples):
        if n_samples > 1:
            logger.info("[%s] sample %d/%d", run_name, si + 1, n_samples)
        for bs in range(0, len(items), batch_size):
            be = min(bs + batch_size, len(items))
            outputs = generate_batch(prompts[bs:be], model, tokenizer,
                                      gen_kwargs)
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
                    "icl_condition": it["icl_condition"],
                    "truth_value": tv,
                    "raw_output_text": raw,
                    "parsed_value": pv,
                    "parse_ok": trace["parse_ok"],
                    "format_ok": trace["format_ok"],
                    "abs_error": abs(pv - tv) if pv is not None else None,
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

    _report(records, run_name, out_dir)


def _report(records: list[dict], run_name: str, out_dir: Path) -> None:
    n = len(records)
    pok = sum(1 for r in records if r["parse_ok"])
    fok = sum(1 for r in records if r["format_ok"])
    acol = sum(1 for r in records if r.get("anchor_collision"))
    mism = sum(1 for r in records if r.get("mismatch_suspect"))
    methods = Counter(r["parse_method"] for r in records)

    summary = {
        "run_name": run_name,
        "n_records": n,
        "parse_rate": pok / n if n else 0,
        "format_rate": fok / n if n else 0,
        "anchor_collision_rate": acol / n if n else 0,
        "mismatch_suspect_rate": mism / n if n else 0,
        "method_distribution": dict(methods.most_common()),
    }

    with (out_dir / "generations.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out_dir / "sanity_summary.json").write_text(
        json.dumps(summary, indent=2))

    suspects = sorted(
        [r for r in records if r.get("mismatch_suspect")],
        key=lambda r: r.get("abs_error") or 0, reverse=True,
    )[:20]
    with (out_dir / "mismatches.jsonl").open("w") as f:
        for r in suspects:
            f.write(json.dumps({
                "item_id": r["item_id"],
                "icl_condition": r["icl_condition"],
                "truth_value": r["truth_value"],
                "parsed_value": r["parsed_value"],
                "abs_error": r["abs_error"],
                "best_candidate_error": r["best_candidate_error"],
                "parse_method": r["parse_method"],
                "chosen_raw": r["chosen_raw"],
                "completion": r["raw_output_text"][:500],
            }) + "\n")

    logger.info(
        "[%s] n=%d  parse=%.3f  fmt=%.3f  mismatch=%.3f  acol=%.3f",
        run_name, n, summary["parse_rate"], summary["format_rate"],
        summary["mismatch_suspect_rate"], summary["anchor_collision_rate"],
    )
    method_str = "  ".join(f"{k}={v}" for k, v in methods.most_common())
    logger.info("[%s] methods: %s", run_name, method_str)
    logger.info("[%s] saved → %s", run_name, out_dir)


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker_main(args)
    else:
        _dispatch(args)


if __name__ == "__main__":
    main()
