#!/usr/bin/env python3
"""CLI: run HF model inference on the anchoring dataset for one config point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llm_anchoring.eval.inference import (
    build_gen_kwargs,
    generate_batch,
    load_model,
    prepare_prompt,
)
from llm_anchoring.eval.parse_utils import compute_base_id, diagnose_parse
from llm_anchoring.eval.parsers import parse_hierarchical

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on anchoring dataset")
    p.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--decoding", type=str, default="greedy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def _build_record(
    run_id: str,
    args: argparse.Namespace,
    item: dict,
    base_id: str,
    sample_idx: int,
    raw: str,
) -> dict:
    """Build one generation record with full parse trace + diagnostics."""
    trace = parse_hierarchical(raw, prefer_last=True)
    diag = diagnose_parse(
        trace,
        truth_value=item["truth_value"],
        anchor_value=item["anchor_value"],
    )

    chosen_raw = None
    if trace["chosen_idx"] is not None and trace["candidates"]:
        chosen_raw = trace["candidates"][trace["chosen_idx"]]["raw"]

    errors = []
    if not trace["parse_ok"]:
        errors.append("parse_failed")
    elif not trace["format_ok"]:
        errors.append("format_mismatch")

    pv = trace["parsed_value"]
    tv = item["truth_value"]
    return {
        "run_id": run_id,
        "model_id": args.model,
        "seed": args.seed,
        "sample_idx": sample_idx,
        "item_id": item["item_id"],
        "base_id": base_id,
        "template_id": item["template_id"],
        "domain": item["domain"],
        "icl_condition": item["icl_condition"],
        "anchor_value": item["anchor_value"],
        "truth_value": tv,
        "truth_formatted": item["truth_formatted"],
        "prompt": item["final_prompt"],
        "raw_output_text": raw,
        "parsed_value": pv,
        "parse_ok": trace["parse_ok"],
        "format_ok": trace["format_ok"],
        "abs_error": abs(pv - tv) if pv is not None else None,
        "signed_error": (pv - tv) if pv is not None else None,
        "errors": errors,
        "parse_method": trace["strategy_used"],
        "candidate_count": trace["candidate_count"],
        "chosen_raw": chosen_raw,
        "anchor_collision": diag["anchor_collision"],
        "mismatch_suspect": diag["mismatch_suspect"],
        "best_candidate_error": diag["best_candidate_error"],
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    items = [
        json.loads(l)
        for l in Path(cfg["dataset_path"]).read_text().strip().split("\n")
    ]
    if args.max_items:
        items = items[: args.max_items]
    logger.info("loaded %d items", len(items))

    dec_cfg = next((d for d in cfg["decoding"] if d["name"] == args.decoding), None)
    if dec_cfg is None:
        raise ValueError(f"decoding '{args.decoding}' not found in config")
    gen_kwargs = build_gen_kwargs(dec_cfg, cfg.get("max_new_tokens", 128))

    if dec_cfg.get("temperature", 0) == 0 and args.n_samples > 1:
        logger.warning("greedy + n_samples>1 is redundant; forcing n_samples=1")
        args.n_samples = 1

    model, tokenizer = load_model(args.model, dtype=cfg.get("dtype", "auto"))
    use_chat = cfg.get("use_chat_template", True)
    batch_size = cfg.get("batch_size", 8)

    model_short = args.model.split("/")[-1]
    run_id = str(uuid.uuid4())[:8]
    out_dir = args.out_dir or (
        Path(cfg.get("output_dir", "results/eval"))
        / model_short
        / f"{args.decoding}_s{args.seed}_n{args.n_samples}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"

    prompts = [prepare_prompt(it["final_prompt"], tokenizer, use_chat) for it in items]
    base_ids = [compute_base_id(it) for it in items]

    torch.manual_seed(args.seed)
    total = 0
    with out_path.open("w") as f:
        for si in range(args.n_samples):
            logger.info("sample %d/%d", si + 1, args.n_samples)
            for bs in range(0, len(items), batch_size):
                be = min(bs + batch_size, len(items))
                outputs = generate_batch(
                    prompts[bs:be], model, tokenizer, gen_kwargs
                )
                for i, raw in enumerate(outputs):
                    rec = _build_record(
                        run_id, args, items[bs + i], base_ids[bs + i], si, raw
                    )
                    f.write(json.dumps(rec) + "\n")
                    total += 1

    logger.info("wrote %d records to %s", total, out_path)

    run_cfg = {
        "run_id": run_id,
        "model": args.model,
        "decoding": args.decoding,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_items": len(items),
        "total_records": total,
        "gen_kwargs": {k: v for k, v in gen_kwargs.items()},
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))


if __name__ == "__main__":
    main()
