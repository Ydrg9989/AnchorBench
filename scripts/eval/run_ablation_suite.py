#!/usr/bin/env python3
"""Run ablation promptviews for one suite/model and write parsed results.

Unlike core runners, this supports arbitrary condition sets in
`promptviews_ablation.jsonl` (not fixed 5-condition tuples).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench_eval.backends import HFBackend, VLLMBackend
from anchorbench_eval.evaluator import parse_response
from anchorbench_eval.io import load_itemspecs, load_promptviews
from anchorbench_eval.parsing import LLMFallbackExtractor

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Run suite ablation promptviews")
    p.add_argument("--suite", type=str, required=True)
    p.add_argument("--promptviews", type=Path, required=True)
    p.add_argument("--itemspecs", type=Path, required=True)
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--backend", type=str, choices=["hf", "vllm"], default="vllm")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--llm_fallback", action="store_true")
    p.add_argument("--fallback_model", type=str, default=None)
    p.add_argument("--fallback_device", type=str, default="auto")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)

    tasks: list[tuple[dict, dict, str]] = []
    for item_id, cond_views in views.items():
        spec = specs.get(item_id, {})
        for cond, pv in cond_views.items():
            item = {
                "item_id": item_id,
                "suite": pv.get("suite", args.suite),
                "domain": pv.get("domain", spec.get("domain", "unknown")),
                "difficulty": spec.get("difficulty", "standard"),
                "y_star_evidence": spec.get("y_star_evidence", spec.get("y_star")),
                "y_star_theta": spec.get("y_star_theta"),
            }
            tasks.append((item, pv, cond))

    log.info(
        "Loaded %d ablation prompts from %s",
        len(tasks), args.promptviews,
    )
    if not tasks:
        log.error("No ablation prompts found")
        sys.exit(1)

    fallback = None
    if args.llm_fallback:
        fb_model = args.fallback_model or args.model_id
        fallback = LLMFallbackExtractor(
            model_id=fb_model, device=args.fallback_device, dtype="bfloat16",
        )

    if args.backend == "vllm":
        backend = VLLMBackend(
            args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=True,
        )
    else:
        backend = HFBackend(args.model_id, device=args.device, dtype=args.dtype)

    prompts = [pv.get("prompt_text", "") for _, pv, _ in tasks]
    raws = backend.generate_batch(
        prompts,
        max_tokens=args.max_tokens,
        temperature=0.0,
        batch_size=args.batch_size,
    )

    out_path = args.out_dir / "results.jsonl"
    records: list[dict] = []
    parse_counts = Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for (item, pv, cond), raw in zip(tasks, raws):
            answer, ok, strategy = parse_response(
                raw, pv.get("prompt_text", ""),
                use_llm_fallback=args.llm_fallback,
                fallback_extractor=fallback,
            )
            rec = {
                "model_id": backend.model_id,
                "item_id": item["item_id"],
                "suite": item["suite"],
                "domain": item["domain"],
                "difficulty": item["difficulty"],
                "condition": cond,
                "anchor_relevance": pv.get("anchor_relevance", "none"),
                "anchor_value": pv.get("anchor_value"),
                "y_star_evidence": item["y_star_evidence"],
                "y_star_theta": item["y_star_theta"],
                "answer_int": answer,
                "parsed_ok": ok,
                "parse_strategy": strategy,
                "raw_text": raw,
            }
            parse_counts[strategy] += 1
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "suite": args.suite,
        "model_id": backend.model_id,
        "n_records": len(records),
        "n_parsed": sum(1 for r in records if r["parsed_ok"] and r["answer_int"] is not None),
        "parse_rate": sum(1 for r in records if r["parsed_ok"] and r["answer_int"] is not None) / len(records),
        "parse_strategy_counts": dict(parse_counts),
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s and summary.json", out_path)


if __name__ == "__main__":
    main()

