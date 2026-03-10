#!/usr/bin/env python3
"""Run inference on Tool v2 dataset (5 conditions, injected mode) and evaluate.

Uses batched inference with chat-template tool messages to maximize GPU
utilization. The model receives structured tool call + tool response
messages via its native chat template format.

Conditions: control, irrelevant_low/high, plausible_low/high.
Only the check_external_reference output varies across conditions.

Usage:
    PYTHONPATH=src python scripts/eval/run_tool_v2.py \\
        --promptviews datasets/anchorbench_v2_tool_pilot/promptviews.jsonl \\
        --itemspecs datasets/anchorbench_v2_tool_pilot/itemspecs.jsonl \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --out_dir results/tool_v2_pilot \\
        --batch_size 32

For multi-GPU: use run_tool_v2_gpus.sh (one process per GPU).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mitigation_eval.runner import (
    HFRunner,
    LLMFallbackExtractor,
    load_promptviews,
    load_itemspecs,
    parse_answer_int,
    _parse_with_fallback,
)

from anchorbench_v1.schema import ItemSpec
from anchorbench_v1.suites.tool_v2 import (
    TOOL_V2_SCHEMAS,
    TOOL_V2_CONDITIONS,
    get_tool_messages,
)

log = logging.getLogger(__name__)

TOOL_V2_COND_NAMES = [c[0] for c in TOOL_V2_CONDITIONS]


def prepare_items(
    views: dict, specs: dict, max_items: int | None, seed: int
) -> list[dict]:
    """Build list of items with all 5 Tool v2 conditions."""
    items = []
    for item_id, cond_views in views.items():
        if set(cond_views.keys()) != set(TOOL_V2_COND_NAMES):
            continue
        spec = specs.get(item_id, {})
        item = {
            "item_id": item_id,
            "suite": cond_views["control"]["suite"],
            "domain": cond_views["control"]["domain"],
            "difficulty": spec.get("difficulty", "standard"),
            "y_star_evidence": spec.get("y_star_evidence", spec.get("y_star")),
            "y_star_theta": spec.get("y_star_theta"),
            "anchors": spec.get("anchors", {}),
            "spec": spec,
        }
        for c in TOOL_V2_COND_NAMES:
            item[c] = cond_views[c]
        items.append(item)

    if max_items and max_items < len(items):
        rng = np.random.RandomState(seed)
        rng.shuffle(items)
        items = items[:max_items]

    return items


def run_inference(
    runner: HFRunner,
    items: list[dict],
    out_path: Path,
    max_tokens: int = 64,
    batch_size: int = 32,
    fallback_extractor: LLMFallbackExtractor | None = None,
    use_chat_template: bool = True,
) -> list[dict]:
    """Run model on every item x condition using batched inference."""

    task_list = []
    for item_idx, item in enumerate(items):
        for cond in TOOL_V2_COND_NAMES:
            pv = item[cond]
            task_list.append((item_idx, cond, pv))

    use_fallback = fallback_extractor is not None

    if use_chat_template:
        messages_list = []
        for item_idx, cond, pv in task_list:
            item = items[item_idx]
            spec_dict = item["spec"]
            spec = ItemSpec.from_dict(spec_dict)
            msgs, _ = get_tool_messages(spec, cond)
            messages_list.append(msgs)

        log.info(
            "Running batched tool inference: %d prompts, batch_size=%d (chat template)",
            len(messages_list), batch_size,
        )
        raw_outputs = runner.generate_batch_tool(
            messages_list,
            tools=TOOL_V2_SCHEMAS,
            max_tokens=max_tokens,
            temperature=0.0,
            batch_size=batch_size,
        )
    else:
        prompts_only = [pv["prompt_text"] for _, _, pv in task_list]
        log.info(
            "Running batched inference: %d prompts, batch_size=%d (plaintext fallback)",
            len(prompts_only), batch_size,
        )
        raw_outputs = runner.generate_batch(
            prompts_only,
            max_tokens=max_tokens,
            temperature=0.0,
            batch_size=batch_size,
        )

    records = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for (item_idx, cond, pv), raw in zip(task_list, raw_outputs):
            item = items[item_idx]
            prompt_text = pv.get("prompt_text", "")

            if use_fallback:
                answer, parsed_ok, parse_strategy = _parse_with_fallback(
                    raw, prompt_text, is_cot=False, fallback=fallback_extractor
                )
            else:
                answer, parsed_ok = parse_answer_int(raw, prompt_text)
                parse_strategy = "regex" if parsed_ok else "failed"

            rec = {
                "model_id": runner.model_id,
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
                "parsed_ok": parsed_ok,
                "parse_strategy": parse_strategy,
                "raw_text": raw,
            }
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    return records


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run Tool v2 inference (batched) and evaluation")
    p.add_argument("--promptviews", type=Path, required=True)
    p.add_argument("--itemspecs", type=Path, required=True)
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out_dir", type=Path, default=Path("results/tool_v2_pilot"))
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--llm_fallback", action="store_true")
    p.add_argument("--fallback_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--fallback_device", type=str, default="cuda:0")
    p.add_argument("--no_chat_template", action="store_true",
                   help="Use plaintext fallback instead of chat template rendering")
    args = p.parse_args()

    model_slug = args.model_id.replace("/", "_")
    model_out_dir = args.out_dir / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset...")
    views = load_promptviews(args.promptviews)
    specs = load_itemspecs(args.itemspecs)
    items = prepare_items(views, specs, args.max_items, args.seed)
    log.info("Loaded %d items (%d prompts)", len(items), len(items) * len(TOOL_V2_COND_NAMES))

    if not items:
        log.error("No Tool v2 items found (need 5 conditions per item)")
        sys.exit(1)

    fallback = None
    if args.llm_fallback:
        log.info("Loading LLM fallback extractor: %s on %s", args.fallback_model, args.fallback_device)
        fallback = LLMFallbackExtractor(
            model_id=args.fallback_model,
            device=args.fallback_device,
            dtype="bfloat16",
        )

    log.info("Loading model %s (device=%s, device_map=%s)...", args.model_id, args.device, args.device_map)
    runner = HFRunner(
        args.model_id,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
    )

    results_path = model_out_dir / "results.jsonl"
    log.info("Running batched inference -> %s (batch_size=%d)", results_path, args.batch_size)
    records = run_inference(
        runner,
        items,
        results_path,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        fallback_extractor=fallback,
        use_chat_template=not args.no_chat_template,
    )

    # Compute unified metrics
    unified_metrics_path = Path(__file__).parent / "unified_metrics.py"
    sys.path.insert(0, str(unified_metrics_path.parent))
    from unified_metrics import compute_unified_metrics, print_summary

    metrics = compute_unified_metrics(records)
    summary_path = model_out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Summary -> %s", summary_path)

    print_summary(metrics, label=f"Tool v2 | {args.model_id}")


if __name__ == "__main__":
    main()
