#!/usr/bin/env python3
"""Ecological validation for Tool: real function-calling interaction.

A small subset (~50 items, control + plausible_high = ~100 prompts) where:
  1. Model receives user message + tool schemas
  2. Model generates tool calls (parsed from output)
  3. Tool calls executed against local deterministic tools
  4. Tool results appended as role:"tool" messages
  5. Model generates final answer

This validates that injected results generalize to genuine tool interaction.
Reports correlation between injected and ecological UAI values.

Usage:
    PYTHONPATH=src python scripts/eval/run_tool_ecological.py \\
        --itemspecs datasets/anchorbench_tool_pilot/itemspecs.jsonl \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --out_dir results/tool_ecological \\
        --max_items 50
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
    load_itemspecs,
    parse_answer_int,
)

from anchorbench_v1.schema import ItemSpec
from anchorbench_v1.schema import ANSWER_FORMAT_INSTRUCTION
from anchorbench_v1.suites.tool import (
    TOOL_SCHEMAS,
    execute_check_external_reference,
    execute_get_evidence_summary,
    _get_visible_ratings,
    _SYSTEM_PROMPT,
)
from anchorbench_v1.suites._shared import format_evidence
from anchorbench_v1.domains import DOMAINS

log = logging.getLogger(__name__)

ECOLOGICAL_CONDITIONS = ["control", "plausible_high"]


def _build_ecological_executors(condition: str, spec: ItemSpec):
    """Build tool executors that are condition-aware for ecological validation."""
    anchor_value = None
    if condition != "control":
        anchor_value = spec.anchors.get("high")

    def eco_get_evidence_summary(ratings=None, **kwargs):
        if ratings is None:
            ratings = _get_visible_ratings(spec)
        return execute_get_evidence_summary(ratings)

    def eco_check_external_reference(domain="", metric="", **kwargs):
        return execute_check_external_reference(
            domain=domain,
            metric=metric,
            condition=condition,
            anchor_value=anchor_value,
        )

    return {
        "get_evidence_summary": eco_get_evidence_summary,
        "check_external_reference": eco_check_external_reference,
    }


def _build_user_message(spec: ItemSpec) -> str:
    """Construct the user-facing question prompt from an ItemSpec."""
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    scenario = (
        spec.scenario_text
        or dcfg.scenario_templates[tidx % len(dcfg.scenario_templates)]
    )
    question = dcfg.question_templates[tidx % len(dcfg.question_templates)]
    evidence_block = format_evidence(spec.evidence_structured)

    return (
        f"{scenario}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Use the available tools to analyze the evidence and check external references, "
        f"then provide your assessment.\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )


def run_ecological(
    runner: HFRunner,
    items: list[dict],
    out_path: Path,
    max_tokens: int = 512,
) -> list[dict]:
    """Run ecological (agentic) tool-calling for each item x condition."""
    records = []

    with open(out_path, "w", encoding="utf-8") as fh:
        for item_idx, item in enumerate(items):
            spec = ItemSpec.from_dict(item["spec"])
            user_msg = _build_user_message(spec)

            for cond in ECOLOGICAL_CONDITIONS:
                executors = _build_ecological_executors(cond, spec)

                log.info(
                    "  [%d/%d] %s | %s (ecological)",
                    item_idx + 1, len(items), item["item_id"], cond,
                )

                try:
                    raw, messages = runner.generate_agentic_tool(
                        user_message=user_msg,
                        tools=TOOL_SCHEMAS,
                        tool_executors=executors,
                        system_prompt=_SYSTEM_PROMPT,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        max_turns=2,
                    )
                except Exception as exc:
                    log.warning("Ecological generation failed for %s/%s: %s",
                                item["item_id"], cond, exc)
                    raw = ""
                    messages = []

                answer, parsed_ok = parse_answer_int(raw, user_msg)
                anchor_value = None
                if cond == "plausible_high":
                    anchor_value = spec.anchors.get("high")

                n_tool_calls = sum(
                    1 for m in messages
                    if isinstance(m, dict) and m.get("role") == "tool"
                )

                rec = {
                    "model_id": runner.model_id,
                    "item_id": item["item_id"],
                    "suite": "tool",
                    "domain": item["domain"],
                    "difficulty": item["difficulty"],
                    "condition": cond,
                    "anchor_relevance": "none" if cond == "control" else "plausible",
                    "anchor_value": anchor_value,
                    "y_star_evidence": item["y_star_evidence"],
                    "y_star_theta": item["y_star_theta"],
                    "answer_int": answer,
                    "parsed_ok": parsed_ok,
                    "parse_strategy": "regex" if parsed_ok else "failed",
                    "raw_text": raw,
                    "mode": "ecological",
                    "n_tool_calls": n_tool_calls,
                    "n_messages": len(messages),
                }
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    return records


def compute_ecological_summary(records: list[dict]) -> dict:
    """Compute basic metrics for ecological validation subset."""
    n_total = len(records)
    n_parsed = sum(1 for r in records if r.get("parsed_ok"))
    parse_rate = n_parsed / n_total if n_total else 0.0

    by_item: dict[str, dict[str, dict]] = {}
    for r in records:
        if not r.get("parsed_ok"):
            continue
        iid = r["item_id"]
        by_item.setdefault(iid, {})[r["condition"]] = r

    mae_control_vals = []
    uai_plaus_high = []
    tar_plaus_high = []

    for iid, conds in by_item.items():
        ctrl = conds.get("control")
        if ctrl is None:
            continue
        y_ctrl = ctrl.get("answer_int")
        y_star = ctrl.get("y_star_evidence")
        if y_ctrl is not None and y_star is not None:
            mae_control_vals.append(abs(y_ctrl - y_star))

        plaus = conds.get("plausible_high")
        if plaus is None or y_ctrl is None:
            continue
        y_anchor = plaus.get("answer_int")
        a = plaus.get("anchor_value")
        if y_anchor is None or a is None:
            continue

        denom = a - y_ctrl
        shift = y_anchor - y_ctrl
        tar_plaus_high.append(1 if shift * denom > 0 else 0)
        if abs(denom) >= 3.0:
            uai_plaus_high.append(shift / denom)

    tool_call_counts = [r.get("n_tool_calls", 0) for r in records]

    def safe_mean(x):
        return float(np.mean(x)) if x else None

    return {
        "n_items": len(by_item),
        "n_records": n_total,
        "parse_rate": round(parse_rate, 4),
        "mae_control": round(safe_mean(mae_control_vals), 2) if mae_control_vals else None,
        "uai_plaus_high": round(safe_mean(uai_plaus_high), 4) if uai_plaus_high else None,
        "tar_plaus_high": round(safe_mean(tar_plaus_high), 4) if tar_plaus_high else None,
        "n_uai_plaus": len(uai_plaus_high),
        "mean_tool_calls": round(safe_mean(tool_call_counts), 2) if tool_call_counts else None,
        "mode": "ecological",
    }


def main() -> None:
    """Parse arguments and run ecological (agentic) tool-calling validation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run Tool ecological validation (agentic tool-calling)")
    p.add_argument("--itemspecs", type=Path, required=True)
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out_dir", type=Path, default=Path("results/tool_ecological"))
    p.add_argument("--max_items", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16")
    args = p.parse_args()

    model_slug = args.model_id.replace("/", "_")
    model_out_dir = args.out_dir / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset...")
    specs = load_itemspecs(args.itemspecs)

    items = []
    for item_id, spec_dict in specs.items():
        if spec_dict.get("suite") != "tool":
            continue
        items.append({
            "item_id": item_id,
            "domain": spec_dict["domain"],
            "difficulty": spec_dict.get("difficulty", "standard"),
            "y_star_evidence": spec_dict.get("y_star_evidence", spec_dict.get("y_star")),
            "y_star_theta": spec_dict.get("y_star_theta"),
            "anchors": spec_dict.get("anchors", {}),
            "spec": spec_dict,
        })

    if args.max_items and args.max_items < len(items):
        rng = np.random.RandomState(args.seed)
        rng.shuffle(items)
        items = items[:args.max_items]

    log.info("Loaded %d items for ecological validation", len(items))
    if not items:
        log.error("No Tool items found")
        sys.exit(1)

    log.info("Loading model %s...", args.model_id)
    runner = HFRunner(
        args.model_id,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
    )

    results_path = model_out_dir / "results_ecological.jsonl"
    log.info("Running ecological inference -> %s", results_path)
    records = run_ecological(runner, items, results_path, max_tokens=args.max_tokens)

    metrics = compute_ecological_summary(records)
    summary_path = model_out_dir / "summary_ecological.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Ecological summary -> %s", summary_path)

    print("\n--- Tool ecological validation summary ---")
    print(f"Model:              {args.model_id}")
    print(f"Items:              {metrics['n_items']}")
    print(f"Parse rate:         {metrics['parse_rate']:.2%}")
    print(f"MAE_control:        {metrics['mae_control']}")
    print(f"UAI_plaus_high:     {metrics['uai_plaus_high']}")
    print(f"TAR_plaus_high:     {metrics['tar_plaus_high']}")
    print(f"Mean tool calls:    {metrics['mean_tool_calls']}")
    print()


if __name__ == "__main__":
    main()
