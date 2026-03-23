#!/usr/bin/env python3
"""Test XML-tag extraction by re-running inference on a subset of items.

Runs the same prompts with the XML tag instruction appended, then parses
with the XML tag extractor. Compares against the original regex-parsed results.

Usage:
    PYTHONPATH=src python scripts/eval/test_xml_tag_extraction.py \
        --results_dir results/smoke_test \
        --suite external \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --backend vllm \
        --max_items 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anchorbench_eval.backends import HFBackend, VLLMBackend
from anchorbench_eval.io import load_itemspecs, load_promptviews
from anchorbench_eval.parsing import (
    XML_TAG_INSTRUCTION,
    parse_answer_int,
    parse_final_answer,
    parse_last_number,
    parse_xml_answer,
)

log = logging.getLogger(__name__)

SUITE_DATASETS = {
    "external": ("datasets/anchorbench_external_core", "promptviews_core.jsonl"),
    "icl": ("datasets/anchorbench_icl_core", "promptviews_core.jsonl"),
    "rag": ("datasets/anchorbench_rag_core", "promptviews_core.jsonl"),
}

CONDITIONS = [
    "control",
    "irrelevant_low",
    "irrelevant_high",
    "plausible_low",
    "plausible_high",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Test XML tag extraction method")
    p.add_argument("--suite", type=str, default="external",
                   choices=list(SUITE_DATASETS.keys()))
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--backend", type=str, choices=["hf", "vllm"], default="vllm")
    p.add_argument("--max_items", type=int, default=5)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--out_dir", type=Path, default=Path("results/xml_tag_test"))
    args = p.parse_args()

    dataset_dir, pv_name = SUITE_DATASETS[args.suite]
    pv_path = Path(dataset_dir) / (pv_name or "promptviews_core.jsonl")
    is_path = Path(dataset_dir) / "itemspecs.jsonl"

    if not pv_path.exists():
        log.error("Dataset not found: %s", pv_path)
        sys.exit(1)

    views = load_promptviews(pv_path)
    specs = load_itemspecs(is_path)

    item_ids = sorted(views.keys())[:args.max_items]
    log.info("Testing %d items from %s suite", len(item_ids), args.suite)

    log.info("Loading model %s (%s backend)...", args.model_id, args.backend)
    if args.backend == "vllm":
        backend = VLLMBackend(
            args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        backend = HFBackend(args.model_id, device=args.device)

    results = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for item_id in item_ids:
        cond_views = views.get(item_id, {})
        spec = specs.get(item_id, {})

        for cond in CONDITIONS:
            pv = cond_views.get(cond)
            if pv is None:
                continue

            prompt_text = pv["prompt_text"]

            raw_baseline = backend.generate(
                prompt_text, max_tokens=args.max_tokens,
            )

            raw_xml = backend.generate(
                prompt_text + XML_TAG_INSTRUCTION,
                max_tokens=args.max_tokens,
            )

            regex_ans, regex_ok = parse_answer_int(raw_baseline, prompt_text)
            last_ans, last_ok = parse_last_number(raw_baseline)
            final_ans, final_ok = parse_final_answer(raw_baseline)
            xml_ans, xml_ok = parse_xml_answer(raw_xml)

            regex_xml_ans, regex_xml_ok = parse_answer_int(raw_xml, prompt_text)
            last_xml_ans, last_xml_ok = parse_last_number(raw_xml)

            rec = {
                "item_id": item_id,
                "condition": cond,
                "y_star_evidence": spec.get("y_star_evidence"),
                "y_star_theta": spec.get("y_star_theta"),
                "baseline": {
                    "regex": {"answer": regex_ans, "ok": regex_ok},
                    "last_number": {"answer": last_ans, "ok": last_ok},
                    "final_answer": {"answer": final_ans, "ok": final_ok},
                    "raw_len": len(raw_baseline),
                    "raw_text": raw_baseline,
                },
                "xml_prompted": {
                    "xml_tag": {"answer": xml_ans, "ok": xml_ok},
                    "regex_on_xml": {"answer": regex_xml_ans, "ok": regex_xml_ok},
                    "last_number_on_xml": {"answer": last_xml_ans, "ok": last_xml_ok},
                    "raw_len": len(raw_xml),
                    "raw_text": raw_xml,
                },
            }
            results.append(rec)

            log.info(
                "%s/%s: regex=%s, last=%s, final=%s | xml_tag=%s",
                item_id, cond,
                regex_ans, last_ans, final_ans, xml_ans,
            )

    out_path = args.out_dir / f"xml_test_{args.suite}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    baseline_regex_rate = sum(1 for r in results if r["baseline"]["regex"]["ok"]) / n
    baseline_last_rate = sum(1 for r in results if r["baseline"]["last_number"]["ok"]) / n
    baseline_final_rate = sum(1 for r in results if r["baseline"]["final_answer"]["ok"]) / n
    xml_tag_rate = sum(1 for r in results if r["xml_prompted"]["xml_tag"]["ok"]) / n
    regex_on_xml_rate = sum(1 for r in results if r["xml_prompted"]["regex_on_xml"]["ok"]) / n

    print(f"\n{'='*60}")
    print(f"XML Tag Extraction Test: {args.suite} suite, {args.model_id}")
    print(f"{'='*60}")
    print(f"Items tested: {n}")
    print(f"\nBaseline (no XML instruction):")
    print(f"  regex:        {baseline_regex_rate:.1%}")
    print(f"  last_number:  {baseline_last_rate:.1%}")
    print(f"  final_answer: {baseline_final_rate:.1%}")
    print(f"\nWith XML tag instruction:")
    print(f"  xml_tag:      {xml_tag_rate:.1%}")
    print(f"  regex on XML: {regex_on_xml_rate:.1%}")

    avg_baseline_len = sum(r["baseline"]["raw_len"] for r in results) / n
    avg_xml_len = sum(r["xml_prompted"]["raw_len"] for r in results) / n
    print(f"\nAvg output length: baseline={avg_baseline_len:.0f}, xml={avg_xml_len:.0f}")

    # Agreement
    agree_xml_regex = sum(
        1 for r in results
        if r["xml_prompted"]["xml_tag"]["ok"] and r["baseline"]["regex"]["ok"]
        and r["xml_prompted"]["xml_tag"]["answer"] == r["baseline"]["regex"]["answer"]
    )
    both_ok = sum(
        1 for r in results
        if r["xml_prompted"]["xml_tag"]["ok"] and r["baseline"]["regex"]["ok"]
    )
    if both_ok:
        print(f"\nXML vs Regex agreement (both parsed): {agree_xml_regex}/{both_ok} ({agree_xml_regex/both_ok:.1%})")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
