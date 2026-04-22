"""Shared helpers for AnchorBench suite runners.

Centralises argparse, backend construction, and suffix building
that were previously duplicated across run_external.py, run_rag.py,
run_icl.py, run_tool.py, and run_history.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Union

from anchorbench_eval.backends import HFBackend, VLLMBackend
from anchorbench_eval.parsing import LLMFallbackExtractor, XML_TAG_INSTRUCTION

log = logging.getLogger(__name__)

Backend = Union[HFBackend, VLLMBackend]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard set of CLI arguments shared by all suite runners."""
    parser.add_argument("--promptviews", type=Path, required=True)
    parser.add_argument("--itemspecs", type=Path, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--request_final_line", action="store_true")
    parser.add_argument("--request_reasoning", action="store_true")
    parser.add_argument(
        "--request_xml_answer", action="store_true",
        help="Append XML tag instruction: wrap answer in <answer>N</answer>",
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    parser.add_argument("--llm_fallback", action="store_true")
    parser.add_argument("--fallback_model", type=str, default=None)
    parser.add_argument("--fallback_device", type=str, default="auto")
    parser.add_argument("--structured", action="store_true")

    parser.add_argument(
        "--backend", type=str, choices=["hf", "vllm"], default="hf",
        help="Inference backend: hf (Transformers) or vllm",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM: number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM: fraction of GPU memory to use (0-1)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM: max sequence length")


def make_backend(args: argparse.Namespace) -> Backend:
    """Build HF or vLLM backend from parsed CLI arguments."""
    log.info("Loading model %s (%s backend)...", args.model_id, args.backend)
    if args.backend == "vllm":
        return VLLMBackend(
            args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=True,
        )
    return HFBackend(
        args.model_id,
        device=args.device,
        device_map=getattr(args, "device_map", None),
        dtype=args.dtype,
    )


def build_suffix(args: argparse.Namespace) -> str:
    """Construct the prompt suffix from the request_* CLI flags."""
    suffix = ""
    if getattr(args, "request_reasoning", False):
        suffix += (
            "\n\nShow your reasoning step by step, then give your final "
            "numeric estimate (0-100) at the end."
        )
    if getattr(args, "request_final_line", False):
        suffix += (
            "\n\nPut your final numeric estimate (0-100) on the last line "
            "only, in this exact form: Answer: [number]"
        )
    if getattr(args, "request_xml_answer", False):
        suffix += XML_TAG_INSTRUCTION
    return suffix


def make_fallback(args: argparse.Namespace) -> LLMFallbackExtractor | None:
    """Optionally build the LLM fallback extractor."""
    if not getattr(args, "llm_fallback", False):
        return None
    fb_model = args.fallback_model or args.model_id
    fb_device = getattr(args, "fallback_device", "auto")
    log.info("Loading LLM fallback extractor: %s on %s", fb_model, fb_device)
    return LLMFallbackExtractor(
        model_id=fb_model, device=fb_device, dtype="bfloat16",
    )


def model_output_dir(args: argparse.Namespace) -> Path:
    """Return ``out_dir / model_slug`` and create the directory."""
    model_slug = args.model_id.replace("/", "_")
    d = args.out_dir / model_slug
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- Result discovery (used by recompute_all_unified.py and friends) ----

from anchorbench_eval.constants import MODEL_SHORT, SUITES  # noqa: E402


def discover_results(
    results_dir: Path,
    suites: tuple[str, ...] = SUITES,
) -> list[tuple[str, str, str, Path]]:
    """Find all results.jsonl and return (suite, model_slug, short_name, path)."""
    found = []
    for suite in suites:
        suite_dir = results_dir / suite
        if not suite_dir.is_dir():
            continue

        for p in sorted(suite_dir.rglob("results.jsonl")):
            rel = p.relative_to(suite_dir)
            parts = list(rel.parts)

            if len(parts) == 1:
                model_slug = suite_dir.name
                if model_slug in suites:
                    continue
            elif len(parts) in (2, 3):
                model_slug = parts[0]
            else:
                continue

            short = MODEL_SHORT.get(model_slug, model_slug)
            found.append((suite, model_slug, short, p))

    return found


def fmt(v: float | None, decimals: int = 2) -> str:
    """Format a float for table display (``---`` if None)."""
    if v is None:
        return "---"
    return f"{v:.{decimals}f}"


def fmt_pct(v: float | None) -> str:
    """Format a float as percentage for table display."""
    if v is None:
        return "---"
    return f"{v * 100:.1f}%"
