"""Core evaluation orchestrator for AnchorBench.

Shared logic extracted from the five suite runners:
  - prepare_items: build item list from promptviews + itemspecs
  - parse_response: 3-tier parsing cascade (structured → regex → LLM fallback)
  - build_record: construct standard result dict
  - run_single_stage: External, ICL, RAG, Tool inference loop
  - run_history_two_stage: History's Stage1→parse→Stage2 protocol
  - write_and_summarize: persist results.jsonl + compute unified metrics
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .backends import Backend, HFBackend
from .metrics import compute_unified_metrics, print_summary
from .parsing import (
    LLMFallbackExtractor,
    has_explicit_final_answer,
    looks_incomplete_response,
    parse_answer_int,
    parse_final_answer,
    parse_last_number,
    parse_structured,
    parse_with_fallback,
    parse_xml_answer,
)

log = logging.getLogger(__name__)

CONDITIONS = [
    "control",
    "irrelevant_low",
    "irrelevant_high",
    "plausible_low",
    "plausible_high",
]


def _load_history_jsonl(path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    """Read existing results.jsonl; return records and (item_id, condition) keys."""
    if not path.is_file():
        return [], set()
    records: list[dict] = []
    keys: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            keys.add((rec["item_id"], rec["condition"]))
    return records, keys


def prepare_items(
    views: dict[str, dict[str, dict]],
    specs: dict[str, dict],
    max_items: int | None,
    seed: int,
    conditions: list[str] | None = None,
) -> list[dict]:
    """Build list of items with all required conditions present."""
    cond_set = set(conditions or CONDITIONS)
    items: list[dict] = []
    for item_id, cond_views in views.items():
        # History shards may ship extra conditions (e.g. control_twostage for
        # ablations); evaluate only the requested cond_set.
        if not cond_set.issubset(cond_views.keys()):
            continue
        spec = specs.get(item_id, {})
        item: dict[str, Any] = {
            "item_id": item_id,
            "suite": cond_views["control"]["suite"],
            "domain": cond_views["control"]["domain"],
            "difficulty": spec.get("difficulty", "standard"),
            "y_star_evidence": spec.get(
                "y_star_evidence", spec.get("y_star")
            ),
            "y_star_theta": spec.get("y_star_theta"),
            "anchors": spec.get("anchors", {}),
            "spec": spec,
        }
        for c in cond_set:
            item[c] = cond_views[c]
        items.append(item)

    if max_items and max_items < len(items):
        rng = np.random.RandomState(seed)
        rng.shuffle(items)
        items = items[:max_items]

    return items


def parse_response(
    raw_text: str,
    prompt_text: str,
    *,
    structured_raw: str | None = None,
    use_llm_fallback: bool = False,
    fallback_extractor: LLMFallbackExtractor | None = None,
    clamp: bool = False,
) -> tuple[int | None, bool, str]:
    """Orchestrate the multi-tier parsing cascade.

    Order: structured → xml_tag → final_answer → (incomplete guard) →
    regex → last_number → llm_fallback.

    Truncated outputs (e.g. max_tokens mid-sentence) skip regex/last_number
    so stray numbers from scratch work are not accepted as answers.

    Tool-call JSON outputs (model trying to call a tool instead of
    answering) are rejected at the regex/last_number level so we
    don't accidentally parse evidence values as answers.

    When *clamp* is True, out-of-range integers are clamped to [0, 100]
    instead of rejected.  The strategy name gets a ``_clamped`` suffix
    when clamping actually changed the value.

    Returns (answer, parsed_ok, parse_strategy).
    """
    if structured_raw is not None:
        answer, ok = parse_structured(structured_raw, clamp=clamp)
        if ok:
            strategy = "structured"
            if clamp:
                strict, sok = parse_structured(structured_raw, clamp=False)
                if not sok:
                    strategy = "structured_clamped"
            return answer, True, strategy

    answer, ok = parse_xml_answer(raw_text, clamp=clamp)
    if ok:
        strategy = "xml_tag"
        if clamp:
            strict, sok = parse_xml_answer(raw_text, clamp=False)
            if not sok:
                strategy = "xml_tag_clamped"
        return answer, True, strategy

    answer, ok = parse_final_answer(raw_text, clamp=clamp)
    if ok:
        strategy = "final_answer"
        if clamp:
            strict, sok = parse_final_answer(raw_text, clamp=False)
            if not sok:
                strategy = "final_answer_clamped"
        return answer, True, strategy

    # If final_answer matched a phrase but the value was out of range,
    # the model DID declare an answer — it's just invalid.  Don't let
    # regex/last_number grab an intermediate CoT number instead.
    if has_explicit_final_answer(raw_text):
        log.debug("Final-answer phrase found but value out of range; skipping regex/last_number")
        if use_llm_fallback and fallback_extractor is not None:
            answer, ok = fallback_extractor.try_extract(raw_text)
            if ok:
                return answer, True, "llm_fallback"
        return None, False, "failed"

    if looks_incomplete_response(raw_text or ""):
        log.debug("Response looks truncated; skipping regex/last_number")
        if use_llm_fallback and fallback_extractor is not None:
            answer, ok = fallback_extractor.try_extract(raw_text)
            if ok:
                return answer, True, "llm_fallback"
        return None, False, "failed"

    answer, ok = parse_answer_int(raw_text, prompt_text, clamp=clamp)
    if ok:
        strategy = "regex"
        if clamp:
            strict, sok = parse_answer_int(raw_text, prompt_text, clamp=False)
            if not sok:
                strategy = "regex_clamped"
        return answer, True, strategy

    answer, ok = parse_last_number(raw_text, clamp=clamp)
    if ok:
        strategy = "last_number"
        if clamp:
            strict, sok = parse_last_number(raw_text, clamp=False)
            if not sok:
                strategy = "last_number_clamped"
        return answer, True, strategy

    if use_llm_fallback and fallback_extractor is not None:
        answer, ok = fallback_extractor.try_extract(raw_text)
        if ok:
            log.debug("LLM fallback extracted %d", answer)
            return answer, True, "llm_fallback"

    return None, False, "failed"


def build_record(
    model_id: str,
    item: dict,
    cond: str,
    pv: dict,
    answer: int | None,
    parsed_ok: bool,
    parse_strategy: str,
    raw_text: str,
    **extras: Any,
) -> dict:
    """Construct a standard result record dict."""
    rec = {
        "model_id": model_id,
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
        "raw_text": raw_text,
    }
    rec.update(extras)
    return rec


def run_single_stage(
    backend: HFBackend | Backend,
    items: list[dict],
    out_path: Path,
    *,
    conditions: list[str] | None = None,
    max_tokens: int = 512,
    batch_size: int = 1,
    prompt_suffix: str = "",
    use_llm_fallback: bool = False,
    fallback_extractor: LLMFallbackExtractor | None = None,
    prompt_fn: Callable | None = None,
    tool_generate_fn: Callable | None = None,
) -> list[dict]:
    """Run single-stage inference for all items x conditions.

    If batch_size > 1 and prompt_fn is None, uses generate_batch.
    If tool_generate_fn is provided, uses that for generation instead.
    If prompt_fn is provided, it is called as prompt_fn(item, cond, pv)
    and may return either a string (plaintext) or a list of dicts
    (chat messages for tool_generate_fn).
    """
    conds = conditions or CONDITIONS
    use_batched = batch_size > 1 and prompt_fn is None and tool_generate_fn is None

    if use_batched:
        return _run_batched(
            backend, items, out_path,
            conds=conds, max_tokens=max_tokens,
            batch_size=batch_size, prompt_suffix=prompt_suffix,
            use_llm_fallback=use_llm_fallback,
            fallback_extractor=fallback_extractor,
        )

    records: list[dict] = []
    total = len(items) * len(conds)
    done = 0

    with open(out_path, "w", encoding="utf-8") as fh:
        for item in items:
            for cond in conds:
                pv = item[cond]

                if tool_generate_fn is not None:
                    raw = tool_generate_fn(item, cond, pv)
                elif prompt_fn is not None:
                    prompt_or_msgs = prompt_fn(item, cond, pv)
                    if isinstance(prompt_or_msgs, list):
                        raw = backend.generate_chat(
                            prompt_or_msgs, max_tokens=max_tokens,
                        )
                    else:
                        raw = backend.generate(
                            prompt_or_msgs + prompt_suffix,
                            max_tokens=max_tokens,
                        )
                else:
                    prompt = pv["prompt_text"] + prompt_suffix
                    try:
                        raw = backend.generate(
                            prompt, max_tokens=max_tokens,
                        )
                    except Exception as e:
                        log.warning(
                            "Failed %s/%s: %s", item["item_id"], cond, e,
                        )
                        raw = f"ERROR: {e}"

                prompt_text = pv.get("prompt_text", "")
                answer, parsed_ok, strategy = parse_response(
                    raw, prompt_text,
                    use_llm_fallback=use_llm_fallback,
                    fallback_extractor=fallback_extractor,
                )

                rec = build_record(
                    backend.model_id, item, cond, pv,
                    answer, parsed_ok, strategy, raw,
                )
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

                done += 1
                if done % 50 == 0:
                    log.info("Inference %d/%d", done, total)

    return records


def _run_batched(
    backend: HFBackend | Backend,
    items: list[dict],
    out_path: Path,
    *,
    conds: list[str],
    max_tokens: int,
    batch_size: int,
    prompt_suffix: str,
    use_llm_fallback: bool,
    fallback_extractor: LLMFallbackExtractor | None,
) -> list[dict]:
    """Batched plaintext inference."""
    task_list: list[tuple[int, str, dict]] = []
    for item_idx, item in enumerate(items):
        for cond in conds:
            pv = item[cond]
            task_list.append((item_idx, cond, pv))

    prompts = [pv["prompt_text"] + prompt_suffix for _, _, pv in task_list]

    log.info(
        "Running batched inference: %d prompts, batch_size=%d",
        len(prompts), batch_size,
    )
    raw_outputs = backend.generate_batch(
        prompts, max_tokens=max_tokens,
        temperature=0.0, batch_size=batch_size,
    )

    records: list[dict] = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for (item_idx, cond, pv), raw in zip(task_list, raw_outputs):
            item = items[item_idx]
            prompt_text = pv.get("prompt_text", "")

            answer, parsed_ok, strategy = parse_response(
                raw, prompt_text,
                use_llm_fallback=use_llm_fallback,
                fallback_extractor=fallback_extractor,
            )

            rec = build_record(
                backend.model_id, item, cond, pv,
                answer, parsed_ok, strategy, raw,
            )
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    return records


def run_history_two_stage(
    backend: HFBackend | Backend,
    items: list[dict],
    out_path: Path,
    *,
    conditions: list[str] | None = None,
    max_tokens: int = 512,
    prompt_suffix: str = "",
    use_llm_fallback: bool = False,
    fallback_extractor: LLMFallbackExtractor | None = None,
    resume: bool = False,
) -> list[dict]:
    """History suite: single-stage for ``control``; two-stage for all other conditions.

    ``control_twostage`` uses the same two-turn chat protocol as plausible/irrelevant
    but Stage~1 is qualitative, so ``anchor_value`` is recorded as None (fair baseline).
    """
    conds = conditions or CONDITIONS
    total = len(items) * len(conds)
    done = 0

    if resume and out_path.is_file():
        records, done_keys = _load_history_jsonl(out_path)
        file_mode = "a" if records else "w"
        if records:
            log.info(
                "Resume: %d lines in %s, skipping completed (item, condition) pairs",
                len(records), out_path,
            )
    else:
        records, done_keys = [], set()
        file_mode = "w"

    done = len(done_keys)

    with open(out_path, file_mode, encoding="utf-8") as fh:
        for item in items:
            for cond in conds:
                if (item["item_id"], cond) in done_keys:
                    continue
                pv = item[cond]
                comp = pv.get("prompt_components", {})

                if cond == "control":
                    prompt = pv["prompt_text"] + prompt_suffix
                    try:
                        raw = backend.generate(prompt, max_tokens=max_tokens)
                    except Exception as e:
                        log.warning(
                            "Failed %s/%s: %s", item["item_id"], cond, e,
                        )
                        raw = f"ERROR: {e}"

                    answer, parsed_ok, strategy = parse_response(
                        raw, prompt,
                        use_llm_fallback=use_llm_fallback,
                        fallback_extractor=fallback_extractor,
                    )
                    rec = build_record(
                        backend.model_id, item, cond, pv,
                        answer, parsed_ok, strategy, raw,
                        anchor_value=None,
                        stage1_answer=None,
                        stage1_raw_text=None,
                    )
                else:
                    stage1_msg = comp.get("stage1_user_message", "")
                    stage2_msg = comp.get("stage2_user_message", "")

                    if not stage1_msg or not stage2_msg:
                        rec = build_record(
                            backend.model_id, item, cond, pv,
                            None, False, "failed",
                            "MISSING_STAGE_COMPONENTS",
                            anchor_value=None,
                            stage1_answer=None,
                            stage1_raw_text=None,
                        )
                        records.append(rec)
                        fh.write(
                            json.dumps(rec, ensure_ascii=False) + "\n"
                        )
                        fh.flush()
                        done += 1
                        continue

                    stage1_prompt = stage1_msg + prompt_suffix
                    try:
                        stage1_raw = backend.generate(
                            stage1_prompt, max_tokens=max_tokens,
                        )
                    except Exception as e:
                        log.warning(
                            "Failed %s/%s stage1: %s",
                            item["item_id"], cond, e,
                        )
                        stage1_raw = f"ERROR: {e}"

                    stage1_answer, _, _ = parse_response(
                        stage1_raw, stage1_prompt,
                        use_llm_fallback=use_llm_fallback,
                        fallback_extractor=fallback_extractor,
                    )

                    use_stage1_as_anchor = cond.startswith(
                        ("plausible_", "irrelevant_"),
                    )
                    anchor_for_record = (
                        stage1_answer if use_stage1_as_anchor else None
                    )

                    messages = [
                        {"role": "user", "content": stage1_msg},
                        {"role": "assistant", "content": stage1_raw},
                        {
                            "role": "user",
                            "content": stage2_msg + prompt_suffix,
                        },
                    ]
                    try:
                        stage2_raw = backend.generate_chat(
                            messages, max_tokens=max_tokens,
                        )
                    except Exception as e:
                        log.warning(
                            "Failed %s/%s stage2: %s",
                            item["item_id"], cond, e,
                        )
                        stage2_raw = f"ERROR: {e}"

                    answer, parsed_ok, strategy = parse_response(
                        stage2_raw, messages[-1]["content"],
                        use_llm_fallback=use_llm_fallback,
                        fallback_extractor=fallback_extractor,
                    )

                    rec = build_record(
                        backend.model_id, item, cond, pv,
                        answer, parsed_ok, strategy, stage2_raw,
                        anchor_value=anchor_for_record,
                        stage1_answer=stage1_answer,
                        stage1_raw_text=stage1_raw,
                    )

                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                if done % 50 == 0:
                    log.info("Inference %d/%d", done, total)

    return records


def write_and_summarize(
    records: list[dict],
    out_dir: Path,
    label: str = "",
    epsilon: float = 3.0,
    *,
    baseline_condition: str = "control",
) -> dict:
    """Compute unified metrics, write summary.json, print summary.

    The results.jsonl is assumed to already be written by the run_* function.
    """
    strategy_counts = Counter(r.get("parse_strategy", "unknown") for r in records)
    log.info("Parse strategy breakdown: %s", dict(strategy_counts))

    metrics = compute_unified_metrics(
        records, epsilon=epsilon, baseline_condition=baseline_condition,
    )
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Summary -> %s", summary_path)

    print_summary(metrics, label=label)
    return metrics
