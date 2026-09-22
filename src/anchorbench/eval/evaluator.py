"""Core evaluation orchestrator for AnchorBench.

Shared logic extracted from the five suite runners:
  - prepare_items: build item list from promptviews + itemspecs
  - parse_response: 3-tier parsing cascade (structured → regex → LLM fallback)
  - build_record: construct standard result dict
  - run_single_stage: one batched round trip for External, ICL, RAG, Tool,
    and every hosted-API run (any :class:`Backend`)
  - run_history_two_stage: History's Stage1→parse→Stage2 protocol, batched
  - write_and_summarize: persist results.jsonl + compute unified metrics
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from anchorbench import __version__

from .backends import Backend
from .metrics import compute_unified_metrics, print_summary
from .parsing import (
    LLMFallbackExtractor,
    has_explicit_final_answer,
    looks_incomplete_response,
    parse_answer_int,
    parse_final_answer,
    parse_last_number,
    parse_structured,
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


Task = tuple[int, str, dict]
"""(index into ``items``, condition, promptview) -- one prompt to evaluate."""


def _tasks(items: list[dict], conds: list[str]) -> list[Task]:
    return [(i, cond, item[cond]) for i, item in enumerate(items) for cond in conds]


def _generate_or_error(generate: Callable[[], list[str]], n: int, label: str) -> list[str]:
    """Run one backend call; on failure record the error in every slot.

    A backend exception used to abort a batched run halfway through while
    the sequential path recorded ``ERROR: ...`` per prompt. Every prompt now
    gets a record either way, and a failed call is loud in the summary
    because the parse rate collapses.
    """
    try:
        out = generate()
    except Exception as e:  # noqa: BLE001 -- any backend failure is recorded, not raised
        log.error("%s: backend call failed (%s); recording ERROR for %d prompts", label, e, n)
        return [f"ERROR: {e}"] * n
    if len(out) != n:
        log.error("%s: backend returned %d outputs for %d prompts", label, len(out), n)
        return [f"ERROR: backend returned {len(out)} outputs for {n} prompts"] * n
    return out


def _usage_for(backend: Any, n: int) -> list[dict] | None:
    """Per-request token usage from the last backend call, when it reports one."""
    usage = getattr(backend, "last_usage", None)
    return usage if isinstance(usage, list) and len(usage) == n else None


def _sum_usage(*parts: dict | None) -> dict:
    total: dict[str, int] = {}
    for part in parts:
        for k, v in (part or {}).items():
            if isinstance(v, (int, float)):
                total[k] = total.get(k, 0) + v
    return total


def run_single_stage(
    backend: Backend,
    items: list[dict],
    out_path: Path,
    *,
    conditions: list[str] | None = None,
    max_tokens: int = 512,
    batch_size: int = 16,
    prompt_suffix: str = "",
    use_llm_fallback: bool = False,
    fallback_extractor: LLMFallbackExtractor | None = None,
    temperature: float = 0.0,
    messages_fn: Callable[[dict, str, dict], list[dict]] | None = None,
    tools: list[dict] | None = None,
    record_extras: dict[str, Any] | None = None,
) -> list[dict]:
    """Evaluate every item x condition in one round trip to the backend.

    This is the loop behind External, ICL, RAG, Tool, the sampling and
    mitigation probes and every hosted-API run. By default each prompt is
    ``pv["prompt_text"] + prompt_suffix`` sent as a single user turn through
    ``backend.generate_batch``. With ``messages_fn`` the prompt is instead the
    chat message list it returns for ``(item, condition, promptview)``, sent
    through ``generate_batch_tool`` when ``tools`` is given (the Tool suite's
    native tool-call rendering) and ``generate_chat_batch`` otherwise.

    ``record_extras`` are copied into every record (e.g. ``{"temperature":
    0.7, "seed_idx": 2}``). If the backend reports per-request token usage in
    ``last_usage``, it is stored under ``api_usage``.

    Records are written to ``out_path`` in items x conditions order and also
    returned.
    """
    conds = conditions or CONDITIONS
    tasks = _tasks(items, conds)
    extras = dict(record_extras or {})
    label = f"{backend.model_id} single-stage"

    if messages_fn is not None:
        messages_list = [messages_fn(items[i], cond, pv) for i, cond, pv in tasks]
        if tools is not None:
            def generate() -> list[str]:
                return backend.generate_batch_tool(
                    messages_list, tools=tools, max_tokens=max_tokens,
                    temperature=temperature, batch_size=batch_size,
                )
        else:
            def generate() -> list[str]:
                return backend.generate_chat_batch(
                    messages_list, max_tokens=max_tokens,
                    temperature=temperature, batch_size=batch_size,
                )
    else:
        prompts = [pv["prompt_text"] + prompt_suffix for _, _, pv in tasks]

        def generate() -> list[str]:
            return backend.generate_batch(
                prompts, max_tokens=max_tokens,
                temperature=temperature, batch_size=batch_size,
            )

    log.info("%s: %d prompts (batch_size=%d)", label, len(tasks), batch_size)
    raws = _generate_or_error(generate, len(tasks), label)
    usage = _usage_for(backend, len(tasks))

    records: list[dict] = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for k, ((i, cond, pv), raw) in enumerate(zip(tasks, raws)):
            answer, parsed_ok, strategy = parse_response(
                raw, pv.get("prompt_text", ""),
                use_llm_fallback=use_llm_fallback,
                fallback_extractor=fallback_extractor,
            )
            rec = build_record(
                backend.model_id, items[i], cond, pv,
                answer, parsed_ok, strategy, raw, **extras,
            )
            if usage is not None:
                rec["api_usage"] = usage[k]
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
    log.info("%s: wrote %d records -> %s", label, len(records), out_path)
    return records


def flatten_conversation(stage1_prompt: str, stage1_raw: str, stage2_prompt: str) -> str:
    """The single-message rendering of a two-turn history.

    This is how the published API-tier History cells were run: the model saw
    its own Stage-1 turn quoted inside one user message rather than as a real
    assistant turn (docs/RECONCILIATION.md D9). Kept so those numbers stay
    reproducible with ``chat_format="flat"``.
    """
    return (
        f"Previous conversation:\n"
        f"User: {stage1_prompt}\n"
        f"Assistant: {stage1_raw}\n\n"
        f"User: {stage2_prompt}"
    )


def run_history_two_stage(
    backend: Backend,
    items: list[dict],
    out_path: Path,
    *,
    conditions: list[str] | None = None,
    max_tokens: int = 512,
    batch_size: int = 16,
    prompt_suffix: str = "",
    use_llm_fallback: bool = False,
    fallback_extractor: LLMFallbackExtractor | None = None,
    resume: bool = False,
    temperature: float = 0.0,
    chat_format: str = "messages",
    record_extras: dict[str, Any] | None = None,
) -> list[dict]:
    """History suite: single turn for ``control``, two turns for everything else.

    Stage 1 sends ``stage1_user_message`` and parses the model's estimate;
    Stage 2 sends ``stage2_user_message`` with the Stage-1 exchange as
    context and is what gets scored. For ``plausible_*`` / ``irrelevant_*``
    the Stage-1 estimate is the anchor and is recorded as ``anchor_value``;
    for ``control_twostage`` Stage 1 is qualitative and ``anchor_value`` is
    None.

    ``chat_format="messages"`` sends Stage 2 as a real three-turn chat;
    ``"flat"`` quotes the exchange inside one user message
    (:func:`flatten_conversation`). Both stages run as batches, so a hosted
    backend fans the requests out concurrently. With ``resume``, (item,
    condition) pairs already in ``out_path`` are kept and skipped.
    """
    if chat_format not in ("messages", "flat"):
        raise ValueError(f"chat_format must be 'messages' or 'flat', got {chat_format!r}")
    conds = conditions or CONDITIONS
    extras = dict(record_extras or {})
    label = f"{backend.model_id} history"

    if resume and out_path.is_file():
        records, done_keys = _load_history_jsonl(out_path)
        file_mode = "a" if records else "w"
        if records:
            log.info("Resume: %d lines in %s, skipping completed (item, condition) pairs",
                     len(records), out_path)
    else:
        records, done_keys = [], set()
        file_mode = "w"

    tasks = [t for t in _tasks(items, conds) if (items[t[0]]["item_id"], t[1]) not in done_keys]

    control: list[Task] = []
    two_stage: list[Task] = []
    missing: list[Task] = []
    for task in tasks:
        _, cond, pv = task
        comp = pv.get("prompt_components", {})
        if cond == "control":
            control.append(task)
        elif comp.get("stage1_user_message") and comp.get("stage2_user_message"):
            two_stage.append(task)
        else:
            missing.append(task)

    def gen_batch(prompts: list[str]) -> list[str]:
        return backend.generate_batch(
            prompts, max_tokens=max_tokens, temperature=temperature, batch_size=batch_size,
        )

    # -- control: one turn -------------------------------------------------
    ctrl_prompts = [pv["prompt_text"] + prompt_suffix for _, _, pv in control]
    log.info("%s: %d control prompts", label, len(ctrl_prompts))
    ctrl_raws = _generate_or_error(lambda: gen_batch(ctrl_prompts), len(control), label) if control else []
    ctrl_usage = _usage_for(backend, len(control)) if control else None

    # -- stage 1 -------------------------------------------------------------
    s1_msgs = [pv["prompt_components"]["stage1_user_message"] for _, _, pv in two_stage]
    s2_msgs = [pv["prompt_components"]["stage2_user_message"] for _, _, pv in two_stage]
    s1_prompts = [m + prompt_suffix for m in s1_msgs]
    log.info("%s: %d two-stage items, stage 1", label, len(two_stage))
    s1_raws = _generate_or_error(lambda: gen_batch(s1_prompts), len(two_stage), label) if two_stage else []
    s1_usage = _usage_for(backend, len(two_stage)) if two_stage else None
    s1_answers = [
        parse_response(raw, prompt, use_llm_fallback=use_llm_fallback,
                       fallback_extractor=fallback_extractor)[0]
        for raw, prompt in zip(s1_raws, s1_prompts)
    ]

    # -- stage 2 -------------------------------------------------------------
    s2_prompts = [m + prompt_suffix for m in s2_msgs]
    if chat_format == "messages":
        conversations = [
            [
                {"role": "user", "content": s1_msg},
                {"role": "assistant", "content": s1_raw},
                {"role": "user", "content": s2_prompt},
            ]
            for s1_msg, s1_raw, s2_prompt in zip(s1_msgs, s1_raws, s2_prompts)
        ]
        s2_parse_text = s2_prompts

        def gen_stage2() -> list[str]:
            return backend.generate_chat_batch(
                conversations, max_tokens=max_tokens, temperature=temperature, batch_size=batch_size,
            )
    else:
        flat = [
            flatten_conversation(s1_prompt, s1_raw, s2_prompt)
            for s1_prompt, s1_raw, s2_prompt in zip(s1_prompts, s1_raws, s2_prompts)
        ]
        # The published API runs parsed against the promptview's own text.
        s2_parse_text = [pv.get("prompt_text", "") for _, _, pv in two_stage]

        def gen_stage2() -> list[str]:
            return gen_batch(flat)

    log.info("%s: stage 2 (%s)", label, chat_format)
    s2_raws = _generate_or_error(gen_stage2, len(two_stage), label) if two_stage else []
    s2_usage = _usage_for(backend, len(two_stage)) if two_stage else None

    # -- records, in items x conditions order --------------------------------
    built: dict[tuple[int, str], dict] = {}
    for k, ((i, cond, pv), raw) in enumerate(zip(control, ctrl_raws)):
        answer, parsed_ok, strategy = parse_response(
            raw, ctrl_prompts[k], use_llm_fallback=use_llm_fallback,
            fallback_extractor=fallback_extractor,
        )
        rec = build_record(
            backend.model_id, items[i], cond, pv, answer, parsed_ok, strategy, raw,
            anchor_value=None, stage1_answer=None, stage1_raw_text=None, **extras,
        )
        if ctrl_usage is not None:
            rec["api_usage"] = ctrl_usage[k]
        built[(i, cond)] = rec
    for k, ((i, cond, pv), s2_raw) in enumerate(zip(two_stage, s2_raws)):
        answer, parsed_ok, strategy = parse_response(
            s2_raw, s2_parse_text[k], use_llm_fallback=use_llm_fallback,
            fallback_extractor=fallback_extractor,
        )
        stage1_answer = s1_answers[k]
        anchor = stage1_answer if cond.startswith(("plausible_", "irrelevant_")) else None
        rec = build_record(
            backend.model_id, items[i], cond, pv, answer, parsed_ok, strategy, s2_raw,
            anchor_value=anchor, stage1_answer=stage1_answer, stage1_raw_text=s1_raws[k],
            **extras,
        )
        if s1_usage is not None and s2_usage is not None:
            rec["api_usage"] = _sum_usage(s1_usage[k], s2_usage[k])
        built[(i, cond)] = rec
    for i, cond, pv in missing:
        built[(i, cond)] = build_record(
            backend.model_id, items[i], cond, pv, None, False, "failed",
            "MISSING_STAGE_COMPONENTS",
            anchor_value=None, stage1_answer=None, stage1_raw_text=None, **extras,
        )

    with open(out_path, file_mode, encoding="utf-8") as fh:
        for i, cond, _pv in tasks:
            rec = built[(i, cond)]
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
    log.info("%s: wrote %d records -> %s", label, len(tasks), out_path)
    return records


def _git_hash() -> str | None:
    """Return short git commit hash, or None if not in a git repo."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def write_and_summarize(
    records: list[dict],
    out_dir: Path,
    label: str = "",
    epsilon: float = 3.0,
    *,
    baseline_condition: str = "control",
    run_metadata: dict[str, Any] | None = None,
) -> dict:
    """Compute unified metrics, write summary.json, print summary.

    The results.jsonl is assumed to already be written by the run_* function.
    If *run_metadata* is provided (model_id, backend, args, etc.), it is
    saved to ``run_config.json`` alongside the summary for traceability.
    """
    import datetime

    strategy_counts = Counter(r.get("parse_strategy", "unknown") for r in records)
    log.info("Parse strategy breakdown: %s", dict(strategy_counts))

    metrics = compute_unified_metrics(
        records, epsilon=epsilon, baseline_condition=baseline_condition,
    )
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Summary -> %s", summary_path)

    if run_metadata is not None:
        config_path = out_dir / "run_config.json"
        meta = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_commit": _git_hash(),
            "package_version": __version__,
            **run_metadata,
        }
        with open(config_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Run config -> %s", config_path)

    print_summary(metrics, label=label)
    return metrics
