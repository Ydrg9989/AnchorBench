"""Unified mitigation runner: HF local + OpenRouter API backends."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .strategies import (
    ALL_STRATEGIES,
    apply_existing_mitigation,
    counterfactual_aggregate,
    counterfactual_prompts,
    get_max_tokens,
    late_binding_pass1,
    late_binding_pass2,
    parse_cot_answer,
    selfhelp_answer_prompt,
    selfhelp_rewrite_prompt,
)

log = logging.getLogger(__name__)

_INT_PAT = re.compile(r"(?<![\d.])-?\d+(?!\d)(?!\.\d)")

# Prefer numbers that look like a final answer (e.g. "is 79", "Answer: 79", "= 79")
_COT_NUM_PAT = re.compile(
    r"(?:is|=|:\s*)\s*(\d{1,3})\s*(?:[.\s\n]|$)",
    re.IGNORECASE,
)

_EXTRACT_PROMPT = (
    "Extract the single final numeric answer (integer 0-100) from the text below.\n"
    "Return ONLY the integer on its own line. If no clear answer, return NONE.\n\n"
    "Text: {completion}\n\nAnswer:"
)


class LLMFallbackExtractor:
    """Loads a small HF model once for fallback answer extraction."""

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
                 device: str = "auto", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = getattr(torch, dtype, "auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype,
            device_map=device, trust_remote_code=True,
        )
        self.model.eval()
        self._device = self.model.device
        log.info("Loaded fallback extractor %s on %s", model_id, self._device)

    def try_extract(self, raw_text: str) -> tuple[int | None, bool]:
        """Ask the small LLM to extract the integer answer from raw_text."""
        import torch

        if not raw_text or len(raw_text.strip()) < 2:
            return None, False

        cleaned = raw_text.strip()
        if len(cleaned) > 1500:
            cleaned = cleaned[:200] + "\n...\n" + cleaned[-1200:]
        prompt = _EXTRACT_PROMPT.format(completion=cleaned)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=8, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

        if response.upper() == "NONE":
            return None, False
        m = re.search(r"\b(\d+)\b", response)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return v, True
        return None, False


def parse_answer_int(raw_text: str, prompt_text: str = "") -> tuple[int | None, bool]:
    """Extract a single integer in [0, 100] from model output.

    The regex allows integers followed by sentence-ending periods (75.)
    but still rejects decimals (3.14 won't match as 3).
    When the response contains multiple candidate integers, we try to
    pick the final answer by: (1) checking if the last line has a single
    integer, (2) filtering out numbers that appear in the prompt (anchor
    echoes), (3) rejecting ambiguous cases.
    """
    if not raw_text:
        return None, False
    text = raw_text.strip()
    if re.fullmatch(r"-?\d+", text):
        v = int(text)
        return (v, True) if 0 <= v <= 100 else (None, False)

    all_ints = [int(m.group()) for m in _INT_PAT.finditer(text)]
    candidates = [c for c in all_ints if 0 <= c <= 100]
    if not candidates:
        return None, False

    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0], True

    # CoT-style: prefer the last number that looks like a conclusion ("is 79", "Answer: 79", "= 79")
    # so we don't take a divisor from "... / 4" when the real answer appeared earlier
    cot_matches = [int(m.group(1)) for m in _COT_NUM_PAT.finditer(text)]
    cot_valid = [v for v in cot_matches if 0 <= v <= 100]
    if cot_valid:
        return cot_valid[-1], True

    # Try last-line heuristic: many models put the final answer on the last line
    last_line = text.strip().split("\n")[-1].strip()
    last_line_ints = [int(m.group()) for m in _INT_PAT.finditer(last_line)]
    last_line_valid = [c for c in last_line_ints if 0 <= c <= 100]
    if len(set(last_line_valid)) == 1:
        return last_line_valid[0], True

    if prompt_text:
        prompt_nums = {
            int(m.group()) for m in _INT_PAT.finditer(prompt_text)
            if 0 <= int(m.group()) <= 100
        }
        filtered = [c for c in unique if c not in prompt_nums]
        if len(filtered) == 1:
            return filtered[0], True

    return None, False


def _parse_with_fallback(
    raw: str,
    prompt_text: str,
    is_cot: bool,
    fallback: LLMFallbackExtractor | None,
) -> tuple[int | None, bool, str]:
    """Parse answer with optional LLM fallback. Returns (answer, ok, strategy)."""
    if is_cot:
        answer, ok = parse_cot_answer(raw)
    else:
        answer, ok = parse_answer_int(raw, prompt_text)

    if ok:
        return answer, True, "regex"

    if fallback and raw and raw.strip():
        answer, ok = fallback.try_extract(raw)
        if ok:
            log.debug("LLM fallback extracted %d from: %.80s", answer, raw)
            return answer, True, "llm_fallback"

    return None, False, "failed"


def load_promptviews(path: Path) -> dict[str, dict[str, dict]]:
    """Load promptviews.jsonl -> {item_id: {condition: view_dict}}."""
    views: dict[str, dict[str, dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            pv = json.loads(line.strip())
            views.setdefault(pv["item_id"], {})[pv["condition"]] = pv
    return views


def load_itemspecs(path: Path) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line.strip())
            specs[s["item_id"]] = s
    return specs


def load_completed_items(output_path: Path) -> set[tuple]:
    """Return set of (item_id, condition, mitigation, sample_idx) already done."""
    done: set[tuple] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line.strip())
            done.add((r["item_id"], r["condition"], r["mitigation"], r.get("sample_idx", 0)))
    return done


class HFRunner:
    """Local HuggingFace model runner."""

    def __init__(self, model_id: str, device: str = "auto",
                 device_map: str | None = None, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        torch_dtype = getattr(torch, dtype, "auto")
        dm = device_map if device_map else device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype,
            device_map=dm, trust_remote_code=True,
        )
        self.model.eval()
        log.info("Loaded HF model %s on %s", model_id, self.model.device)

    def generate(self, prompt: str, max_tokens: int = 8,
                 temperature: float = 0.0) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 1.0

        with torch.no_grad():
            out = self.model.generate(
                **enc, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)

    def generate_chat(self, messages: list[dict], max_tokens: int = 8,
                      temperature: float = 0.0) -> str:
        """Generate one reply given a multi-turn messages list (user/assistant/user/...).

        messages: list of {"role": "user"|"assistant", "content": str}.
        Applies chat_template to the full conversation and generates the next token sequence.
        """
        import torch
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 1.0

        with torch.no_grad():
            out = self.model.generate(
                **enc, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)

    def generate_batch(self, prompts: list[str], max_tokens: int = 8,
                       temperature: float = 0.0, batch_size: int = 16) -> list[str]:
        import torch
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            texts = []
            for p in batch:
                messages = [{"role": "user", "content": p}]
                texts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))
            enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 truncation=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]

            gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                out = self.model.generate(
                    **enc, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id
                )
            decoded = self.tokenizer.batch_decode(out[:, prompt_len:],
                                                  skip_special_tokens=True)
            results.extend(decoded)
        return results

    def generate_batch_tool(
        self,
        messages_list: list[list[dict]],
        tools: list[dict] | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        batch_size: int = 16,
    ) -> list[str]:
        """Batched generation from structured tool-calling conversations.

        Each element of messages_list is a full conversation including
        system, user, assistant (with tool_calls), and tool response messages.
        The model's chat template renders them into the native format.
        """
        import torch

        texts = []
        for messages in messages_list:
            try:
                t = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                t = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            texts.append(t)

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, add_special_tokens=False,
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]

            gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                out = self.model.generate(
                    **enc, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id
                )
            decoded = self.tokenizer.batch_decode(
                out[:, prompt_len:], skip_special_tokens=True
            )
            results.extend(decoded)
        return results

    def generate_agentic_tool(
        self,
        user_message: str,
        tools: list[dict],
        tool_executors: dict,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_turns: int = 2,
    ) -> tuple[str, list[dict]]:
        """Single-item agentic tool-calling: generate -> parse tool_calls -> execute -> generate.

        Returns (final_answer_text, full_messages_list).
        """
        import torch

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        for turn in range(max_turns):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tools=tools,
                    tokenize=False, add_generation_prompt=True,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]

            gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 1.0

            with torch.no_grad():
                out = self.model.generate(
                    **enc, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id
                )
            raw = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)

            tool_calls = self._parse_tool_calls(raw)
            if not tool_calls:
                messages.append({"role": "assistant", "content": raw})
                return raw, messages

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}
                executor = tool_executors.get(fn_name)
                if executor:
                    result = executor(**args)
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tools=tools,
                tokenize=False, add_generation_prompt=True,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": False}
        with torch.no_grad():
            out = self.model.generate(
                **enc, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id
            )
        final = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": final})
        return final, messages

    @staticmethod
    def _parse_tool_calls(raw_text: str) -> list[dict]:
        """Best-effort parse of tool calls from model output."""
        import re as _re

        calls = []
        fn_pattern = _re.compile(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', _re.DOTALL
        )
        for m in fn_pattern.finditer(raw_text):
            try:
                obj = json.loads(m.group(1))
                name = obj.get("name", "")
                arguments = obj.get("arguments", {})
                calls.append({
                    "id": f"call_{len(calls)+1}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                    },
                })
            except json.JSONDecodeError:
                continue

        if not calls:
            json_pattern = _re.compile(
                r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
                _re.DOTALL,
            )
            for m in json_pattern.finditer(raw_text):
                calls.append({
                    "id": f"call_{len(calls)+1}",
                    "type": "function",
                    "function": {
                        "name": m.group(1),
                        "arguments": m.group(2),
                    },
                })

        return calls


def run_mitigation_hf(
    runner: HFRunner,
    views: dict[str, dict[str, dict]],
    specs: dict[str, dict],
    mitigation: str,
    output_path: Path,
    max_items: int | None = None,
    batch_size: int = 16,
    seed: int = 42,
    fallback_extractor: LLMFallbackExtractor | None = None,
) -> None:
    """Run a single mitigation on all items using HF backend."""
    done = load_completed_items(output_path)
    items = _prepare_items(views, specs, max_items, seed)
    conditions = ["control", "low_anchor", "high_anchor"]

    fh = open(output_path, "a", encoding="utf-8")
    total = len(items) * len(conditions)
    processed = 0
    fb = fallback_extractor

    for item in items:
        for cond in conditions:
            if (item["item_id"], cond, mitigation, 0) in done:
                processed += 1
                continue

            pv = item.get(cond)
            if pv is None:
                continue

            prompt = pv["prompt_text"]
            max_tok = get_max_tokens(mitigation)

            try:
                if mitigation in ("B0", "B1", "B2", "B3"):
                    modified = apply_existing_mitigation(prompt, mitigation)
                    raw = runner.generate(modified, max_tokens=max_tok)
                    answer, ok, strat = _parse_with_fallback(
                        raw, modified, mitigation == "B3", fb)
                    _write_result(fh, runner.model_id, item, cond, mitigation,
                                  answer, ok, raw, parse_strategy=strat)

                elif mitigation == "SELFHELP":
                    rewrite_prompt = selfhelp_rewrite_prompt(prompt)
                    rewritten = runner.generate(rewrite_prompt, max_tokens=512)
                    answer_prompt = selfhelp_answer_prompt(rewritten)
                    raw = runner.generate(answer_prompt, max_tokens=max_tok)
                    answer, ok, strat = _parse_with_fallback(
                        raw, answer_prompt, False, fb)
                    _write_result(fh, runner.model_id, item, cond, mitigation,
                                  answer, ok, raw,
                                  extra={"rewritten_prompt": rewritten},
                                  parse_strategy=strat)

                elif mitigation == "COUNTERFACTUAL_ENSEMBLE":
                    anchor_str = pv.get("anchor_string")
                    anchor_span = pv.get("anchor_span")
                    if anchor_str and anchor_span:
                        cf_prompts = counterfactual_prompts(
                            prompt, anchor_str, anchor_span
                        )
                        raw_outputs = runner.generate_batch(
                            cf_prompts, max_tokens=max_tok, batch_size=batch_size
                        )
                        answers = []
                        any_fb = False
                        for ro in raw_outputs:
                            a, ok_a, s = _parse_with_fallback(ro, "", False, fb)
                            if s == "llm_fallback":
                                any_fb = True
                            answers.append(a)
                        median, iqr = counterfactual_aggregate(answers)
                        strat = "llm_fallback" if any_fb else "regex"
                        _write_result(fh, runner.model_id, item, cond, mitigation,
                                      median, median is not None,
                                      json.dumps(raw_outputs),
                                      extra={"cf_answers": answers, "iqr": iqr},
                                      parse_strategy=strat)
                    else:
                        raw = runner.generate(prompt, max_tokens=max_tok)
                        answer, ok, strat = _parse_with_fallback(
                            raw, "", False, fb)
                        _write_result(fh, runner.model_id, item, cond, mitigation,
                                      answer, ok, raw, parse_strategy=strat)

                elif mitigation == "LATE_BINDING":
                    anchor_str = pv.get("anchor_string")
                    anchor_span = pv.get("anchor_span")
                    if anchor_str and anchor_span:
                        p1 = late_binding_pass1(prompt, anchor_str, anchor_span)
                        raw1 = runner.generate(p1, max_tokens=128)
                        a1, _, _ = _parse_with_fallback(raw1, "", False, fb)
                        prev = str(a1) if a1 is not None else "uncertain"
                        p2 = late_binding_pass2(prompt, prev, anchor_str)
                        raw2 = runner.generate(p2, max_tokens=max_tok)
                        answer, ok, strat = _parse_with_fallback(
                            raw2, p2, False, fb)
                        _write_result(fh, runner.model_id, item, cond, mitigation,
                                      answer, ok, raw2,
                                      extra={"pass1_answer": a1, "pass1_raw": raw1},
                                      parse_strategy=strat)
                    else:
                        raw = runner.generate(prompt, max_tokens=max_tok)
                        answer, ok, strat = _parse_with_fallback(
                            raw, "", False, fb)
                        _write_result(fh, runner.model_id, item, cond, mitigation,
                                      answer, ok, raw, parse_strategy=strat)

            except Exception as e:
                log.warning("Failed %s/%s/%s: %s", item["item_id"], cond, mitigation, e)
                _write_result(fh, runner.model_id, item, cond, mitigation,
                              None, False, f"ERROR: {e}",
                              parse_strategy="failed")

            processed += 1
            if processed % 50 == 0:
                log.info("[%s] %d/%d done", mitigation, processed, total)

    fh.close()
    log.info("[%s] Complete: %d items -> %s", mitigation, processed, output_path)


async def run_mitigation_api(
    model_id: str,
    views: dict[str, dict[str, dict]],
    specs: dict[str, dict],
    mitigation: str,
    output_path: Path,
    max_concurrent: int = 20,
    max_items: int | None = None,
    seed: int = 42,
    api_key: str | None = None,
    fallback_extractor: LLMFallbackExtractor | None = None,
) -> None:
    """Run a single mitigation using async OpenRouter API."""
    from .async_api import AsyncOpenRouterClient

    client = AsyncOpenRouterClient(api_key=api_key, max_concurrent=max_concurrent)
    done = load_completed_items(output_path)
    items = _prepare_items(views, specs, max_items, seed)
    conditions = ["control", "low_anchor", "high_anchor"]
    fb = fallback_extractor

    fh = open(output_path, "a", encoding="utf-8")
    tasks = []

    for item in items:
        for cond in conditions:
            if (item["item_id"], cond, mitigation, 0) in done:
                continue
            pv = item.get(cond)
            if pv is None:
                continue
            tasks.append((item, cond, pv))

    log.info("[%s] %d queries to run (async, max_concurrent=%d)",
             mitigation, len(tasks), max_concurrent)

    max_tok = get_max_tokens(mitigation, backend="openrouter")

    for batch_start in range(0, len(tasks), 100):
        batch = tasks[batch_start:batch_start + 100]
        prompts_to_send = []
        meta = []

        for item, cond, pv in batch:
            prompt = pv["prompt_text"]
            if mitigation in ("B0", "B1", "B2", "B3"):
                modified = apply_existing_mitigation(prompt, mitigation)
                prompts_to_send.append(modified)
                meta.append({"item": item, "cond": cond, "type": "simple",
                             "modified_prompt": modified})
            elif mitigation == "COUNTERFACTUAL_ENSEMBLE":
                anchor_str = pv.get("anchor_string")
                anchor_span = pv.get("anchor_span")
                if anchor_str and anchor_span:
                    cf_ps = counterfactual_prompts(prompt, anchor_str, anchor_span)
                    for cp in cf_ps:
                        prompts_to_send.append(cp)
                    meta.append({"item": item, "cond": cond, "type": "cf",
                                 "n_prompts": len(cf_ps)})
                else:
                    prompts_to_send.append(prompt)
                    meta.append({"item": item, "cond": cond, "type": "simple",
                                 "modified_prompt": prompt})
            else:
                prompts_to_send.append(prompt)
                meta.append({"item": item, "cond": cond, "type": "simple",
                             "modified_prompt": prompt})

        results = await client.query_batch(
            model_id, prompts_to_send, max_tokens=max_tok
        )

        ridx = 0
        for m in meta:
            item, cond = m["item"], m["cond"]
            if m["type"] == "cf":
                n = m["n_prompts"]
                answers = []
                raws = []
                any_fb = False
                for j in range(n):
                    raw = results[ridx + j]["raw_text"]
                    raws.append(raw)
                    a, _, s = _parse_with_fallback(raw, "", False, fb)
                    if s == "llm_fallback":
                        any_fb = True
                    answers.append(a)
                ridx += n
                median, iqr = counterfactual_aggregate(answers)
                strat = "llm_fallback" if any_fb else "regex"
                _write_result(fh, model_id, item, cond, mitigation,
                              median, median is not None,
                              json.dumps(raws),
                              extra={"cf_answers": answers, "iqr": iqr},
                              parse_strategy=strat)
            else:
                raw = results[ridx]["raw_text"]
                ridx += 1
                answer, ok, strat = _parse_with_fallback(
                    raw, m.get("modified_prompt", ""),
                    mitigation == "B3", fb)
                _write_result(fh, model_id, item, cond, mitigation,
                              answer, ok, raw, parse_strategy=strat)

        log.info("[%s] batch %d-%d done", mitigation, batch_start,
                 min(batch_start + 100, len(tasks)))

    await client.close()
    fh.close()
    log.info("[%s] API complete -> %s", mitigation, output_path)


def _prepare_items(
    views: dict[str, dict[str, dict]],
    specs: dict[str, dict],
    max_items: int | None,
    seed: int,
) -> list[dict]:
    """Build item list with prompt views attached."""
    items = []
    for item_id, cond_views in views.items():
        if "control" not in cond_views:
            continue
        spec = specs.get(item_id, {})
        item = {
            "item_id": item_id,
            "suite": cond_views["control"]["suite"],
            "domain": cond_views["control"]["domain"],
            "y_star": spec.get("y_star", spec.get("theta")),
            "anchors": spec.get("anchors", {}),
        }
        for c in ("control", "low_anchor", "high_anchor"):
            if c in cond_views:
                item[c] = cond_views[c]
        items.append(item)

    if max_items and max_items < len(items):
        rng = np.random.RandomState(seed)
        rng.shuffle(items)
        items = items[:max_items]

    return items


def _write_result(
    fh, model_id: str, item: dict, condition: str, mitigation: str,
    answer: int | None, parsed_ok: bool, raw_text: str,
    extra: dict | None = None,
    parse_strategy: str = "regex",
) -> None:
    record = {
        "model_id": model_id,
        "item_id": item["item_id"],
        "suite": item["suite"],
        "domain": item["domain"],
        "condition": condition,
        "mitigation": mitigation,
        "answer_int": answer,
        "parsed_ok": parsed_ok,
        "parse_strategy": parse_strategy,
        "raw_text": raw_text,
        "sample_idx": 0,
        "y_star": item.get("y_star"),
    }
    if extra:
        record.update(extra)
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()
