"""The tiers of the answer parser.

The cascade that orders them is ``evaluator.parse_response``:

    structured (JSON handed in as structured_raw; no runner produces it)
    -> xml_tag -> final_answer
    -> [a declared but out-of-range answer, or a truncated response, stops here]
    -> regex (parse_answer_int) -> last_number -> llm_fallback

Every record stores the tier that succeeded as ``parse_strategy``:
"structured" | "xml_tag" | "final_answer" | "regex" | "last_number"
| "llm_fallback" | "failed". A failed parse is counted, never defaulted.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)


def _range_check(v: int, lo: int = 0, hi: int = 100) -> tuple[int | None, bool]:
    """Return (value, True) if v is in [lo, hi], else (None, False)."""
    if lo <= v <= hi:
        return v, True
    return None, False


_INT_PAT = re.compile(r"(?<![\d.])-?\d+(?!\d)(?!\.\d)")
_NUM_PAT = re.compile(r"-?\d+(?:\.\d+)?")

_COT_NUM_PAT = re.compile(
    r"(?:is|=|:\s*)\s*(\d{1,3})\s*(?:[.\s\n]|$)",
    re.IGNORECASE,
)

_XML_ANSWER_PAT = re.compile(
    r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>",
    re.IGNORECASE,
)

# High-precision: safe on full document (avoid evidence lines like
# "Focus group rating: 59" or "The maximum index is 27"). "is" may be followed
# by a colon ("the index is: 2188"): a declared answer must be recognised even
# when it is out of range, or the cascade falls through to the regex tier and
# returns an evidence value as the answer.
_FINAL_ANSWER_STRICT = [
    re.compile(r"(?:final\s+answer|my\s+(?:final\s+)?(?:answer|estimate))\s*(?:is|:)\s*(\d{1,3})", re.IGNORECASE),
    re.compile(r"(?:the\s+answer\s+is|answer\s*:)\s*(\d{1,3})", re.IGNORECASE),
    re.compile(r"Answer:\s*\[?(\d{1,3}(?:\.\d+)?)\]?", re.IGNORECASE),
    re.compile(r"(?:overall|estimated|composite)\s+\S+\s+(?:index|score|rating)\s+(?:is\s*:?|=|:)\s*(?:approximately\s+|about\s+|around\s+)?(\d{1,3})", re.IGNORECASE),
    re.compile(
        r"(?:overall\s+)?(?:willingness-to-pay|WTP)\s+index\s+(?:is\s*:?|=|:)\s*(?:approximately\s+|about\s+|around\s+)?(\d{1,3})",
        re.IGNORECASE,
    ),
    re.compile(r"\*\*(\d{1,3})\*\*\s*$", re.MULTILINE),
]

# Only applied to the last *tail_chars* of the response so we never
# latch onto mid-document evidence ("maximum index is 27").
_FINAL_ANSWER_TAIL_PATS = [
    re.compile(
        r"(?<!maximum\s)(?<!minimum\s)(?<!average\s)"
        r"\bindex\s+is\s*:?\s*(?:approximately\s+|about\s+|around\s+)?(\d{1,3})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:therefore|thus|hence|so|,)\s+(?:the\s+)?(?:overall\s+)?(?:willingness-to-pay\s+)?(?:index|estimate|rating|score)\s+(?:is\s*:?|=|:)\s*(\d{1,3})\b",
        re.IGNORECASE,
    ),
]

_FINAL_ANSWER_TAIL_CHARS = 520

# Union for has_explicit_final_answer (out-of-range guard)
# Detect model outputs that are tool calls (JSON with "name" field) rather than answers
_TOOL_CALL_PAT = re.compile(r'^\s*\{["\s]*name["\s]*:', re.IGNORECASE)

_EXTRACT_PROMPT = (
    "Extract the single final numeric answer (integer 0-100) from the text below.\n"
    "Return ONLY the integer on its own line. If no clear answer, return NONE.\n\n"
    "Text: {completion}\n\nAnswer:"
)

def parse_structured(
    raw_text: str,
) -> tuple[int | None, bool]:
    """Extract answer from a structured JSON response like {"answer": 72}.

    Returns (value, True) on success, (None, False) on any failure.
    """
    if not raw_text:
        return None, False
    try:
        obj = json.loads(raw_text.strip())
    except (json.JSONDecodeError, ValueError):
        return None, False
    ans = obj.get("answer")
    if isinstance(ans, int):
        return _range_check(ans)
    if isinstance(ans, float) and ans == int(ans):
        return _range_check(int(ans))
    return None, False


def parse_answer_int(
    raw_text: str, prompt_text: str = "",
) -> tuple[int | None, bool]:
    """Extract a single integer in [0, 100] from model output.

    Rejects tool-call JSON outputs to avoid picking up evidence/anchor
    values from structured data.  Uses echo-filtering against prompt_text
    to avoid counting anchor values.
    """
    if not raw_text:
        return None, False
    text = raw_text.strip()

    if is_tool_call_output(text):
        return None, False

    if re.fullmatch(r"-?\d+", text):
        v = int(text)
        return _range_check(v)

    all_ints = [int(m.group()) for m in _INT_PAT.finditer(text)]
    candidates = [c for c in all_ints if 0 <= c <= 100]

    if not candidates:
        return None, False

    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0], True

    last_line = text.strip().split("\n")[-1].strip()
    last_line_ints = [int(m.group()) for m in _INT_PAT.finditer(last_line)]
    last_line_valid = [c for c in last_line_ints if 0 <= c <= 100]
    if len(set(last_line_valid)) == 1:
        return last_line_valid[0], True

    cot_matches = [int(m.group(1)) for m in _COT_NUM_PAT.finditer(text)]
    cot_valid = [v for v in cot_matches if 0 <= v <= 100]
    if cot_valid:
        return cot_valid[-1], True

    if prompt_text:
        prompt_nums = {
            int(m.group()) for m in _INT_PAT.finditer(prompt_text)
            if 0 <= int(m.group()) <= 100
        }
        filtered = [c for c in unique if c not in prompt_nums]
        if len(filtered) == 1:
            return filtered[0], True

    return None, False


def parse_xml_answer(
    raw_text: str,
) -> tuple[int | None, bool]:
    """Extract answer from <answer>N</answer> tags."""
    if not raw_text:
        return None, False
    matches = _XML_ANSWER_PAT.findall(raw_text)
    if not matches:
        return None, False
    val_str = matches[-1]
    try:
        val = float(val_str)
        rounded = round(val)
        return _range_check(rounded)
    except ValueError:
        pass
    return None, False


def is_tool_call_output(raw_text: str) -> bool:
    """Detect when the model outputs a tool call JSON instead of an answer."""
    if not raw_text:
        return False
    return bool(_TOOL_CALL_PAT.match(raw_text.strip()))


def parse_last_number(
    raw_text: str,
) -> tuple[int | None, bool]:
    """Last-number heuristic: prefer last line, then last number overall.

    Rejects text that looks like a tool-call JSON (numbers there are
    evidence values or anchor values, not the model's answer).
    Falls back to rounding if the last numeric value is a float.
    """
    if not raw_text:
        return None, False
    text = raw_text.strip()

    if is_tool_call_output(text):
        return None, False

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        last_line = lines[-1]
        last_line_nums = list(_NUM_PAT.finditer(last_line))
        for m in reversed(last_line_nums):
            try:
                val = float(m.group())
                rounded = round(val)
                result, ok = _range_check(rounded)
                if ok:
                    return result, True
            except ValueError:
                continue

    all_nums = list(_NUM_PAT.finditer(text))
    if not all_nums:
        return None, False

    for m in reversed(all_nums):
        try:
            val = float(m.group())
            rounded = round(val)
            return _range_check(rounded)
        except ValueError:
            continue
    return None, False


def parse_final_answer(
    raw_text: str,
) -> tuple[int | None, bool]:
    """Extract answer via explicit final-answer patterns.

    Phase 1 (strict): phrases like "final answer is 42", "estimated WTP
    index is 73" — applied to the **full** text but avoids generic
    ``rating/index is N`` that matches evidence bullets.

    Phase 2 (tail only): ``index is N`` and ``therefore the index is N``
    on the **last ~520 characters** so mid-document lines like
    "The maximum index is 27" are never used as the parsed answer.
    """
    if not raw_text:
        return None, False
    text = raw_text.strip()

    for pat in _FINAL_ANSWER_STRICT:
        matches = list(pat.finditer(text))
        if matches:
            try:
                v = round(float(matches[-1].group(1)))
            except ValueError:
                continue
            result, ok = _range_check(v)
            if ok:
                return result, True

    tail = text[-_FINAL_ANSWER_TAIL_CHARS:] if len(text) > _FINAL_ANSWER_TAIL_CHARS else text
    for pat in _FINAL_ANSWER_TAIL_PATS:
        matches = list(pat.finditer(tail))
        if matches:
            try:
                v = round(float(matches[-1].group(1)))
            except ValueError:
                continue
            result, ok = _range_check(v)
            if ok:
                return result, True

    return None, False


def has_explicit_final_answer(raw_text: str) -> bool:
    """Check if text contains an explicit final-answer phrase,
    regardless of whether the value is in range.

    Mirrors parse_final_answer's strict + tail scope.
    """
    if not raw_text:
        return False
    text = raw_text.strip()
    for pat in _FINAL_ANSWER_STRICT:
        if pat.search(text):
            return True
    tail = text[-_FINAL_ANSWER_TAIL_CHARS:] if len(text) > _FINAL_ANSWER_TAIL_CHARS else text
    for pat in _FINAL_ANSWER_TAIL_PATS:
        if pat.search(tail):
            return True
    return False


_INCOMPLETE_TAIL_WORDS = frozenset({
    "of", "the", "a", "an", "to", "and", "or", "with", "that", "which",
    "when", "if", "as", "by", "for", "in", "on", "at", "from", "into",
    "would", "could", "should", "will", "can", "may", "might", "be",
    "is", "are", "was", "were", "have", "has", "had", "do", "does", "did",
    "following", "using", "considering", "assuming", "average", "sum",
    "total", "where", "while", "however", "although", "because",
})


def looks_incomplete_response(raw_text: str) -> bool:
    """True if generation likely stopped mid-sentence (hit max_tokens).

    Prevents ``last_number`` from returning a random earlier value (e.g.
    100 from a discarded calculation) when the model never finished.
    """
    if not raw_text:
        return False
    t = raw_text.rstrip()
    if len(t) < 100:
        return False
    if t.endswith((".", "!", "?", ")", "]", "</answer>", '"')):
        return False
    if re.search(r"\b\d{1,3}\s*$", t):
        return False
    last_line = t.split("\n")[-1].strip()
    if not last_line:
        return False
    if re.search(r"\d", last_line):
        return False
    last_w = last_line.split()[-1].lower().rstrip(",;:'\"")
    if last_w in _INCOMPLETE_TAIL_WORDS:
        return True
    return False


class LLMFallbackExtractor:
    """Uses an HF model to extract the final integer answer from raw text."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
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
        """Ask the LLM to extract the integer answer from raw_text."""
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
        response = self.tokenizer.decode(
            out[0, prompt_len:], skip_special_tokens=True
        ).strip()

        if response.upper() == "NONE":
            return None, False
        m = re.search(r"\b(\d+)\b", response)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return v, True
        return None, False


XML_TAG_INSTRUCTION = (
    "\n\nAfter your reasoning, wrap your final integer estimate (0-100) in "
    "XML tags exactly like this: <answer>42</answer>"
)

