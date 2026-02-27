"""Prompt construction and model inference."""

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REGIME_A_SUFFIX = '\n\nRespond with ONLY "Option X" where X is a number from 1 to 11.'

REGIME_B_SYSTEM = (
    "You are a graduate admissions committee member reviewing applicants.\n"
    "For each student below, output a JSON list of objects with this exact format:\n"
    '[{"student_index": 0, "decision": "admit", "confidence": 75}, ...]\n'
    'Where "decision" is "admit" or "reject" and "confidence" is 0-100.\n'
    "Respond with ONLY the JSON array. No explanation."
)


def build_regime_a_messages(text):
    return [{"role": "user", "content": text + REGIME_A_SUFFIX}]


def build_regime_b_messages(students, perm_indices):
    lines = []
    for pos, student_idx in enumerate(perm_indices):
        lines.append(f"Student {pos}: {students[student_idx]}")
    return [
        {"role": "system", "content": REGIME_B_SYSTEM},
        {"role": "user", "content": "\n\n".join(lines)},
    ]


class ModelInference:
    def __init__(self, model_name, max_new_tokens_a=64, max_new_tokens_b=512):
        self.model_name = model_name
        self.max_new_tokens_a = max_new_tokens_a
        self.max_new_tokens_b = max_new_tokens_b
        self._is_qwen3 = "qwen3" in model_name.lower()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, messages, max_new_tokens):
        template_kwargs = {}
        if self._is_qwen3:
            template_kwargs["enable_thinking"] = False

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def query_regime_a(self, prompt_text):
        msgs = build_regime_a_messages(prompt_text)
        return self._generate(msgs, self.max_new_tokens_a)

    def query_regime_b(self, students, perm_indices):
        msgs = build_regime_b_messages(students, perm_indices)
        return self._generate(msgs, self.max_new_tokens_b)

    def cleanup(self):
        """Free GPU memory so the next model can be loaded."""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
