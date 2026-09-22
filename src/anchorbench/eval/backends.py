"""Backend abstractions for AnchorBench evaluation.

HFBackend   — local HuggingFace Transformers inference
VLLMBackend — vLLM inference (PagedAttention, high GPU utilization)

Hosted models are served by
:class:`anchorbench.inference.openrouter_backend.OpenRouterBackend`, which
implements the same :class:`Backend` protocol, so the evaluation loop in
:mod:`anchorbench.eval.evaluator` does not know which kind it is driving.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


@runtime_checkable
class Backend(Protocol):
    """What the evaluation loop needs from a model, local or hosted.

    :class:`HFBackend`, :class:`VLLMBackend` and
    :class:`anchorbench.inference.openrouter_backend.OpenRouterBackend` all
    satisfy it, as does the ``FakeBackend`` the tests use. ``batch_size`` is
    a hint for backends that batch locally; hosted backends may ignore it.
    """

    model_id: str

    @property
    def supports_structured(self) -> bool: ...

    @property
    def supports_tool_messages(self) -> bool:
        """Whether native tool-call messages can be sent (else use plaintext)."""
        ...

    def generate(
        self, prompt: str, *, max_tokens: int = 512,
        temperature: float = 0.0, structured: bool = False,
    ) -> str: ...

    def generate_chat(
        self, messages: list[dict], *, max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str: ...

    def generate_batch(
        self, prompts: list[str], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]: ...

    def generate_chat_batch(
        self, messages_list: list[list[dict]], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]: ...

    def generate_batch_tool(
        self, messages_list: list[list[dict]], tools: list[dict] | None = None,
        max_tokens: int = 64, temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]: ...

    def generate_for_extraction(
        self, raw_output: str, extraction_prompt: str,
    ) -> str: ...


class HFBackend:
    """Local HuggingFace model backend."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        device_map: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        torch_dtype = getattr(torch, dtype, "auto")
        dm = device_map if device_map else device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=dm,
            trust_remote_code=True,
        )
        self.model.eval()
        self._torch = torch
        log.info("Loaded HF model %s on %s", model_id, self.model.device)

        self._outlines_available = False
        try:
            import outlines  # noqa: F401
            self._outlines_available = True
        except ImportError:
            pass

    @property
    def supports_structured(self) -> bool:
        return self._outlines_available

    def _encode_and_generate(
        self, text: str, max_tokens: int, temperature: float,
    ) -> str:
        enc = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 1.0

        with self._torch.no_grad():
            out = self.model.generate(
                **enc, **gen_kwargs,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0, prompt_len:], skip_special_tokens=True,
        )

    def generate(
        self, prompt: str, *, max_tokens: int = 512,
        temperature: float = 0.0, structured: bool = False,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self._encode_and_generate(text, max_tokens, temperature)

    def generate_chat(
        self, messages: list[dict], *, max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self._encode_and_generate(text, max_tokens, temperature)

    @property
    def supports_tool_messages(self) -> bool:
        return True

    def _template(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        if tools is not None:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, add_generation_prompt=True,
                )
            except TypeError:
                pass  # template without tool support: fall through
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _generate_texts(
        self, texts: list[str], max_tokens: int, temperature: float, batch_size: int,
    ) -> list[str]:
        """Left-padded batched generation over already-templated texts."""
        results: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, add_special_tokens=False,
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with self._torch.no_grad():
                out = self.model.generate(
                    **enc, **gen_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            results.extend(self.tokenizer.batch_decode(
                out[:, prompt_len:], skip_special_tokens=True,
            ))
        return results

    def generate_batch(
        self, prompts: list[str], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        texts = [self._template([{"role": "user", "content": p}]) for p in prompts]
        return self._generate_texts(texts, max_tokens, temperature, batch_size)

    def generate_chat_batch(
        self, messages_list: list[list[dict]], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        texts = [self._template(m) for m in messages_list]
        return self._generate_texts(texts, max_tokens, temperature, batch_size)

    def generate_batch_tool(
        self,
        messages_list: list[list[dict]],
        tools: list[dict] | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        batch_size: int = 16,
    ) -> list[str]:
        texts = [self._template(m, tools=tools) for m in messages_list]
        return self._generate_texts(texts, max_tokens, temperature, batch_size)

    def generate_for_extraction(
        self, raw_output: str, extraction_prompt: str,
    ) -> str:
        return self.generate(extraction_prompt, max_tokens=16, temperature=0.0)


class VLLMBackend:
    """vLLM backend for high-throughput inference with full GPU utilization.

    Uses PagedAttention and continuous batching. Same chat/tool formatting as
    HFBackend (via Transformers tokenizer). Optional tensor parallelism for
    large models.
    """

    def __init__(
        self,
        model_id: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = 4096,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_id = model_id
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len or 4096
        self._dtype = dtype
        self._trust_remote_code = trust_remote_code

        # Tokenizer for chat/tool template (same as HF for consistency)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # vLLM engine
        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self._max_model_len,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )
        log.info(
            "Loaded vLLM model %s (tp=%s, gpu_util=%.2f)",
            model_id, tensor_parallel_size, gpu_memory_utilization,
        )

        self._sampling_params = SamplingParams(
            temperature=0.0, max_tokens=512,
        )

    @property
    def supports_structured(self) -> bool:
        return False

    @property
    def supports_tool_messages(self) -> bool:
        return True

    def _sampling(self, max_tokens: int, temperature: float = 0.0) -> Any:
        from vllm import SamplingParams
        return SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _prompt_for_user(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def generate(
        self, prompt: str, *, max_tokens: int = 512,
        temperature: float = 0.0, structured: bool = False,
    ) -> str:
        text = self._prompt_for_user(prompt)
        sampling = self._sampling(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate([text], sampling)
        return outputs[0].outputs[0].text

    def generate_chat(
        self, messages: list[dict], *, max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        sampling = self._sampling(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate([text], sampling)
        return outputs[0].outputs[0].text

    def generate_batch(
        self, prompts: list[str], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        prompt_strings = [self._prompt_for_user(p) for p in prompts]
        sampling = self._sampling(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate(prompt_strings, sampling)
        return [o.outputs[0].text for o in outputs]

    def generate_chat_batch(
        self, messages_list: list[list[dict]], *, max_tokens: int = 512,
        temperature: float = 0.0, batch_size: int = 16,
    ) -> list[str]:
        texts = [
            self._tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
            )
            for m in messages_list
        ]
        sampling = self._sampling(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate(texts, sampling)
        return [o.outputs[0].text for o in outputs]

    def generate_batch_tool(
        self,
        messages_list: list[list[dict]],
        tools: list[dict] | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        batch_size: int = 16,
    ) -> list[str]:
        texts = []
        for messages in messages_list:
            try:
                t = self._tokenizer.apply_chat_template(
                    messages, tools=tools,
                    tokenize=False, add_generation_prompt=True,
                )
            except TypeError:
                t = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            texts.append(t)
        sampling = self._sampling(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate(texts, sampling)
        return [o.outputs[0].text for o in outputs]

    def generate_for_extraction(
        self, raw_output: str, extraction_prompt: str,
    ) -> str:
        return self.generate(extraction_prompt, max_tokens=16, temperature=0.0)


