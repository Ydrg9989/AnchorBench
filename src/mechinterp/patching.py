"""Activation patching via HF forward hooks for causal tracing."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .int_vocab import single_token_ids
from .metrics import js_divergence

log = logging.getLogger(__name__)


def _get_layers(model: Any) -> list[Any]:
    """Extract the transformer layer list from various model architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def _get_lm_head(model: Any):
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise ValueError(f"Cannot find lm_head in {type(model).__name__}")


def _get_final_norm(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


class ResidualPatcher:
    """Cache and patch residual-stream activations using forward hooks."""

    def __init__(self, model: Any, int_vocab: dict[int, list[int]]):
        self.model = model
        self.layers = _get_layers(model)
        self.n_layers = len(self.layers)
        self.int_vocab = int_vocab
        self.stid = single_token_ids(int_vocab)

    def _logits_to_probs(self, logits: Tensor) -> np.ndarray:
        logits = logits.float()
        token_ids = torch.tensor(
            [self.stid[y] for y in range(101)], device=logits.device
        )
        int_logits = logits[token_ids]
        return torch.softmax(int_logits, dim=0).detach().cpu().numpy()

    @torch.no_grad()
    def cache_activations(
        self, input_ids: Tensor
    ) -> dict[int, Tensor]:
        """Forward pass caching residual stream output at every layer."""
        cache: dict[int, Tensor] = {}
        handles = []

        for layer_idx, layer in enumerate(self.layers):
            def hook(module, inp, out, idx=layer_idx):
                if isinstance(out, tuple):
                    cache[idx] = out[0].detach().clone()
                else:
                    cache[idx] = out.detach().clone()
            handles.append(layer.register_forward_hook(hook))

        self.model(input_ids)

        for h in handles:
            h.remove()
        return cache

    @torch.no_grad()
    def patch_and_score(
        self,
        input_ids: Tensor,
        clean_cache: dict[int, Tensor],
        patch_layers: list[int],
        token_positions: list[int] | None = None,
    ) -> np.ndarray:
        """Forward with patching: replace residual at patch_layers with clean_cache.

        If token_positions is None, patches the last token position only.
        Returns P(y) shape (101,).
        """
        def make_hook(layer_idx: int):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    hidden = out[0]
                else:
                    hidden = out
                clean = clean_cache[layer_idx]
                positions = token_positions if token_positions is not None else [-1]
                for pos in positions:
                    if clean.shape[1] > abs(pos):
                        hidden[:, pos, :] = clean[:, pos, :]
                if isinstance(out, tuple):
                    return (hidden,) + out[1:]
                return hidden
            return hook

        handles = []
        for li in patch_layers:
            handles.append(self.layers[li].register_forward_hook(make_hook(li)))

        output = self.model(input_ids)
        for h in handles:
            h.remove()

        last_logits = output.logits[0, -1, :]
        return self._logits_to_probs(last_logits)

    def layer_sweep(
        self,
        ctrl_ids: Tensor,
        anch_ids: Tensor,
        p_ctrl: np.ndarray,
        p_anch: np.ndarray,
        token_positions: list[int] | None = None,
    ) -> list[dict]:
        """Patch one layer at a time, measure CAE."""
        js_base = js_divergence(p_anch, p_ctrl)
        if js_base < 1e-10:
            return [{"layer": l, "cae": 0.0, "js_patched": 0.0, "js_base": 0.0}
                    for l in range(self.n_layers)]

        clean_cache = self.cache_activations(ctrl_ids)
        results = []
        for layer_idx in range(self.n_layers):
            p_patched = self.patch_and_score(
                anch_ids, clean_cache, [layer_idx], token_positions
            )
            js_patched = js_divergence(p_patched, p_ctrl)
            cae = 1.0 - js_patched / js_base
            results.append({
                "layer": layer_idx,
                "cae": float(cae),
                "js_patched": float(js_patched),
                "js_base": float(js_base),
            })
        return results

    @torch.no_grad()
    def head_sweep(
        self,
        ctrl_ids: Tensor,
        anch_ids: Tensor,
        p_ctrl: np.ndarray,
        p_anch: np.ndarray,
        target_layers: list[int],
        token_positions: list[int] | None = None,
    ) -> list[dict]:
        """Patch individual attention heads at target layers."""
        js_base = js_divergence(p_anch, p_ctrl)
        if js_base < 1e-10:
            return []

        clean_cache = self.cache_activations(ctrl_ids)
        results = []

        for layer_idx in target_layers:
            layer = self.layers[layer_idx]
            attn = layer.self_attn if hasattr(layer, "self_attn") else None
            if attn is None:
                continue

            n_heads = getattr(attn, "num_heads", None)
            if n_heads is None:
                config = self.model.config
                n_heads = getattr(config, "num_attention_heads", 32)

            head_dim = self.model.config.hidden_size // n_heads

            for head_idx in range(n_heads):
                start = head_idx * head_dim
                end = start + head_dim

                def make_head_hook(li, s, e):
                    def hook(module, inp, out):
                        if isinstance(out, tuple):
                            hidden = out[0]
                        else:
                            hidden = out
                        clean = clean_cache[li]
                        positions = token_positions if token_positions is not None else [-1]
                        for pos in positions:
                            hidden[:, pos, s:e] = clean[:, pos, s:e]
                        if isinstance(out, tuple):
                            return (hidden,) + out[1:]
                        return hidden
                    return hook

                handle = self.layers[layer_idx].register_forward_hook(
                    make_head_hook(layer_idx, start, end)
                )
                output = self.model(anch_ids)
                handle.remove()

                last_logits = output.logits[0, -1, :]
                p_patched = self._logits_to_probs(last_logits)
                js_patched = js_divergence(p_patched, p_ctrl)
                effect = 1.0 - js_patched / js_base

                results.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "effect_size": float(effect),
                    "js_patched": float(js_patched),
                })

        results.sort(key=lambda x: x["effect_size"], reverse=True)
        return results
