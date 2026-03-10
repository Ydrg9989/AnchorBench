"""Logit lens: project residual stream at each layer through the unembedding."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .int_vocab import single_token_ids
from .patching import _get_layers, _get_lm_head, _get_final_norm

log = logging.getLogger(__name__)


@torch.no_grad()
def logit_lens_sweep(
    model: Any,
    input_ids: Tensor,
    int_vocab: dict[int, list[int]],
) -> list[np.ndarray]:
    """Apply final LN + LM head at each layer to get P_l(y).

    Returns list of length n_layers, each np.ndarray of shape (101,).
    """
    layers = _get_layers(model)
    lm_head = _get_lm_head(model)
    final_norm = _get_final_norm(model)
    stid = single_token_ids(int_vocab)
    token_id_list = torch.tensor([stid[y] for y in range(101)], device=model.device)

    residuals: dict[int, Tensor] = {}
    handles = []
    for li, layer in enumerate(layers):
        def hook(module, inp, out, idx=li):
            if isinstance(out, tuple):
                residuals[idx] = out[0].detach()
            else:
                residuals[idx] = out.detach()
        handles.append(layer.register_forward_hook(hook))

    model(input_ids)
    for h in handles:
        h.remove()

    distributions = []
    for li in range(len(layers)):
        h = residuals[li][:, -1, :]  # last token
        if final_norm is not None:
            h = final_norm(h)
        logits = lm_head(h)[0].float()  # (vocab_size,), cast to fp32
        int_logits = logits[token_id_list]
        probs = torch.softmax(int_logits, dim=0).cpu().numpy()
        distributions.append(probs)

    return distributions
