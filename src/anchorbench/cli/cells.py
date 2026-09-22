"""Turn one (model, data, decoding) triple into the runner command for it.

``anchorbench eval`` and ``anchorbench experiment`` each used to carry a copy
of this logic, and the copies disagreed on the one decision that matters:
how a model is routed. The ``experiment`` copy was fixed to route by the
model's declared ``backend`` after Gemma was sent to OpenRouter (see
``tests/test_model_routing.py``); the ``eval`` copy kept prefix-matching the
``hf_id`` against ``google/`` and so still dispatched ``google/gemma-*`` to
the API runner. Both commands now call here, so there is one routing rule
and one place to change the runner argument set.

Everything in this module is pure: it reads dicts and returns a list of
strings. Directory creation and process launching stay in the callers.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from anchorbench.eval.constants import API_MODEL_IDS
from anchorbench.paths import ROOT

# Backends that mean "call a hosted endpoint" rather than "load weights here".
API_BACKENDS = frozenset({"openrouter", "api"})
API_RUNNER = "anchorbench.runners.api"


def is_api_model(model: Mapping[str, Any]) -> bool:
    """Route by the model's declared ``backend``, not by its ``hf_id`` prefix.

    Prefix matching swept up the Gemma models: ``google/gemma-3-4b-it`` is
    open-weight and declares ``backend: vllm``, but shares a namespace with
    ``google/gemini-2.5-flash``. Sent to OpenRouter, it produced 1,800 records
    with ``parse_rate`` 0.0 and every metric null, which looks like a finished
    run and contains no data. A model config without a ``backend`` key falls
    back to the fixed list of paper API models.
    """
    backend = (model.get("backend") or "").lower()
    if backend:
        return backend in API_BACKENDS
    return model.get("hf_id") in set(API_MODEL_IDS)


def runner_module(model: Mapping[str, Any], data: Mapping[str, Any]) -> str:
    """Name of the ``anchorbench.runners`` module that evaluates this cell.

    Local cells go to the runner named after ``data["suite"]``. A variant
    such as ``icl_dist`` keeps ``suite: icl`` in its data config and only
    changes the dataset, so it reaches the ``icl`` runner the same way.
    """
    if is_api_model(model):
        return API_RUNNER
    return f"anchorbench.runners.{data['suite']}"


def build_cell_cmd(
    model: Mapping[str, Any],
    data: Mapping[str, Any],
    decoding: Mapping[str, Any],
    out_dir: Path,
    *,
    batch_size: int | None = None,
    seed: int | None = None,
    baseline_condition: str | None = None,
    tool_plaintext: bool = False,
) -> list[str]:
    """Return the ``python -m anchorbench.runners.<x> ...`` argv for one cell.

    ``model``, ``data`` and ``decoding`` are plain dicts with the keys of the
    corresponding ``conf/`` YAML files. ``batch_size`` and ``seed`` are only
    passed through when given, so callers that rely on the runner defaults
    (``experiment``) and callers that set them explicitly (``eval``) produce
    the same command when the values coincide.
    """
    suite = data["suite"]
    module = runner_module(model, data)

    if is_api_model(model):
        # runners.api takes the suite *name* and resolves the dataset itself,
        # so a variant is passed as its own suite name (icl_dist), not as a
        # flag. ``baseline_condition`` is deliberately not forwarded: the
        # published API-tier History numbers were scored against the runner's
        # default two-stage control, and paper_main must keep reproducing
        # them (docs/RECONCILIATION.md D8).
        return [
            sys.executable, "-m", module,
            "--model_id", model["hf_id"],
            "--suites", data.get("variant") or suite,
            "--out_dir", str(out_dir),
            "--max_tokens", str(decoding["max_tokens"]),
        ]

    dataset_dir = ROOT / data["dataset_dir"]  # conf paths are repo-relative
    cmd = [
        sys.executable, "-m", module,
        "--model_id", model["hf_id"],
        "--promptviews", str(dataset_dir / data["promptviews_file"]),
        "--itemspecs", str(dataset_dir / data["itemspecs_file"]),
        "--out_dir", str(out_dir),
        "--backend", model.get("backend", "vllm"),
    ]
    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]
    cmd += ["--max_tokens", str(decoding["max_tokens"])]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    cmd += [
        "--gpu_memory_utilization", str(model.get("gpu_memory_utilization", 0.9)),
        "--max_model_len", str(model.get("max_model_len", 4096)),
        "--tensor_parallel_size", str(model.get("tensor_parallel_size", 1)),
        "--dtype", model.get("dtype", "bfloat16"),
    ]
    if suite == "history" and baseline_condition:
        cmd += ["--baseline_condition", baseline_condition]
    if suite == "tool" and tool_plaintext:
        # Forces the plaintext rendering for models that would otherwise get
        # native structured tool messages, which is what makes the
        # within-model comparison in tab:tool_plaintext possible.
        cmd += ["--tool_plaintext"]
    return cmd
