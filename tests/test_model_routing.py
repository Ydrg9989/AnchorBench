"""Open-weight models must not be dispatched to the hosted-API runner.

_is_api used to prefix-match hf_ids against ("openai/", "anthropic/",
"google/", "x-ai/"). That swept up Gemma: google/gemma-3-1b-it and
google/gemma-3-4b-it are open-weight and declare backend: vllm, but share a
namespace with google/gemini-2.5-flash. They were sent to OpenRouter, which
answered 429 and produced 1800 records with parse_rate 0.0 and every metric
null -- a summary that looks complete and contains no data.
"""

import glob

import yaml

from anchorbench.cli.experiment import _is_api


def _models():
    for path in sorted(glob.glob("conf/model/*.yaml")):
        yield yaml.safe_load(open(path))


def test_routing_follows_declared_backend():
    wrong = [
        f"{m['short']} (backend={m.get('backend')}, hf_id={m['hf_id']}) -> "
        f"{'API' if _is_api(m) else 'local'}"
        for m in _models()
        if _is_api(m) != (m.get("backend") == "openrouter")
    ]
    assert not wrong, "models routed against their declared backend:\n  " + "\n  ".join(wrong)


def test_gemma_is_local_despite_the_google_namespace():
    """The specific collision that caused the failure, pinned by name."""
    for m in _models():
        if m["hf_id"].startswith("google/gemma"):
            assert not _is_api(m), f"{m['hf_id']} must run locally, not via the API"
