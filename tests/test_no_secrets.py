"""Guard against committing credentials.

The OpenRouter key used to live in `.env` / `.env.local` at the repo root.
Those files were never committed, but they sat inside the working tree, so any
folder upload, `huggingface-cli upload .`, or tarball of the repo would have
shipped them. The key now lives at ~/.config/anchorbench/env (see
scripts/run_with_env.sh); this test stops it from coming back.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Provider key prefixes. Each is a literal vendor prefix followed by the
# high-entropy body, so these cannot match ordinary prose or code.
SECRET_PATTERNS = {
    "OpenRouter": re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}"),
    "Anthropic": re.compile(r"sk-ant-[A-Za-z0-9\-_]{16,}"),
    "OpenAI": re.compile(r"sk-proj-[A-Za-z0-9\-_]{16,}"),
    "AWS": re.compile(r"AKIA[0-9A-Z]{16}"),
    "HuggingFace": re.compile(r"hf_[A-Za-z0-9]{34,}"),
}

SKIP_DIRS = {".git", ".venv", "build", "__pycache__", ".pytest_cache", ".ruff_cache"}

# Bulk model outputs: 578 MB of generations that no key is ever written into.
# Scanning them would make this test take minutes for no coverage gain.
SKIP_SUFFIXES = {".jsonl", ".pdf", ".png", ".ico", ".gz", ".synctex"}


def _candidate_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        yield path


def test_no_provider_keys_in_working_tree():
    findings = []
    for path in _candidate_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for provider, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{provider} key in {path.relative_to(ROOT)}")
    assert not findings, "Credentials found in the working tree:\n" + "\n".join(findings)


@pytest.mark.parametrize("name", [".env", ".env.local"])
def test_dotenv_files_absent_from_repo_root(name):
    assert not (ROOT / name).exists(), (
        f"{name} is back at the repo root. Keys belong in "
        "~/.config/anchorbench/env, which scripts/run_with_env.sh sources."
    )
