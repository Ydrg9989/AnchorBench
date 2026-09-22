"""Hydra-driven command-line interface for AnchorBench.

The main entry point is exposed via the ``anchorbench`` console script
(see ``pyproject.toml``). Subcommands:

    anchorbench eval data=external model=qwen_7b
    anchorbench eval data=icl_dist model=llama_8b
    anchorbench experiment +experiment=paper_main
    anchorbench tables --paper
    anchorbench generate data=external +size=smoke
    anchorbench verify --strict
    anchorbench add-model openai/gpt-5o

Any key of the composed config can be overridden with Hydra syntax
(``batch_size=128``, ``decoding.max_tokens=256``, ``+tool_plaintext=true``).
"""

from __future__ import annotations
