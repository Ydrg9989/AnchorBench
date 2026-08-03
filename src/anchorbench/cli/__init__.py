"""Hydra-driven command-line interface for AnchorBench.

The main entry point is exposed via the ``anchorbench`` console script
(see ``pyproject.toml``). Subcommands:

    anchorbench eval data=external model=qwen_7b
    anchorbench eval data=icl data.variant=icl_dist model=llama_8b
    anchorbench experiment=paper_main
    anchorbench tables --paper
    anchorbench generate data=external
    anchorbench add-model openai/gpt-5o

All Hydra overrides are accepted (``model.batch_size=128``,
``+decoding.n_samples=5``).
"""

from __future__ import annotations
