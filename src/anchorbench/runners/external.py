"""Run inference on the External suite (5 conditions) and evaluate.

Usage:
    python -m anchorbench.runners.external \\
        --promptviews datasets/anchorbench_external_core/promptviews_core.jsonl \\
        --itemspecs datasets/anchorbench_external_core/itemspecs.jsonl \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --out_dir results/external_core --backend vllm
"""

from __future__ import annotations

from anchorbench.runners._single_stage import main as _main


def main() -> None:
    _main("External")


if __name__ == "__main__":
    main()
