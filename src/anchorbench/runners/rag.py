"""Run inference on the RAG suite (5 conditions, batched) and evaluate.

Usage:
    python -m anchorbench.runners.rag \\
        --promptviews datasets/anchorbench_rag_core/promptviews_core.jsonl \\
        --itemspecs datasets/anchorbench_rag_core/itemspecs.jsonl \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --out_dir results/rag_core --backend vllm --batch_size 32
"""

from __future__ import annotations

from anchorbench.runners._single_stage import main as _main


def main() -> None:
    _main("RAG")


if __name__ == "__main__":
    main()
