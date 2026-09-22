"""Run inference on the ICL suites (5 core conditions, batched) and evaluate.

Serves both the metadata-header ICL suite (``icl``) and ``icl_dist``
(demo-answer bands x framing); they ship the same five condition names.

Usage:
    python -m anchorbench.runners.icl \\
        --promptviews datasets/anchorbench_icl_core/promptviews_core.jsonl \\
        --itemspecs datasets/anchorbench_icl_core/itemspecs.jsonl \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --out_dir results/icl_core --backend vllm --batch_size 32

    python -m anchorbench.runners.icl \\
        --promptviews datasets/anchorbench_icl_dist_core/promptviews_core.jsonl \\
        --itemspecs datasets/anchorbench_icl_dist_core/itemspecs.jsonl \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --out_dir results/icl_dist_core/icl --backend vllm
"""

from __future__ import annotations

from anchorbench.runners._single_stage import main as _main


def main() -> None:
    _main("ICL")


if __name__ == "__main__":
    main()
