#!/usr/bin/env bash
# Final analyzer + aggregator pass for P1-P5.
# Run after all per-model chains have completed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /data/yiderigun/miniforge3/etc/profile.d/conda.sh
conda activate LLM_anchoring

echo "[$(date -Iseconds)] P1 cross-pathway intensity analyzer"
PYTHONPATH=src python -m anchorbench.analysis.intensity_pathway

echo "[$(date -Iseconds)] P2 RAG realism analyzer"
PYTHONPATH=src python -m anchorbench.analysis.rag_realism

echo "[$(date -Iseconds)] P3 Tool realism analyzer"
PYTHONPATH=src python -m anchorbench.analysis.tool_realism

echo "[$(date -Iseconds)] P4 task-spec analyzer"
PYTHONPATH=src python -m anchorbench.analysis.task_spec

echo "[$(date -Iseconds)] P5 uncertain-judgment analyzer"
PYTHONPATH=src python -m anchorbench.analysis.uncertain

echo "[$(date -Iseconds)] Refreshing rebuttal deliverables"
: # (rebuttal_deliverables removed: it only assembled the review-cycle bundle)

echo "[$(date -Iseconds)] WRAP_UP done."
