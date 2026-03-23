#!/usr/bin/env bash
# =============================================================================
# Cleanup datasets/ and results/ to keep only final benchmarking data.
#
# See docs/CLEANUP_PLAN.md for the full rationale.
#
# Usage:
#   bash scripts/cleanup_datasets_and_results.sh              # dry-run (default)
#   bash scripts/cleanup_datasets_and_results.sh --execute     # actually remove
#   bash scripts/cleanup_datasets_and_results.sh --execute --archive  # tar first, then remove
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

EXECUTE=false
ARCHIVE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute) EXECUTE=true; shift ;;
        --archive) ARCHIVE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# -----------------------------------------------------------------------------
# Datasets to REMOVE (everything not in KEEP)
# -----------------------------------------------------------------------------
DATASETS_TO_REMOVE=(
    datasets/anchorbench_external_pilot12
    datasets/anchorbench_history_pilot12
    datasets/anchorbench_icl_pilot12
    datasets/anchorbench_rag_pilot12
    datasets/anchorbench_tool_pilot12
    datasets/anchorbench_tool_read_pilot12
    datasets/anchorbench_v1
    datasets/anchorbench_v2_external_pilot
    datasets/anchorbench_v2_history_pilot
    datasets/anchorbench_v2_icl_pilot
    datasets/anchorbench_v2_rag_pilot
    datasets/anchorbench_v2_tool_pilot
)

# -----------------------------------------------------------------------------
# Results to REMOVE
# -----------------------------------------------------------------------------
RESULTS_TO_REMOVE=(
    results/external_v2_pilot
    results/external_v2_pilot_Llama31_8B_512
    results/external_v2_pilot_Llama32_3B
    results/external_v2_pilot_Llama32_3B_512
    results/external_v2_pilot_Llama32_3B_with_fallback
    results/external_v2_pilot_Llama32_3B_with_fallback_256
    results/external_v2_pilot_Qwen25_3B
    results/external_v2_pilot_Qwen25_3B_512
    results/external_v2_pilot_Qwen25_7B_512
    results/history_v2_pilot_Llama31_8B_512
    results/history_v2_pilot_Llama32_3B_512
    results/history_v2_pilot_Qwen25_3B_512
    results/history_v2_pilot_Qwen25_7B_512
    results/icl_v2_pilot
    results/rag_v2_pilot
    results/rag_v2_smoke_test
    results/tool_v2_pilot
    results/tool_v2_smoke
    results/tool_v2_smoke_chat
    results/smoke_test
    results/anchoring_eval_multi
    results/anchoring_eval_test
    results/sanity
    results/runs
    results/external_70b
    results/mitigation
    results/mitigation_api_retest
    results/mitigation_api_retest2
    results/mitigation_api_retest3
    results/mitigation_smoke_fallback
    results/mitigation_smoke_fallback2
    results/mechinterp
    results/rerror_report
    results/syn_anchors
    results/xml_tag_test
    results/EXTERNAL_V2_PILOT_SUMMARY.md
    results/parsing_comparison.json
    results/parsing_comparison_with_llm.json
    results/unified_all_suites.json
)

# Optional: remove figures/tables/summary if you don't need them (uncomment to include)
# RESULTS_TO_REMOVE+=( results/figure1_nai_vs_accuracy.png results/figure2_dose_response.png results/figure3_aalc_sweep.png )
# RESULTS_TO_REMOVE+=( results/table1_nai.csv results/table2_rr.csv results/table3_stress.csv results/summary.json results/dataset_and_experiment_summary.pdf )

total_size_k=0
list_paths() {
    local arr_name="$1"
    local arr
    eval "arr=(\"\${${arr_name}[@]}\")"
    for p in "${arr[@]}"; do
        if [[ -e "$p" ]]; then
            if [[ -d "$p" ]]; then
                local sz
                sz=$(du -sk "$p" 2>/dev/null | cut -f1)
                total_size_k=$((total_size_k + sz))
                echo "  $p ($(echo $sz | awk '{if($1>=1024) printf "%.1fM", $1/1024; else printf "%dK", $1}'))"
            else
                echo "  $p"
            fi
        fi
    done
}

echo "============================================"
echo "  Cleanup: datasets/ and results/"
echo "  Mode: $($EXECUTE && echo 'EXECUTE (will delete)' || echo 'DRY-RUN (no changes)')"
echo "============================================"

echo ""
echo "--- Datasets to REMOVE ---"
list_paths DATASETS_TO_REMOVE

echo ""
echo "--- Results to REMOVE ---"
list_paths RESULTS_TO_REMOVE

echo ""
echo "Total size of items to remove: $(echo $total_size_k | awk '{if($1>=1024) printf "%.1f GB", $1/1024/1024; else if($1>=0) printf "%.1f MB", $1/1024}')"
echo ""
echo "KEEPING:"
echo "  datasets: anchorbench_*_core, DATASET_CARD.md, upload_hf.py, hf_release/"
echo "  results:  full_benchmark/, smoke_experiment/"
echo ""

if [[ "$EXECUTE" != true ]]; then
    echo "To perform cleanup, run: bash scripts/cleanup_datasets_and_results.sh --execute"
    echo "To archive before cleanup: bash scripts/cleanup_datasets_and_results.sh --execute --archive"
    exit 0
fi

# -----------------------------------------------------------------------------
# Archive (optional)
# -----------------------------------------------------------------------------
if [[ "$ARCHIVE" == true ]]; then
    ARCHIVE_NAME="cleanup_archive_$(date +%Y%m%d_%H%M%S).tar.gz"
    echo "Creating archive $ARCHIVE_NAME ..."
    to_archive=()
    for p in "${DATASETS_TO_REMOVE[@]}" "${RESULTS_TO_REMOVE[@]}"; do
        [[ -e "$p" ]] && to_archive+=("$p")
    done
    if [[ ${#to_archive[@]} -gt 0 ]]; then
        tar czf "results/$ARCHIVE_NAME" "${to_archive[@]}"
        echo "  Archive saved to results/$ARCHIVE_NAME"
    fi
fi

# -----------------------------------------------------------------------------
# Remove
# -----------------------------------------------------------------------------
removed=0
for p in "${DATASETS_TO_REMOVE[@]}" "${RESULTS_TO_REMOVE[@]}"; do
    if [[ -e "$p" ]]; then
        echo "Removing: $p"
        rm -rf "$p"
        removed=$((removed + 1))
    fi
done

echo ""
echo "Done. Removed $removed items."
echo "Kept: datasets/anchorbench_*_core, results/full_benchmark/"
