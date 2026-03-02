#!/bin/bash
# Launch all mitigation experiments in parallel:
#   - OpenRouter API mitigations (B1, B2, B3, EFR, CxDP) run sequentially on API
#   - HF mitigations run on GPUs 0-3 in parallel
set -euo pipefail
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true
export PYTHONUNBUFFERED=1

CFG="mitigation/config.yaml"
SUITES="external,rag,tool"
LOG_DIR="runner_outputs/logs"
mkdir -p "$LOG_DIR"

echo "====== Launching parallel mitigation experiments ======"
echo "Time: $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) detected"

# ─── OpenRouter API mitigations (sequential, runs alongside GPU jobs) ───
(
  echo "[API] Starting B1..."
  python mitigation/run_mitigation.py --config $CFG --mitigation B1 \
      --run_name mit_b1 --suites $SUITES 2>&1

  echo "[API] Starting B2..."
  python mitigation/run_mitigation.py --config $CFG --mitigation B2 \
      --run_name mit_b2 --suites $SUITES 2>&1

  echo "[API] Starting B3..."
  python mitigation/run_mitigation.py --config $CFG --mitigation B3 \
      --run_name mit_b3 --suites $SUITES 2>&1

  echo "[API] Starting EFR..."
  python mitigation/run_mitigation.py --config $CFG --mitigation EFR \
      --run_name mit_efr --suites $SUITES 2>&1

  echo "[API] Starting CxDP..."
  python mitigation/run_mitigation.py --config $CFG --mitigation CxDP \
      --run_name mit_cxdp --suites $SUITES 2>&1

  echo "[API] All API mitigations done at $(date)"
) > "$LOG_DIR/api_mitigations.log" 2>&1 &
API_PID=$!
echo "  API mitigations launched (PID $API_PID)"

# ─── HF GPU 0: Llama-3.2-1B — B1, B2, B3, EFR, B5(DoLa) ───
(
  export CUDA_VISIBLE_DEVICES=0
  for MIT in B1 B2 B3 EFR; do
    echo "[GPU0/1B] Running $MIT..."
    python mitigation/run_mitigation_hf.py --config $CFG --mitigation $MIT \
        --run_name hf_1b_${MIT,,} --suites $SUITES --model_idx 0 2>&1
  done
  echo "[GPU0/1B] Running B5 (DoLa)..."
  python mitigation/run_mitigation_hf.py --config $CFG --mitigation B5 \
      --run_name hf_1b_b5 --suites $SUITES --model_idx 0 \
      --dola_layers low 2>&1
  echo "[GPU0/1B] Done at $(date)"
) > "$LOG_DIR/gpu0_1b.log" 2>&1 &
GPU0_PID=$!
echo "  GPU0 (Llama-1B) launched (PID $GPU0_PID)"

# ─── HF GPU 1: Llama-3.2-3B — B1, B2, B3, EFR, B5(DoLa) ───
(
  export CUDA_VISIBLE_DEVICES=1
  for MIT in B1 B2 B3 EFR; do
    echo "[GPU1/3B] Running $MIT..."
    python mitigation/run_mitigation_hf.py --config $CFG --mitigation $MIT \
        --run_name hf_3b_${MIT,,} --suites $SUITES --model_idx 1 2>&1
  done
  echo "[GPU1/3B] Running B5 (DoLa)..."
  python mitigation/run_mitigation_hf.py --config $CFG --mitigation B5 \
      --run_name hf_3b_b5 --suites $SUITES --model_idx 1 \
      --dola_layers low 2>&1
  echo "[GPU1/3B] Done at $(date)"
) > "$LOG_DIR/gpu1_3b.log" 2>&1 &
GPU1_PID=$!
echo "  GPU1 (Llama-3B) launched (PID $GPU1_PID)"

# ─── HF GPU 2: Llama-3.1-8B — B1, B2, B3, EFR, B5(DoLa) ───
(
  export CUDA_VISIBLE_DEVICES=2
  for MIT in B1 B2 B3 EFR; do
    echo "[GPU2/8B] Running $MIT..."
    python mitigation/run_mitigation_hf.py --config $CFG --mitigation $MIT \
        --run_name hf_8b_${MIT,,} --suites $SUITES --model_idx 2 2>&1
  done
  echo "[GPU2/8B] Running B5 (DoLa)..."
  python mitigation/run_mitigation_hf.py --config $CFG --mitigation B5 \
      --run_name hf_8b_b5 --suites $SUITES --model_idx 2 \
      --dola_layers low 2>&1
  echo "[GPU2/8B] Done at $(date)"
) > "$LOG_DIR/gpu2_8b.log" 2>&1 &
GPU2_PID=$!
echo "  GPU2 (Llama-8B) launched (PID $GPU2_PID)"

# ─── HF GPU 3: stress-test items (all 3 models, B0 baseline) ───
(
  export CUDA_VISIBLE_DEVICES=3
  echo "[GPU3] Running stress-test B0 baselines..."
  # Use a temp config pointing to stress-test dataset
  cat > /tmp/stress_cfg.yaml <<EOCFG
dataset_path: poc_dataset/stress_test.jsonl
output_dir: runner_outputs
n_samples_per_item: 5
decoding:
  temperature: 0.7
  top_p: 1.0
  max_tokens: 8
system_prompt: "You are a helpful assistant. Follow the instructions exactly."
hf_models:
  - model_id: meta-llama/Llama-3.2-1B-Instruct
    dtype: bfloat16
    device_map: auto
  - model_id: meta-llama/Llama-3.2-3B-Instruct
    dtype: bfloat16
    device_map: auto
  - model_id: meta-llama/Llama-3.1-8B-Instruct
    dtype: bfloat16
    device_map: auto
EOCFG
  python mitigation/run_mitigation_hf.py --config /tmp/stress_cfg.yaml \
      --mitigation B0 --run_name hf_stress_b0 2>&1
  echo "[GPU3] Done at $(date)"
) > "$LOG_DIR/gpu3_stress.log" 2>&1 &
GPU3_PID=$!
echo "  GPU3 (stress tests) launched (PID $GPU3_PID)"

echo ""
echo "All jobs launched. PIDs:"
echo "  API: $API_PID"
echo "  GPU0: $GPU0_PID  GPU1: $GPU1_PID  GPU2: $GPU2_PID  GPU3: $GPU3_PID"
echo ""
echo "Monitor with: tail -f $LOG_DIR/*.log"
echo "Or check status: ps aux | grep run_mitigation"

# Wait for all background jobs
wait $API_PID $GPU0_PID $GPU1_PID $GPU2_PID $GPU3_PID
echo ""
echo "====== All experiments complete at $(date) ======"
