#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"

# Process one parent at a time so TTFT and TPOT are not affected by admission
# of other parents. K=50 still creates 50 document branches within that parent.
export BATCH_SIZE=1
export TAKE_N="${TAKE_N:-64}"
export SEED="${SEED:-42}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export NUM_ACCUMULATOR_ROWS="${NUM_ACCUMULATOR_ROWS:-64}"
export PRUNING_TOP_P="${PRUNING_TOP_P:-0.99}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/inference_benchmark/infoseek_batch1}"

echo "[InfoSeek batch=1] Running BERAG without pruning"
bash "${SCRIPT_DIR}/run_berag_k50.sh"

echo "[InfoSeek batch=1] Running BERAG with Top-P=${PRUNING_TOP_P}"
bash "${SCRIPT_DIR}/run_berag_topp_k50.sh"

BASE_DIR="${OUTPUT_ROOT}/qwen3-vl-8b"
printf -v TOP_P_DIR "berag-topp-p=%.3f" "${PRUNING_TOP_P}"
echo "[InfoSeek batch=1] Completed both runs"
echo "Top-P pruning threshold: ${PRUNING_TOP_P}"
echo "BERAG metrics:"
echo "  ${BASE_DIR}/berag/K=50-TakeN=${TAKE_N}/run_metrics.json"
echo "Top-P metrics:"
echo "  ${BASE_DIR}/${TOP_P_DIR}/K=50-TakeN=${TAKE_N}/run_metrics.json"

if command -v jq >/dev/null 2>&1; then
    jq '{batch_size, p90_ttft_s, p90_tpot_s, vllm_run_wall_time_s, requests_per_second}' \
        "${BASE_DIR}/berag/K=50-TakeN=${TAKE_N}/run_metrics.json" \
        "${BASE_DIR}/${TOP_P_DIR}/K=50-TakeN=${TAKE_N}/run_metrics.json"
fi
