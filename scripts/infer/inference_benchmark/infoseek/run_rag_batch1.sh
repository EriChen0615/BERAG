#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"

# Submit one RAG request per vLLM.generate call while retaining the same
# deterministic 64-example, K=50 benchmark sample.
export BATCH_SIZE=1
export TAKE_N="${TAKE_N:-64}"
export SEED="${SEED:-42}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export RAG_MAX_MODEL_LEN="${RAG_MAX_MODEL_LEN:-262144}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/inference_benchmark/infoseek_batch1}"

echo "[InfoSeek batch=1] Running RAG baseline"
bash "${SCRIPT_DIR}/run_rag_k50.sh"

METRICS="${OUTPUT_ROOT}/qwen3-vl-8b/rag/K=50-TakeN=${TAKE_N}/run_metrics.json"
echo "[InfoSeek batch=1] RAG metrics: ${METRICS}"

if command -v jq >/dev/null 2>&1; then
    jq '{batch_size, p90_ttft_s, p90_tpot_s, vllm_run_wall_time_s, requests_per_second}' \
        "${METRICS}"
fi
