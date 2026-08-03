#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
source "${SCRIPT_DIR}/_common.sh"

export TAKE_N="${TAKE_N:-64}"
export SEED="${SEED:-42}"
export NUM_ACCUMULATOR_ROWS="${NUM_ACCUMULATOR_ROWS:-64}"
export RAG_MAX_MODEL_LEN="${RAG_MAX_MODEL_LEN:-65536}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/inference_benchmark/evqa_batch1}"

for k in 10 25 50; do
    run_evqa_batch1_benchmark rag "${k}"
    run_evqa_batch1_benchmark berag "${k}"
    run_evqa_batch1_benchmark berag_topp "${k}"
done

echo "[E-VQA batch=1] Completed K=10,25,50."
echo "[E-VQA batch=1] Results: ${OUTPUT_ROOT}/qwen3-vl-8b"
