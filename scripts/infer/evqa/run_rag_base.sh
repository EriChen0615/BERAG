#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:-/workspace/projects/BERAG/outputs/jinghong_chen/PreFLMR-L_post_retrieval}"
IMG_BASEDIR="${IMG_BASEDIR:-/workspace/projects/BERAG/src/train/LlamaFactory-0.9.5-beft/data/EVQA}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/evqa}"
K_VALUES="${K_VALUES:-${RETRIEVAL_TOPK:-1,2,3,5,10,15,20,30,40,50}}"

K_VALUES="${K_VALUES//,/ }"
read -r -a K_ARRAY <<< "${K_VALUES}"

if [[ "${#K_ARRAY[@]}" -eq 0 ]]; then
    echo "No K values provided. Set K_VALUES, e.g. K_VALUES=1,3,5." >&2
    exit 1
fi

for RETRIEVAL_TOPK in "${K_ARRAY[@]}"; do
    EXP_NAME="K=${RETRIEVAL_TOPK}"
    if [[ -n "${TAKE_N:-}" ]]; then
        EXP_NAME="${EXP_NAME}-TakeN=${TAKE_N}"
    fi

    EXP_DIR="${OUTPUT_DIR}/rag-base/${EXP_NAME}"
    OUTPUT_PATH="${EXP_DIR}/predictions.jsonl"
    mkdir -p "${EXP_DIR}"

    echo "[EVQA RAG] Running K=${RETRIEVAL_TOPK}"
    echo "[EVQA RAG] Output: ${OUTPUT_PATH}"

    python "${REPO_ROOT}/src/infer/evqa_vllm_berag_inference.py"  \
        --mode rag   \
        --model "${MODEL}"   \
        --tokenizer-path "${TOKENIZER_PATH}"   \
        --retrieval-ds-path "${RETRIEVAL_DS_PATH}"   \
        --dataset-name "${DATASET_NAME:-EVQA}"   \
        --img-basedir "${IMG_BASEDIR}"   \
        --retrieval-field "${RETRIEVAL_FIELD:-retrieved_passage}"   \
        --retrieval-topk "${RETRIEVAL_TOPK}"   \
        --output-path "${OUTPUT_PATH}"   \
        --batch-size "${BATCH_SIZE:-64}"   \
        --max-model-len "${MAX_MODEL_LEN:-65536}"  \
        --max-tokens "${MAX_TOKENS:-32}"   \
        --dtype "${DTYPE:-bfloat16}"   \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"   \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"   \
        ${TAKE_N:+--take-n "${TAKE_N}"}   \
        ${ENSURE_GT:+--ensure-gt-passage-in-ensemble}  \
        ${DRY_RUN:+--dry-run --dry-run-no-images}
done
