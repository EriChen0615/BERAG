#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
# MODEL="${MODEL:-/workspace/projects/BERAG/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-EVQA64000}"
# MODEL="${MODEL:-/workspace/projects/BERAG/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-EVQA-ckpt6000}"
MODEL="${MODEL:-/workspace/projects/BERAG/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-EVQA-epoch1}"
MODEL_SLUG="${MODEL_SLUG:-qwen3-vl-8b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:-/workspace/projects/BERAG/outputs/jinghong_chen/PreFLMR-L_post_retrieval}"
# PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-/workspace/projects/BERAG/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-EVQA64000/prior_head.pt}"
# PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-/workspace/projects/BERAG/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-EVQA-ckpt6000/prior_head.pt}"
PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-/workspace/projects/BERAG/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-EVQA-epoch1/prior_head.pt}"
IMG_BASEDIR="${IMG_BASEDIR:-/workspace/projects/BERAG/src/train/LlamaFactory-0.9.5-beft/data/EVQA}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/evqa}"
K_VALUES="${K_VALUES:-${RETRIEVAL_TOPK:-1,2,3,5,10,15,20,30,40,50}}"
# K_VALUES="${K_VALUES:-${RETRIEVAL_TOPK:-10}}"
# K_VALUES="${K_VALUES:-${RETRIEVAL_TOPK:-4}}"
# K_VALUES="${K_VALUES:-${RETRIEVAL_TOPK:-30,50}}"
PRUNING_TOP_P="${PRUNING_TOP_P:-1.0}"

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
    if [[ "${PRUNING_TOP_P}" != "1.0" ]]; then
        EXP_NAME="${EXP_NAME}-TopP=${PRUNING_TOP_P}"
    fi

    EXP_DIR="${OUTPUT_DIR}/${MODEL_SLUG}/berag_prior/${EXP_NAME}"
    OUTPUT_PATH="${EXP_DIR}/predictions.jsonl"
    mkdir -p "${EXP_DIR}"

    echo "[EVQA BERAG prior] Running K=${RETRIEVAL_TOPK}"
    echo "[EVQA BERAG prior] Output: ${OUTPUT_PATH}"

    python "${REPO_ROOT}/src/infer/evqa_vllm_berag_inference.py"   \
        --mode berag   \
        --prior-mode module   \
        --prior-head-path "${PRIOR_HEAD_PATH}"   \
        --prior-modeling "${PRIOR_MODELING:-mlp_head}"   \
        --prior-head-num-layers "${PRIOR_HEAD_NUM_LAYERS:-2}"   \
        --prior-head-proj-dim "${PRIOR_HEAD_PROJ_DIM:-1024}"   \
        --default-prior-token-offset "${DEFAULT_PRIOR_TOKEN_OFFSET:--4}"   \
        --model "${MODEL}"   \
        --tokenizer-path "${TOKENIZER_PATH}"   \
        --retrieval-ds-path "${RETRIEVAL_DS_PATH}"   \
        --dataset-name "${DATASET_NAME:-EVQA}"   \
        --img-basedir "${IMG_BASEDIR}"   \
        --retrieval-field "${RETRIEVAL_FIELD:-retrieved_passage}"   \
        --retrieval-topk "${RETRIEVAL_TOPK}"   \
        --output-path "${OUTPUT_PATH}"   \
        --batch-size "${BATCH_SIZE:-64}"   \
        --max-model-len "${MAX_MODEL_LEN:-32768}"   \
        --max-tokens "${MAX_TOKENS:-64}"   \
        --dtype "${DTYPE:-bfloat16}"   \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"  \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"   \
        --num-accumulator-rows "${NUM_ACCUMULATOR_ROWS:-512}"   \
        --pruning-top-p "${PRUNING_TOP_P}"   \
        ${TAKE_N:+--take-n "${TAKE_N}"}   \
        ${ENSURE_GT:+--ensure-gt-passage-in-ensemble}   \
        ${DRY_RUN:+--dry-run --dry-run-no-images}
done
