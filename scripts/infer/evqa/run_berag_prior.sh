#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL}"
RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:?Set RETRIEVAL_DS_PATH to the EVQA retrieval dataset path.}"
PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:?Set PRIOR_HEAD_PATH to prior_head.pt.}"
IMG_BASEDIR="${IMG_BASEDIR:-/root}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/infer/evqa/berag_prior_predictions.jsonl}"

python "${REPO_ROOT}/src/infer/evqa_vllm_berag_inference.py"   --mode berag   --prior-mode module   --prior-head-path "${PRIOR_HEAD_PATH}"   --prior-modeling "${PRIOR_MODELING:-mlp_head}"   --prior-head-num-layers "${PRIOR_HEAD_NUM_LAYERS:-2}"   --prior-head-proj-dim "${PRIOR_HEAD_PROJ_DIM:-1024}"   --default-prior-token-offset "${DEFAULT_PRIOR_TOKEN_OFFSET:--4}"   --model "${MODEL}"   --tokenizer-path "${TOKENIZER_PATH}"   --retrieval-ds-path "${RETRIEVAL_DS_PATH}"   --dataset-name "${DATASET_NAME:-EVQA}"   --img-basedir "${IMG_BASEDIR}"   --retrieval-field "${RETRIEVAL_FIELD:-reranked_passage}"   --retrieval-topk "${RETRIEVAL_TOPK:-5}"   --output-path "${OUTPUT_PATH}"   --batch-size "${BATCH_SIZE:-64}"   --max-model-len "${MAX_MODEL_LEN:-32768}"   --max-tokens "${MAX_TOKENS:-64}"   --dtype "${DTYPE:-bfloat16}"   --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"   --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"   --num-accumulator-rows "${NUM_ACCUMULATOR_ROWS:-512}"   --pruning-top-p "${PRUNING_TOP_P:-1.0}"   ${TAKE_N:+--take-n "${TAKE_N}"}   ${ENSURE_GT:+--ensure-gt-passage-in-ensemble}   ${DRY_RUN:+--dry-run --dry-run-no-images}
