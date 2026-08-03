#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
MODEL="${MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1}"

[[ -f "${MODEL}/config.json" ]] || {
    echo "Missing merged DPO model at ${MODEL}" >&2
    exit 1
}
[[ -f "${MODEL}/model.safetensors.index.json" ]] || {
    echo "Missing sharded model index at ${MODEL}" >&2
    exit 1
}

export RUN_MODE="sft"
export SFT_MODEL="${MODEL}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL}}"
export MODEL_SLUG="${MODEL_SLUG:-qwen2-vl-7b}"
export RAG_RUN_SLUG="${RAG_RUN_SLUG:-rag-dpo}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/infoseek_full}"

# Natural full-set sweep used by the other InfoSeek launchers.
export K_VALUES="${K_VALUES:-1,2,3,5,10}"

# Standard page-level RAG: no assistant prefill, pseudo-document, oracle
# injection, or page-to-section explosion.
export CONTINUE_GENERATION_FROM_ANS=0
export ADD_PSEUDO_DOCUMENT=0
export EXPLODE_PAGE_SECTIONS=0
export SAVE_FULL_PROMPTS=0

export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
export SFT_MAX_MODEL_LEN="${SFT_MAX_MODEL_LEN:-32768}"
export MAX_WORDS_PER_EVIDENCE="${MAX_WORDS_PER_EVIDENCE:-2048}"
export MAX_TOKENS="${MAX_TOKENS:-64}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# The merged Qwen2-VL config declares 32K; the existing full RAG setup uses
# a 64K runtime window. Override SFT_MAX_MODEL_LEN if K=50 needs more context.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

# Always run the complete test set without oracle passage replacement.
unset TAKE_N ENSURE_GT

echo "[InfoSeek Qwen2 RAG-DPO full] K=${K_VALUES}; output=${OUTPUT_DIR}/${MODEL_SLUG}/${RAG_RUN_SLUG}"
exec bash "${REPO_ROOT}/scripts/infer/infoseek/run_sft_beft.sh"
