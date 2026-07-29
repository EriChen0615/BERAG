#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
MODEL="${MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG5_LoRA-SFT}"

export RUN_MODE="sft"
export SFT_MODEL="${MODEL}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL}}"
export MODEL_SLUG="${MODEL_SLUG:-qwen2-vl-7b}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/infoseek_prefill_full}"

# The checkpoint was trained with five retrieved documents. Override K_VALUES
# for a sweep, e.g. K_VALUES=15,20,30,1,2,3,5,10,50.
export K_VALUES="${K_VALUES:-5}"
export CONTINUE_GENERATION_FROM_ANS="${CONTINUE_GENERATION_FROM_ANS:-1}"
export ADD_PSEUDO_DOCUMENT="${ADD_PSEUDO_DOCUMENT:-0}"

export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
export SFT_MAX_MODEL_LEN="${SFT_MAX_MODEL_LEN:-32768}"
export MAX_WORDS_PER_EVIDENCE="${MAX_WORDS_PER_EVIDENCE:-2048}"

exec bash "${REPO_ROOT}/scripts/infer/infoseek/run_sft_beft.sh"
