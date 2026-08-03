#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
MODEL="${MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-SFT-Infoseek64000}"

export RUN_MODE="sft"
export SFT_MODEL="${MODEL}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export MODEL_SLUG="${MODEL_SLUG:-qwen3-vl-8b}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/infoseek_prefill_full}"

export K_VALUES="${K_VALUES:-30,50}"
export CONTINUE_GENERATION_FROM_ANS="${CONTINUE_GENERATION_FROM_ANS:-1}"
export ADD_PSEUDO_DOCUMENT="${ADD_PSEUDO_DOCUMENT:-0}"

# The exported Qwen3-VL model declares a 262,144-token context window. Long
# contexts consume substantial KV-cache memory; override this with 131072 on
# lower-memory GPUs, or use tensor parallelism across multiple GPUs.
export SFT_MAX_MODEL_LEN="${SFT_MAX_MODEL_LEN:-262144}"
export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-16}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

export MAX_WORDS_PER_EVIDENCE="${MAX_WORDS_PER_EVIDENCE:-2048}"
export MAX_TOKENS="${MAX_TOKENS:-64}"

# This launcher always targets the complete InfoSeek evaluation set.
unset TAKE_N ENSURE_GT

exec bash "${REPO_ROOT}/scripts/infer/infoseek/run_sft_beft.sh"
