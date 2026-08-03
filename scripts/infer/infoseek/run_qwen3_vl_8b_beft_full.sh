#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
MODEL="${MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-Infoseek-ckpt8792}"

export RUN_MODE="beft"
export BEFT_MODEL="${MODEL}"
export PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${MODEL}/prior_head.pt}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export MODEL_SLUG="${MODEL_SLUG:-qwen3-vl-8b}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/infoseek_prefill_full}"

# Run the larger K values first, followed by the small-K sweep and K=50.
export K_VALUES="${K_VALUES:-30,50}"
export CONTINUE_GENERATION_FROM_ANS="${CONTINUE_GENERATION_FROM_ANS:-1}"
export ADD_PSEUDO_DOCUMENT="${ADD_PSEUDO_DOCUMENT:-0}"

export BEFT_BATCH_SIZE="${BEFT_BATCH_SIZE:-1}"
export BEFT_MAX_MODEL_LEN="${BEFT_MAX_MODEL_LEN:-32768}"
export MAX_WORDS_PER_EVIDENCE="${MAX_WORDS_PER_EVIDENCE:-2048}"

export PRIOR_MODELING="${PRIOR_MODELING:-mlp_head}"
export PRIOR_HEAD_NUM_LAYERS="${PRIOR_HEAD_NUM_LAYERS:-2}"
export PRIOR_HEAD_PROJ_DIM="${PRIOR_HEAD_PROJ_DIM:-1024}"
# Training used beft_hidden_state_offset=4 and selected L - 4 - 1 = L - 5.
export DEFAULT_PRIOR_TOKEN_OFFSET=-5
export NUM_ACCUMULATOR_ROWS="${NUM_ACCUMULATOR_ROWS:-512}"
export PRUNING_TOP_P="${PRUNING_TOP_P:-1.0}"

# This launcher is specifically for the complete InfoSeek evaluation set.
unset TAKE_N

exec bash "${REPO_ROOT}/scripts/infer/infoseek/run_sft_beft.sh"
