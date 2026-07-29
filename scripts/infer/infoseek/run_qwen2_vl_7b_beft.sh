#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
MODEL="${MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-BEFT[K=2]-data=64000-LoRA}"

export RUN_MODE="beft"
export BEFT_MODEL="${MODEL}"
export PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${MODEL}/prior_head.pt}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL}}"
export MODEL_SLUG="${MODEL_SLUG:-qwen2-vl-7b}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/infoseek_prefill}"

# The checkpoint was trained with two documents. Override K_VALUES for a
# sweep, e.g. K_VALUES=15,20,30,1,2,3,5,10,50.
export K_VALUES="${K_VALUES:-10,15,20,30}"
export CONTINUE_GENERATION_FROM_ANS="${CONTINUE_GENERATION_FROM_ANS:-1}"
export ADD_PSEUDO_DOCUMENT="${ADD_PSEUDO_DOCUMENT:-0}"

export BEFT_BATCH_SIZE="${BEFT_BATCH_SIZE:-16}"
export BEFT_MAX_MODEL_LEN="${BEFT_MAX_MODEL_LEN:-32768}"
export MAX_WORDS_PER_EVIDENCE="${MAX_WORDS_PER_EVIDENCE:-2048}"

# This transferred prior_head.pt is a two-layer MLP:
# 3584 -> 1024 -> 1, matching the original Qwen2-VL-7B BEFT training config.
export PRIOR_MODELING="${PRIOR_MODELING:-mlp_head}"
export PRIOR_HEAD_NUM_LAYERS="${PRIOR_HEAD_NUM_LAYERS:-2}"
export PRIOR_HEAD_PROJ_DIM="${PRIOR_HEAD_PROJ_DIM:-1024}"
export DEFAULT_PRIOR_TOKEN_OFFSET="${DEFAULT_PRIOR_TOKEN_OFFSET:--5}"
export NUM_ACCUMULATOR_ROWS="${NUM_ACCUMULATOR_ROWS:-512}"
export PRUNING_TOP_P="${PRUNING_TOP_P:-1.0}"
export TAKE_N="${TAKE_N:-64}"

exec bash "${REPO_ROOT}/scripts/infer/infoseek/run_sft_beft.sh"
