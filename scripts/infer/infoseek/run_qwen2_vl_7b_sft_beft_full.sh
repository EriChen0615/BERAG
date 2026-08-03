#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"

export SFT_MODEL="${SFT_MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG5_LoRA-SFT}"
export BEFT_MODEL="${BEFT_MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-BEFT[K=2]-data=64000-LoRA}"
export PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${BEFT_MODEL}/prior_head.pt}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${BEFT_MODEL}}"
export MODEL_SLUG="${MODEL_SLUG:-qwen2-vl-7b}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/infoseek_prefill_full}"

# Use the same naturally ordered K sweep for SFT and BEFT.
K_VALUES="${K_VALUES:-1,2,3,5,10,15,20,30,50}"
export CONTINUE_GENERATION_FROM_ANS=1
export ADD_PSEUDO_DOCUMENT="${ADD_PSEUDO_DOCUMENT:-0}"

export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
export BEFT_BATCH_SIZE="${BEFT_BATCH_SIZE:-16}"
export SFT_MAX_MODEL_LEN="${SFT_MAX_MODEL_LEN:-65536}"
export BEFT_MAX_MODEL_LEN="${BEFT_MAX_MODEL_LEN:-32768}"
export MAX_WORDS_PER_EVIDENCE="${MAX_WORDS_PER_EVIDENCE:-2048}"

# The Qwen2 BEFT checkpoint was trained with hidden_state_offset=4, which
# selects L - 4 - 1 = L - 5. Continuation mode resolves this base offset once.
export DEFAULT_PRIOR_TOKEN_OFFSET=-5
export PRIOR_MODELING="${PRIOR_MODELING:-mlp_head}"
export PRIOR_HEAD_NUM_LAYERS="${PRIOR_HEAD_NUM_LAYERS:-2}"
export PRIOR_HEAD_PROJ_DIM="${PRIOR_HEAD_PROJ_DIM:-1024}"
export NUM_ACCUMULATOR_ROWS="${NUM_ACCUMULATOR_ROWS:-512}"
export PRUNING_TOP_P=1.0

# This launcher always runs the complete test set without oracle injection.
unset TAKE_N ENSURE_GT

echo "[InfoSeek Qwen2 full] Running BEFT with K=${K_VALUES}, offset=-5"
# RUN_MODE=beft K_VALUES="${K_VALUES}" \
RUN_MODE=beft K_VALUES=50 \
    exec bash "${REPO_ROOT}/scripts/infer/infoseek/run_sft_beft.sh"

echo "[InfoSeek Qwen2 full] Running SFT with K=${K_VALUES}"
RUN_MODE=sft K_VALUES="${K_VALUES}" \
    bash "${REPO_ROOT}/scripts/infer/infoseek/run_sft_beft.sh"

