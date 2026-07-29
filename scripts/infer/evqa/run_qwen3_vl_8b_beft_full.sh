#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
MODEL="${MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-EVQA-epoch1}"

export MODEL
export MODEL_SLUG="${MODEL_SLUG:-qwen3-vl-8b}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${MODEL}/prior_head.pt}"
export RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:-${REPO_ROOT}/outputs/jinghong_chen/EVQA-testfull-with-retrieval}"
export IMG_BASEDIR="${IMG_BASEDIR:-${REPO_ROOT}/src/train/LlamaFactory-0.9.5-beft/data/EVQA}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/evqa_full}"

# Run the large-K experiments first, then complete the standard E-VQA sweep.
export K_VALUES="${K_VALUES:-1,2,3,5,10,15,20,30,50}"
export DEFAULT_PRIOR_TOKEN_OFFSET=-5
export PRUNING_TOP_P=1.0

export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_TOKENS="${MAX_TOKENS:-64}"
export NUM_ACCUMULATOR_ROWS="${NUM_ACCUMULATOR_ROWS:-512}"

# This is the canonical unpruned full-set run; ignore stale shell flags.
unset TAKE_N DO_TOP_P_PRUNE SET_TOP_P_WITH_K ENSURE_GT DRY_RUN

exec bash "${REPO_ROOT}/scripts/infer/evqa/test_full/run_berag_prior.sh"
