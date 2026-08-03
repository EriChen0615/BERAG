#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
export REPO_ROOT
export MODEL="${MODEL:-google/gemma-4-E4B-it}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL}}"
export MODEL_SLUG="${MODEL_SLUG:-gemma4-e4b}"
export DISABLE_WEIGHTS_LOADING_TRACKING=1
export RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:-${REPO_ROOT}/outputs/jinghong_chen/EVQA-testfull-with-retrieval}"
export IMG_BASEDIR="${IMG_BASEDIR:-${REPO_ROOT}/src/train/LlamaFactory-0.9.5-beft/data/EVQA}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/evqa_full}"
export K_VALUES="${K_VALUES:-1,2,3,5,10,15,20,30,50}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MAX_TOKENS="${MAX_TOKENS:-64}"

unset TAKE_N ENSURE_GT DRY_RUN
exec bash "${REPO_ROOT}/scripts/infer/evqa/test_full/run_rag_base.sh"
