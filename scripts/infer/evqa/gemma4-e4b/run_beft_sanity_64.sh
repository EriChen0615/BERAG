#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Gemma-4-E4B-it-BEFT-EVQA}"
EVAL_VENV="${EVAL_VENV:-${REPO_ROOT}/src/eval/.venv}"
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval/evqa/evaluate_experiments.sh"

if [[ ! -f "${EVAL_VENV}/bin/activate" ]]; then
    echo "E-VQA eval environment not found: ${EVAL_VENV}" >&2
    exit 1
fi
if [[ ! -x "${EVAL_SCRIPT}" ]]; then
    echo "E-VQA eval script not found or not executable: ${EVAL_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${MODEL}/config.json" || ! -f "${MODEL}/prior_head.pt" ]]; then
    if [[ "${AUTO_EXPORT:-1}" == "1" ]]; then
        bash "${SCRIPT_DIR}/export_models.sh" beft
    else
        echo "Merged BEFT model or prior head not found: ${MODEL}" >&2
        echo "Run: bash ${SCRIPT_DIR}/export_models.sh beft" >&2
        exit 1
    fi
fi

export REPO_ROOT MODEL
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL}}"
export PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${MODEL}/prior_head.pt}"
export MODEL_SLUG="${MODEL_SLUG:-gemma4-e4b}"
export DISABLE_WEIGHTS_LOADING_TRACKING=1
export RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:-${REPO_ROOT}/outputs/jinghong_chen/EVQA-testfull-with-retrieval}"
export IMG_BASEDIR="${IMG_BASEDIR:-${REPO_ROOT}/src/train/LlamaFactory-0.9.5-beft/data/EVQA}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/evqa_sanity}"
export K_VALUES=10
export TAKE_N=64
# Training used beft_hidden_state_offset=4, selecting L - 4 - 1 = L - 5.
export DEFAULT_PRIOR_TOKEN_OFFSET="${DEFAULT_PRIOR_TOKEN_OFFSET:--5}"
export PRUNING_TOP_P="${PRUNING_TOP_P:-1.0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_TOKENS="${MAX_TOKENS:-64}"
export NUM_ACCUMULATOR_ROWS="${NUM_ACCUMULATOR_ROWS:-512}"
unset DO_TOP_P_PRUNE SET_TOP_P_WITH_K ENSURE_GT DRY_RUN

EXP_DIR="${OUTPUT_DIR}/${MODEL_SLUG}/berag_prior/K=10-TakeN=64"

bash "${REPO_ROOT}/scripts/infer/evqa/test_full/run_berag_prior.sh"

# BEM has a separate TensorFlow environment from vLLM inference.
source "${EVAL_VENV}/bin/activate"
export PYTHON="${EVAL_VENV}/bin/python"
bash "${EVAL_SCRIPT}" "${EXP_DIR}"
