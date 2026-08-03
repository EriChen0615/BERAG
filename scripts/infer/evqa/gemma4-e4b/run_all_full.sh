#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SLUG="${MODEL_SLUG:-gemma4-e4b}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/evqa_full}"
EVAL_VENV="${EVAL_VENV:-${REPO_ROOT}/src/eval/.venv}"
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval/evqa/evaluate_infer_outputs.sh"

if [[ ! -f "${EVAL_VENV}/bin/activate" || ! -x "${EVAL_VENV}/bin/python" ]]; then
    echo "E-VQA eval environment is incomplete: ${EVAL_VENV}" >&2
    exit 1
fi
if [[ ! -x "${EVAL_SCRIPT}" ]]; then
    echo "E-VQA eval script not found or not executable: ${EVAL_SCRIPT}" >&2
    exit 1
fi

export REPO_ROOT MODEL_SLUG OUTPUT_DIR

bash "${SCRIPT_DIR}/run_pretrained_full.sh"
bash "${SCRIPT_DIR}/run_sft_full.sh"
bash "${SCRIPT_DIR}/run_beft_full.sh"

source "${EVAL_VENV}/bin/activate"
export PYTHON="${EVAL_VENV}/bin/python"
env -u K_VALUES -u EXP_NAMES \
    BASE_DIR="${OUTPUT_DIR}/${MODEL_SLUG}" PYTHON="${PYTHON}" \
    bash "${EVAL_SCRIPT}" rag-base rag berag_prior
