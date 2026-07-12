#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
PYTHON="${PYTHON:-python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/src/eval/evaluate_evqa_predictions.py}"
BEM_MODEL_PATH="${BEM_MODEL_PATH:-${REPO_ROOT}/src/eval/models/bem}"
VOCAB_PATH="${VOCAB_PATH:-${REPO_ROOT}/src/eval/models/vocab.txt}"
PREDICTIONS_FILENAME="${PREDICTIONS_FILENAME:-predictions.jsonl}"
SCORES_FILENAME="${SCORES_FILENAME:-score.json}"
MARKED_FILENAME="${MARKED_FILENAME:-marked_inference_results.csv}"
INSTANCES_FILENAME="${INSTANCES_FILENAME:-evaluated_instances.csv}"
EVAL_DEVICE="${EVAL_DEVICE:-gpu}"
if [[ -z "${NUM_PROCESSES+x}" ]]; then
    if [[ "${EVAL_DEVICE}" == "gpu" ]]; then
        NUM_PROCESSES=1
    else
        NUM_PROCESSES=4
    fi
fi

usage() {
    cat <<'EOF'
Usage:
  scripts/eval/evqa/evaluate_experiments.sh EXP_DIR [EXP_DIR ...]

Each EXP_DIR must contain predictions.jsonl by default. The script writes
score.json and marked_inference_results.csv under each EXP_DIR.

Environment overrides:
  PYTHON                 Python executable to use. Default: python
  BEM_MODEL_PATH         Default: ${REPO_ROOT}/src/eval/models/bem
  VOCAB_PATH             Default: ${REPO_ROOT}/src/eval/models/vocab.txt
  EVAL_DEVICE            cpu or gpu. Default: cpu
  NUM_PROCESSES          Default: 4 for CPU, 1 for GPU
  PREDICTIONS_FILENAME   Default: predictions.jsonl
  SCORES_FILENAME        Default: score.json
  MARKED_FILENAME        Default: marked_inference_results.csv
  INSTANCES_FILENAME     Default: evaluated_instances.csv
  LIMIT                  Optional row limit for smoke tests

Example:
  source src/eval/bem/bin/activate
  scripts/eval/evqa/evaluate_experiments.sh \
    outputs/infer/evqa/berag_prior/K=1 \
    outputs/infer/evqa/berag_prior/K=2
EOF
}

if [[ "$#" -eq 0 ]]; then
    usage >&2
    exit 1
fi

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Evaluator not found: ${EVAL_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${BEM_MODEL_PATH}" ]]; then
    echo "BEM model path does not exist: ${BEM_MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${VOCAB_PATH}" ]]; then
    echo "BEM vocab file does not exist: ${VOCAB_PATH}" >&2
    exit 1
fi

for EXP_DIR in "$@"; do
    if [[ ! -d "${EXP_DIR}" ]]; then
        echo "[EVQA eval] Skipping missing directory: ${EXP_DIR}" >&2
        continue
    fi

    PREDICTIONS_FILE="${EXP_DIR}/${PREDICTIONS_FILENAME}"
    if [[ ! -f "${PREDICTIONS_FILE}" ]]; then
        echo "[EVQA eval] Skipping ${EXP_DIR}; missing ${PREDICTIONS_FILENAME}" >&2
        continue
    fi

    echo "[EVQA eval] Evaluating: ${EXP_DIR}"
    echo "[EVQA eval] Predictions: ${PREDICTIONS_FILE}"
    echo "[EVQA eval] Scores: ${EXP_DIR}/${SCORES_FILENAME}"

    "${PYTHON}" "${EVAL_SCRIPT}" \
        --predictions_file "${PREDICTIONS_FILE}" \
        --output_dir "${EXP_DIR}" \
        --bem_model_path "${BEM_MODEL_PATH}" \
        --vocab_path "${VOCAB_PATH}" \
        --num_processes "${NUM_PROCESSES}" \
        --device "${EVAL_DEVICE}" \
        --scores_filename "${SCORES_FILENAME}" \
        --marked_filename "${MARKED_FILENAME}" \
        --instances_filename "${INSTANCES_FILENAME}" \
        ${LIMIT:+--limit "${LIMIT}"}
done
