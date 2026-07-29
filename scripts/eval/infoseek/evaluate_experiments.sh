#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
PYTHON="${PYTHON:-python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/src/eval/evaluate_infoseek_predictions.py}"
TOOLKIT_DIR="${TOOLKIT_DIR:-${REPO_ROOT}/third_party/infoseek_eval}"
REFERENCE_FILE="${REFERENCE_FILE:-${TOOLKIT_DIR}/infoseek/infoseek_val.jsonl}"
REFERENCE_QTYPE_FILE="${REFERENCE_QTYPE_FILE:-${TOOLKIT_DIR}/infoseek/infoseek_val_qtype.jsonl}"
PREDICTIONS_FILENAME="${PREDICTIONS_FILENAME:-predictions.jsonl}"
SCORES_FILENAME="${SCORES_FILENAME:-score.json}"
MARKED_FILENAME="${MARKED_FILENAME:-marked_inference_results.csv}"
INSTANCES_FILENAME="${INSTANCES_FILENAME:-evaluated_instances.csv}"
OFFICIAL_PREDICTIONS_FILENAME="${OFFICIAL_PREDICTIONS_FILENAME:-infoseek_predictions.jsonl}"

usage() {
    cat <<'EOF'
Usage:
  scripts/eval/infoseek/evaluate_experiments.sh EXP_DIR [EXP_DIR ...]

Each EXP_DIR must contain predictions.jsonl by default. The script writes
score.json, infoseek_predictions.jsonl, marked_inference_results.csv, and
evaluated_instances.csv under each experiment directory.

Environment overrides:
  PYTHON, TOOLKIT_DIR, REFERENCE_FILE, REFERENCE_QTYPE_FILE,
  PREDICTIONS_FILENAME, SCORES_FILENAME, MARKED_FILENAME,
  INSTANCES_FILENAME, OFFICIAL_PREDICTIONS_FILENAME
  PREDICTION_FIELD       Default: auto
  LIMIT                  Optional input row limit
  ALLOW_UNKNOWN_IDS=1    Ignore prediction IDs absent from the references
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "$#" -eq 0 ]]; then
    usage >&2
    exit 1
fi

for required_file in "${EVAL_SCRIPT}" "${TOOLKIT_DIR}/infoseek_eval.py" "${REFERENCE_FILE}" "${REFERENCE_QTYPE_FILE}"; do
    [[ -f "${required_file}" ]] || { echo "Required file not found: ${required_file}" >&2; exit 1; }
done

optional_args=()
[[ -n "${LIMIT:-}" ]] && optional_args+=(--limit "${LIMIT}")
[[ "${ALLOW_UNKNOWN_IDS:-0}" == "1" ]] && optional_args+=(--allow_unknown_ids)

for EXP_DIR in "$@"; do
    if [[ ! -d "${EXP_DIR}" ]]; then
        echo "[InfoSeek eval] Skipping missing directory: ${EXP_DIR}" >&2
        continue
    fi

    PREDICTIONS_FILE="${EXP_DIR}/${PREDICTIONS_FILENAME}"
    if [[ ! -f "${PREDICTIONS_FILE}" ]]; then
        echo "[InfoSeek eval] Skipping ${EXP_DIR}; missing ${PREDICTIONS_FILENAME}" >&2
        continue
    fi

    echo "[InfoSeek eval] Evaluating: ${EXP_DIR}"
    echo "[InfoSeek eval] Predictions: ${PREDICTIONS_FILE}"
    echo "[InfoSeek eval] Scores: ${EXP_DIR}/${SCORES_FILENAME}"

    "${PYTHON}" "${EVAL_SCRIPT}" \
        --predictions_file "${PREDICTIONS_FILE}" \
        --output_dir "${EXP_DIR}" \
        --prediction_field "${PREDICTION_FIELD:-auto}" \
        --toolkit_dir "${TOOLKIT_DIR}" \
        --reference_file "${REFERENCE_FILE}" \
        --reference_qtype_file "${REFERENCE_QTYPE_FILE}" \
        --scores_filename "${SCORES_FILENAME}" \
        --marked_filename "${MARKED_FILENAME}" \
        --instances_filename "${INSTANCES_FILENAME}" \
        --official_predictions_filename "${OFFICIAL_PREDICTIONS_FILENAME}" \
        "${optional_args[@]}"
done
