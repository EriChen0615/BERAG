#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/outputs/infer/infoseek_full}"
MODEL_SLUG="${MODEL_SLUG:-qwen3-vl-8b}"
EVAL_EXPERIMENTS_SCRIPT="${EVAL_EXPERIMENTS_SCRIPT:-${REPO_ROOT}/scripts/eval/infoseek/evaluate_experiments.sh}"
PREDICTIONS_FILENAME="${PREDICTIONS_FILENAME:-predictions.jsonl}"
SCORES_FILENAME="${SCORES_FILENAME:-score.json}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage:
  scripts/eval/infoseek/evaluate_infer_outputs.sh [SELECTOR ...]

Selectors are resolved under outputs/infer/infoseek_full by default:
  sft                  Evaluate every qwen3-vl-8b/sft experiment
  beft                 Evaluate every qwen3-vl-8b/beft experiment
  sft/K=2              Evaluate one exact experiment
  qwen3-vl-8b/sft      Evaluate every experiment under an explicit model
  /abs/path/to/K=2     Evaluate one exact experiment

No selector defaults to: sft beft

Environment overrides:
  BASE_DIR, MODEL_SLUG, K_VALUES, EXP_NAMES, SKIP_EXISTING, DRY_RUN,
  PREDICTIONS_FILENAME, SCORES_FILENAME

Pass-through variables for evaluate_experiments.sh:
  PYTHON, TOOLKIT_DIR, REFERENCE_FILE, REFERENCE_QTYPE_FILE,
  PREDICTION_FIELD, LIMIT, ALLOW_UNKNOWN_IDS, MARKED_FILENAME,
  INSTANCES_FILENAME, OFFICIAL_PREDICTIONS_FILENAME

Examples:
  scripts/eval/infoseek/evaluate_infer_outputs.sh
  K_VALUES=2,5 scripts/eval/infoseek/evaluate_infer_outputs.sh sft beft
  scripts/eval/infoseek/evaluate_infer_outputs.sh sft/K=2
EOF
}

normalize_list() {
    local raw="${1:-}"
    echo "${raw//,/ }"
}

add_dir_if_valid() {
    local candidate="$1"
    if [[ ! -d "${candidate}" ]]; then
        echo "[InfoSeek eval select] Skipping missing directory: ${candidate}" >&2
        return
    fi
    if [[ ! -f "${candidate}/${PREDICTIONS_FILENAME}" ]]; then
        echo "[InfoSeek eval select] Skipping ${candidate}; missing ${PREDICTIONS_FILENAME}" >&2
        return
    fi
    if [[ "${SKIP_EXISTING}" == "1" && -f "${candidate}/${SCORES_FILENAME}" ]]; then
        echo "[InfoSeek eval select] Skipping ${candidate}; ${SCORES_FILENAME} already exists" >&2
        return
    fi
    EXP_DIRS+=("${candidate}")
}

add_group_dirs() {
    local group_dir="$1"
    if [[ -n "${EXP_NAMES:-}" ]]; then
        local exp_name
        for exp_name in $(normalize_list "${EXP_NAMES}"); do
            add_dir_if_valid "${group_dir}/${exp_name}"
        done
        return
    fi
    if [[ -n "${K_VALUES:-}" ]]; then
        local k
        for k in $(normalize_list "${K_VALUES}"); do
            add_dir_if_valid "${group_dir}/K=${k}"
        done
        return
    fi

    local subdir
    shopt -s nullglob
    for subdir in "${group_dir}"/*; do
        [[ -d "${subdir}" ]] || continue
        add_dir_if_valid "${subdir}"
    done
    shopt -u nullglob
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
[[ -x "${EVAL_EXPERIMENTS_SCRIPT}" ]] || {
    echo "Eval wrapper is not executable or not found: ${EVAL_EXPERIMENTS_SCRIPT}" >&2
    exit 1
}

SELECTORS=("$@")
[[ "${#SELECTORS[@]}" -gt 0 ]] || SELECTORS=("sft" "beft")
EXP_DIRS=()

for selector in "${SELECTORS[@]}"; do
    if [[ "${selector}" == /* ]]; then
        [[ -f "${selector}" ]] && add_dir_if_valid "$(dirname "${selector}")" || add_dir_if_valid "${selector}"
        continue
    fi

    candidate="${BASE_DIR}/${selector}"
    model_candidate="${BASE_DIR}/${MODEL_SLUG}/${selector}"
    if [[ -f "${candidate}" ]]; then
        add_dir_if_valid "$(dirname "${candidate}")"
    elif [[ -f "${candidate}/${PREDICTIONS_FILENAME}" ]]; then
        add_dir_if_valid "${candidate}"
    elif [[ -d "${candidate}" ]]; then
        add_group_dirs "${candidate}"
    elif [[ -f "${model_candidate}/${PREDICTIONS_FILENAME}" ]]; then
        add_dir_if_valid "${model_candidate}"
    elif [[ -d "${model_candidate}" ]]; then
        add_group_dirs "${model_candidate}"
    else
        echo "[InfoSeek eval select] Unknown selector: ${selector}" >&2
    fi
done

if [[ "${#EXP_DIRS[@]}" -eq 0 ]]; then
    echo "[InfoSeek eval select] No experiment directories selected." >&2
    exit 1
fi

echo "[InfoSeek eval select] Selected ${#EXP_DIRS[@]} experiment directories:"
printf '  %s\n' "${EXP_DIRS[@]}"
[[ "${DRY_RUN}" == "1" ]] && exit 0

"${EVAL_EXPERIMENTS_SCRIPT}" "${EXP_DIRS[@]}"
