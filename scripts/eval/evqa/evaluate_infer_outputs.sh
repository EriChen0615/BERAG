#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/outputs/infer/evqa}"
EVAL_EXPERIMENTS_SCRIPT="${EVAL_EXPERIMENTS_SCRIPT:-${REPO_ROOT}/scripts/eval/evqa/evaluate_experiments.sh}"
PREDICTIONS_FILENAME="${PREDICTIONS_FILENAME:-predictions.jsonl}"
SCORES_FILENAME="${SCORES_FILENAME:-score.json}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage:
  scripts/eval/evqa/evaluate_infer_outputs.sh [SELECTOR ...]

Selectors are resolved under outputs/infer/evqa by default:
  berag_prior          Evaluate every experiment folder under berag_prior/
  rag                  Evaluate every experiment folder under rag/
  berag_prior/K=1      Evaluate one exact experiment folder
  /abs/path/to/K=1     Evaluate one exact experiment folder

No selector defaults to: berag_prior rag

Environment overrides:
  BASE_DIR              Default: ${REPO_ROOT}/outputs/infer/evqa
  K_VALUES              Optional K filter, e.g. "1,2,3,5" or "1 2 3 5"
  EXP_NAMES             Optional exact experiment names, e.g. "K=1 K=2-TakeN=10"
  SKIP_EXISTING         Set to 1 to skip dirs with an existing score file
  DRY_RUN               Set to 1 to print selected dirs without evaluating
  SCORES_FILENAME       Default: score.json

Pass-through env vars for evaluate_experiments.sh:
  PYTHON, BEM_MODEL_PATH, VOCAB_PATH, EVAL_DEVICE, NUM_PROCESSES, LIMIT,
  MARKED_FILENAME, INSTANCES_FILENAME, PREDICTIONS_FILENAME

Examples:
  source src/eval/bem/bin/activate

  # Evaluate all discovered berag_prior and rag runs.
  scripts/eval/evqa/evaluate_infer_outputs.sh

  # Evaluate only K=1 and K=2 for both groups.
  K_VALUES=1,2 scripts/eval/evqa/evaluate_infer_outputs.sh berag_prior rag

  # Evaluate a single exact run.
  scripts/eval/evqa/evaluate_infer_outputs.sh berag_prior/K=1
EOF
}

normalize_list() {
    local raw="${1:-}"
    raw="${raw//,/ }"
    echo "${raw}"
}

add_dir_if_valid() {
    local candidate="$1"
    if [[ ! -d "${candidate}" ]]; then
        echo "[EVQA eval select] Skipping missing directory: ${candidate}" >&2
        return
    fi
    if [[ ! -f "${candidate}/${PREDICTIONS_FILENAME}" ]]; then
        echo "[EVQA eval select] Skipping ${candidate}; missing ${PREDICTIONS_FILENAME}" >&2
        return
    fi
    if [[ "${SKIP_EXISTING}" == "1" && -f "${candidate}/${SCORES_FILENAME}" ]]; then
        echo "[EVQA eval select] Skipping ${candidate}; ${SCORES_FILENAME} already exists" >&2
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

if [[ ! -x "${EVAL_EXPERIMENTS_SCRIPT}" ]]; then
    echo "Eval wrapper is not executable or not found: ${EVAL_EXPERIMENTS_SCRIPT}" >&2
    exit 1
fi

SELECTORS=("$@")
if [[ "${#SELECTORS[@]}" -eq 0 ]]; then
    SELECTORS=("berag_prior" "rag")
fi

EXP_DIRS=()

for selector in "${SELECTORS[@]}"; do
    if [[ "${selector}" == /* ]]; then
        if [[ -f "${selector}" ]]; then
            add_dir_if_valid "$(dirname "${selector}")"
        else
            add_dir_if_valid "${selector}"
        fi
        continue
    fi

    candidate="${BASE_DIR}/${selector}"
    if [[ -f "${candidate}" ]]; then
        add_dir_if_valid "$(dirname "${candidate}")"
    elif [[ -f "${candidate}/${PREDICTIONS_FILENAME}" ]]; then
        add_dir_if_valid "${candidate}"
    elif [[ -d "${candidate}" ]]; then
        add_group_dirs "${candidate}"
    else
        echo "[EVQA eval select] Unknown selector: ${selector} (${candidate} not found)" >&2
    fi
done

if [[ "${#EXP_DIRS[@]}" -eq 0 ]]; then
    echo "[EVQA eval select] No experiment directories selected." >&2
    exit 1
fi

echo "[EVQA eval select] Selected ${#EXP_DIRS[@]} experiment directories:"
printf '  %s\n' "${EXP_DIRS[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

"${EVAL_EXPERIMENTS_SCRIPT}" "${EXP_DIRS[@]}"
