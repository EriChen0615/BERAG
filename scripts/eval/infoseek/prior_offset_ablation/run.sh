#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
INFER_SCRIPT="${REPO_ROOT}/scripts/infer/infoseek/run_sft_beft.sh"
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval/infoseek/evaluate_experiments.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

BEFT_MODEL="${BEFT_MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-Infoseek-ckpt8792}"
PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${BEFT_MODEL}/prior_head.pt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-VL-8B-Instruct}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/infer/infoseek_prior_offset_ablation_take64}"
MODEL_SLUG="${MODEL_SLUG:-qwen3-vl-8b}"
K_VALUES="${K_VALUES:-10,20,30}"
TAKE_N="${TAKE_N:-64}"
OFFSETS="${OFFSETS:--4,-5}"

[[ -f "${INFER_SCRIPT}" ]] || { echo "Missing inference script: ${INFER_SCRIPT}" >&2; exit 1; }
[[ -f "${EVAL_SCRIPT}" ]] || { echo "Missing evaluation script: ${EVAL_SCRIPT}" >&2; exit 1; }
[[ "${TAKE_N}" =~ ^[1-9][0-9]*$ ]] || { echo "TAKE_N must be a positive integer." >&2; exit 2; }

normalized_k_values="${K_VALUES//,/ }"
normalized_offsets="${OFFSETS//,/ }"
read -r -a k_array <<< "${normalized_k_values}"
read -r -a offset_array <<< "${normalized_offsets}"
[[ "${#k_array[@]}" -gt 0 ]] || { echo "No K values supplied." >&2; exit 2; }
[[ "${#offset_array[@]}" -gt 0 ]] || { echo "No offsets supplied." >&2; exit 2; }

experiment_dirs=()

for offset in "${offset_array[@]}"; do
    [[ "${offset}" =~ ^-[1-9][0-9]*$ ]] || {
        echo "Offsets must be negative integers; received: ${offset}" >&2
        exit 2
    }

    offset_root="${OUTPUT_ROOT}/offset=${offset}"
    echo "[InfoSeek offset ablation] Running offset=${offset}, K=${K_VALUES}, TAKE_N=${TAKE_N}"

    RUN_MODE=beft \
    BEFT_MODEL="${BEFT_MODEL}" \
    PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    OUTPUT_DIR="${offset_root}" \
    MODEL_SLUG="${MODEL_SLUG}" \
    K_VALUES="${K_VALUES}" \
    TAKE_N="${TAKE_N}" \
    CONTINUE_GENERATION_FROM_ANS=1 \
    ADD_PSEUDO_DOCUMENT=0 \
    DEFAULT_PRIOR_TOKEN_OFFSET="${offset}" \
    PYTHON_BIN="${PYTHON_BIN}" \
        bash "${INFER_SCRIPT}"

    for k in "${k_array[@]}"; do
        experiment_dirs+=(
            "${offset_root}/${MODEL_SLUG}/beft/K=${k}-continue-from-answer"
        )
    done
done

echo "[InfoSeek offset ablation] Evaluating ${#experiment_dirs[@]} runs"
PYTHON="${PYTHON_BIN}" bash "${EVAL_SCRIPT}" "${experiment_dirs[@]}"

summary_path="${OUTPUT_ROOT}/summary.tsv"
"${PYTHON_BIN}" - "${summary_path}" "${experiment_dirs[@]}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
experiment_dirs = [Path(path) for path in sys.argv[2:]]
columns = (
    "offset",
    "K",
    "num_examples",
    "mean_example_score",
    "unseen_question_score",
    "unseen_entity_score",
    "retrieval_recall_at_K",
    "prior_recall_at_1",
    "posterior_recall_at_1",
)
lines = ["\t".join(columns)]

for experiment_dir in experiment_dirs:
    score = json.loads((experiment_dir / "score.json").read_text())
    offset = next(
        part.split("=", 1)[1]
        for part in experiment_dir.parts
        if part.startswith("offset=")
    )
    k = experiment_dir.name.split("=", 1)[1].split("-", 1)[0]
    values = (
        offset,
        k,
        score.get("num_examples"),
        score.get("mean_example_score"),
        score.get("unseen_question_score"),
        score.get("unseen_entity_score"),
        score.get("retrieval_recall_at_k", {}).get(k),
        score.get("prior_recall_at_k", {}).get("1"),
        score.get("posterior_recall_at_k", {}).get("1"),
    )
    lines.append("\t".join("" if value is None else str(value) for value in values))

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Summary saved to {summary_path}")
PY
