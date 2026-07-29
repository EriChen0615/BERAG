#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

RUN_MODE="${RUN_MODE:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFERENCE_SCRIPT="${REPO_ROOT}/src/infer/infoseek_vllm_berag_inference.py"

RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:-${REPO_ROOT}/outputs/jinghong_chen/Infoseek-testfull-with-retrieval}"
KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-/workspace/projects/Infoseek-retrieval/data/echosight/infoseek_100k_wiki/wiki_100_dict_v4.json}"
IMG_BASEDIR="${IMG_BASEDIR:-${REPO_ROOT}/src/train/LlamaFactory-0.9.5-beft/data/Infoseek/infoseek_val_images}"

SFT_MODEL="${SFT_MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-SFT-Infoseek64000}"
BEFT_MODEL="${BEFT_MODEL:-${REPO_ROOT}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-Infoseek-ckpt8792}"
PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${BEFT_MODEL}/prior_head.pt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-VL-8B-Instruct}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/infer/infoseek_full}"
MODEL_SLUG="${MODEL_SLUG:-qwen3-vl-8b}"
K_VALUES="${K_VALUES:-${RETRIEVAL_TOPK:-5}}"
K_VALUES="${K_VALUES//,/ }"
read -r -a K_ARRAY <<< "${K_VALUES}"

usage() {
    echo "Usage: RUN_MODE=all|sft|beft [K_VALUES=2,5] [ADD_PSEUDO_DOCUMENT=1] [CONTINUE_GENERATION_FROM_ANS=1] bash $0" >&2
}

case "${RUN_MODE}" in
    all|sft|beft) ;;
    *) usage; exit 2 ;;
esac

[[ "${#K_ARRAY[@]}" -gt 0 ]] || {
    echo "No K values supplied." >&2
    exit 1
}
[[ -f "${INFERENCE_SCRIPT}" ]] || { echo "Missing ${INFERENCE_SCRIPT}" >&2; exit 1; }
[[ -f "${RETRIEVAL_DS_PATH}/state.json" ]] || { echo "Missing Hugging Face dataset at ${RETRIEVAL_DS_PATH}" >&2; exit 1; }
[[ -f "${KNOWLEDGE_BASE}" ]] || { echo "Missing EchoSight KB at ${KNOWLEDGE_BASE}" >&2; exit 1; }
[[ -d "${IMG_BASEDIR}" ]] || { echo "Missing InfoSeek image directory at ${IMG_BASEDIR}" >&2; exit 1; }

require_merged_model() {
    local label="$1"
    local model_path="$2"
    if [[ "${model_path}" == /* && ! -f "${model_path}/config.json" ]]; then
        echo "Missing merged ${label} model at ${model_path}" >&2
        echo "Export it first with: bash ${REPO_ROOT}/scripts/infer/infoseek/export_models.sh" >&2
        exit 1
    fi
}

common_args=(
    --retrieval-ds-path "${RETRIEVAL_DS_PATH}"
    --knowledge-base "${KNOWLEDGE_BASE}"
    --img-basedir "${IMG_BASEDIR}"
    --retrieval-field "${RETRIEVAL_FIELD:-retrieved_passage}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --max-words-per-evidence "${MAX_WORDS_PER_EVIDENCE:-2048}"
    --max-tokens "${MAX_TOKENS:-64}"
    --dtype "${DTYPE:-bfloat16}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
    --resume-from-jsonl
    --write-on-batch-complete
)

[[ -n "${TAKE_N:-}" ]] && common_args+=(--take-n "${TAKE_N}")
[[ "${ENSURE_GT:-0}" == "1" ]] && common_args+=(--ensure-gt-passage-in-ensemble)
[[ "${ENFORCE_EAGER:-0}" == "1" ]] && common_args+=(--enforce-eager)
[[ "${DISABLE_TQDM:-0}" == "1" ]] && common_args+=(--disable-tqdm)
[[ "${SAVE_FULL_PROMPTS:-0}" == "1" ]] && common_args+=(--save-full-prompts)
[[ -n "${MAX_NUM_SEQS:-}" ]] && common_args+=(--max-num-seqs "${MAX_NUM_SEQS}")
[[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] && common_args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")

output_suffix=""
case "${ADD_PSEUDO_DOCUMENT:-0}" in
    0) ;;
    1)
        common_args+=(--add-pseudo-document)
        output_suffix+="-pseudo-document"
        ;;
    *)
        echo "ADD_PSEUDO_DOCUMENT must be 0 or 1." >&2
        exit 2
        ;;
esac

case "${CONTINUE_GENERATION_FROM_ANS:-0}" in
    0) ;;
    1)
        common_args+=(--continue_generation_from_ans)
        output_suffix+="-continue-from-answer"
        ;;
    *)
        echo "CONTINUE_GENERATION_FROM_ANS must be 0 or 1." >&2
        exit 2
        ;;
esac

run_sft() {
    require_merged_model "SFT" "${SFT_MODEL}"
    for k in "${K_ARRAY[@]}"; do
        local exp_dir="${OUTPUT_DIR}/${MODEL_SLUG}/sft/K=${k}${output_suffix}"
        mkdir -p "${exp_dir}"
        echo "[InfoSeek SFT/RAG] K=${k}; output=${exp_dir}/predictions.jsonl"
        "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
            --mode rag \
            --model "${SFT_MODEL}" \
            --retrieval-topk "${k}" \
            --output-path "${exp_dir}/predictions.jsonl" \
            --batch-size "${SFT_BATCH_SIZE:-1}" \
            --max-model-len "${SFT_MAX_MODEL_LEN:-65536}" \
            "${common_args[@]}"
    done
}

run_beft() {
    require_merged_model "BEFT" "${BEFT_MODEL}"
    [[ -f "${PRIOR_HEAD_PATH}" ]] || { echo "Missing prior head at ${PRIOR_HEAD_PATH}" >&2; exit 1; }
    for k in "${K_ARRAY[@]}"; do
        local exp_dir="${OUTPUT_DIR}/${MODEL_SLUG}/beft/K=${k}${output_suffix}"
        mkdir -p "${exp_dir}"
        echo "[InfoSeek BEFT] K=${k}; output=${exp_dir}/predictions.jsonl"
        "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
            --mode berag \
            --model "${BEFT_MODEL}" \
            --prior-mode module \
            --prior-head-path "${PRIOR_HEAD_PATH}" \
            --prior-modeling "${PRIOR_MODELING:-mlp_head}" \
            --prior-head-num-layers "${PRIOR_HEAD_NUM_LAYERS:-2}" \
            --prior-head-proj-dim "${PRIOR_HEAD_PROJ_DIM:-1024}" \
            --default-prior-token-offset "${DEFAULT_PRIOR_TOKEN_OFFSET:--5}" \
            --num-accumulator-rows "${NUM_ACCUMULATOR_ROWS:-512}" \
            --pruning-top-p "${PRUNING_TOP_P:-1.0}" \
            --retrieval-topk "${k}" \
            --output-path "${exp_dir}/predictions.jsonl" \
            --batch-size "${BEFT_BATCH_SIZE:-16}" \
            --max-model-len "${BEFT_MAX_MODEL_LEN:-32768}" \
            "${common_args[@]}"
    done
}

[[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "sft" ]] && run_sft
[[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "beft" ]] && run_beft
