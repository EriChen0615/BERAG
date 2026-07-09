#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

set -euo pipefail

MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-data/jinghong_chen/LLaVA-Llama3-8B-v1_1-transformers_InfoseekNew-RAG5_LoRA-SFT}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${MERGED_MODEL_PATH}}"
RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:-outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B_post_reranked}"
RETRIEVE_FIELD="${RETRIEVE_FIELD:-retrieved_passage}"
RETRIEVAL_TOPK_LIST=(1 2 3 5 10 15 20 30 50)
TAKE_N="${TAKE_N:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
MAX_PIXELS="${MAX_PIXELS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/0526/InfoseekNew/LLaVA-Llama3/VLLM/sft_merged}"
USE_CACHE="${USE_CACHE:-false}"
DO_EVAL="${DO_EVAL:-false}"
ENSURE_GT_PASSAGE_IN_ENSEMBLE="${ENSURE_GT_PASSAGE_IN_ENSEMBLE:-false}"

if [[ "${DO_EVAL}" == "true" ]]; then
    echo "Infoseek evaluation is not implemented in src/vllm_vqa_inference.py; run inference only or add an eval path first." >&2
    exit 1
fi

if [[ ! -d "${MERGED_MODEL_PATH}" ]]; then
    echo "Missing merged SFT model directory: ${MERGED_MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -d "${RETRIEVAL_DS_PATH}" ]]; then
    echo "Missing retrieval dataset path: ${RETRIEVAL_DS_PATH}" >&2
    exit 1
fi

generate_exp_name() {
    local topk="$1"
    local name="InfoseekNew-VLLM-LLaVA-Llama3-SFT-Merged-Top${topk}"
    if [[ "${ENSURE_GT_PASSAGE_IN_ENSEMBLE}" == "true" ]]; then
        name="${name}-hasGTdoc"
    fi
    if [[ "${RETRIEVE_FIELD}" == "reranked_passage" ]]; then
        name="${name}-Rerank"
    else
        name="${name}-Retrieve"
    fi
    name="${name}-TakeN=${TAKE_N}"
    echo "${name}"
}

for retrieval_topk in "${RETRIEVAL_TOPK_LIST[@]}"; do
    full_exp_name="$(generate_exp_name "${retrieval_topk}")"

    echo "--------------------------------"
    echo "Running inference for ${full_exp_name}"
    echo "Merged model path: ${MERGED_MODEL_PATH}"
    echo "Processor path: ${PROCESSOR_PATH}"
    echo "Retrieval topk: ${retrieval_topk}"
    echo "Retrieval field: ${RETRIEVE_FIELD}"
    echo "Take N: ${TAKE_N}"
    echo "Batch size: ${BATCH_SIZE}"

    args=(
        --retrieval_ds_path "${RETRIEVAL_DS_PATH}"
        --dataset_name "InfoseekNew_FullPassage"
        --take_n "${TAKE_N}"
        --img_basedir "."
        --retrieval_field "${RETRIEVE_FIELD}"
        --retrieval_topk "${retrieval_topk}"
        --base_model_path "${MERGED_MODEL_PATH}"
        --processor_path "${PROCESSOR_PATH}"
        --model_family "llava"
        --seed 0
        --batch_size "${BATCH_SIZE}"
        --max_model_len "${MAX_MODEL_LEN}"
        --max_tokens "${MAX_TOKENS}"
        --exp_name "${OUTPUT_ROOT}/${full_exp_name}"
    )

    if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
        args+=(--tensor_parallel_size "${TENSOR_PARALLEL_SIZE}")
    fi

    if [[ -n "${MAX_PIXELS}" ]]; then
        args+=(--max_pixels "${MAX_PIXELS}")
    fi

    if [[ "${USE_CACHE}" == "true" ]]; then
        args+=(--use_cache)
    fi

    if [[ "${ENSURE_GT_PASSAGE_IN_ENSEMBLE}" == "true" ]]; then
        args+=(--ensure_gt_passage_in_ensemble)
    fi

    echo "Args: ${args[*]}"
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/vllm_vqa_inference.py "${args[@]}"

    echo "Finished inference for ${full_exp_name}"
    echo "--------------------------------"
done
