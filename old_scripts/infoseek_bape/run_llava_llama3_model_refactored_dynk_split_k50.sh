#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --array=0-4

set -euo pipefail

# Usage:
#   sbatch scripts/infoseek_bape/run_llava_llama3_model_refactored_noprune_split_k50.sh
#   CHECKPOINT_PATH=/path/to/checkpoint sbatch scripts/infoseek_bape/run_llava_llama3_model_refactored_noprune_split_k50.sh

BASE_MODEL_PATH="${BASE_MODEL_PATH:-xtuner/llava-llama-3-8b-v1_1-transformers}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-third_party/LLaMA-Factory-2502/saves/llava_llama3_8b_v1_1_transformers/lora/infoseek_new/beft/rag2-k2-prior=mlp-lr1e-6-h4-r64-size64000-max4096/checkpoint-4500}"
ADAPTER_PATH="${ADAPTER_PATH:-${CHECKPOINT_PATH}}"
PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${CHECKPOINT_PATH}/prior_head.pt}"

if [[ -f "${CHECKPOINT_PATH}/processor_config.json" || -f "${CHECKPOINT_PATH}/preprocessor_config.json" ]]; then
    PROCESSOR_PATH="${PROCESSOR_PATH:-${CHECKPOINT_PATH}}"
else
    PROCESSOR_PATH="${PROCESSOR_PATH:-${BASE_MODEL_PATH}}"
fi

INCLUDE_Z0_IN_ENSEMBLE="${INCLUDE_Z0_IN_ENSEMBLE:-false}"
ENSURE_GT_PASSAGE_IN_ENSEMBLE="${ENSURE_GT_PASSAGE_IN_ENSEMBLE:-false}"
RETRIEVAL_TOPK_LIST=(50)
PASSAGE_PRIOR="${PASSAGE_PRIOR:-prior_head}"
RETRIEVE_FIELD="${RETRIEVE_FIELD:-retrieved_passage}"
RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:-outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B_post_reranked}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/0526/InfoseekNew/LLaVA-Llama3/BAPE}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_BATCH_SIZE_PER_FORWARD="${MAX_BATCH_SIZE_PER_FORWARD:-5}"
MAX_WORDS_PER_EVIDENCE="${MAX_WORDS_PER_EVIDENCE:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HIDDEN_STATE_OFFSET="${HIDDEN_STATE_OFFSET:-4}"
PRIOR_HEAD_MODELING="${PRIOR_HEAD_MODELING:-mlp_head}"
PRIOR_HEAD_NUM_LAYERS="${PRIOR_HEAD_NUM_LAYERS:-2}"
USE_CACHE="${USE_CACHE:-false}"
DO_EVAL="${DO_EVAL:-false}"
PREFILL_ANS_TOKEN="${PREFILL_ANS_TOKEN:-false}"

# Chunks cover 4708 Infoseek validation examples.
OFFSETS=(${OFFSETS:-0 1000 2000 3000 4000})
TAKE_N_LIST=(${TAKE_N_LIST:-1000 1000 1000 1000 708})
CHUNK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TAKE_N="${TAKE_N_LIST[$CHUNK_INDEX]:-}"
DS_OFFSET="${OFFSETS[$CHUNK_INDEX]:-}"

if [[ -z "${DS_OFFSET}" || -z "${TAKE_N}" ]]; then
    echo "Invalid chunk index: ${CHUNK_INDEX}. Check OFFSETS/TAKE_N_LIST." >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
    echo "Missing checkpoint directory: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    echo "Missing adapter_config.json in ${ADAPTER_PATH}" >&2
    exit 1
fi

if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" && ! -f "${ADAPTER_PATH}/adapter_model.bin" ]]; then
    echo "Missing LoRA adapter weights in ${ADAPTER_PATH}: expected adapter_model.safetensors or adapter_model.bin" >&2
    exit 1
fi

if [[ "${PASSAGE_PRIOR}" == "prior_head" && ! -f "${PRIOR_HEAD_PATH}" ]]; then
    echo "Missing prior head for PASSAGE_PRIOR=prior_head: ${PRIOR_HEAD_PATH}" >&2
    exit 1
fi

if [[ ! -d "${RETRIEVAL_DS_PATH}" ]]; then
    echo "Missing retrieval dataset path: ${RETRIEVAL_DS_PATH}" >&2
    exit 1
fi

generate_exp_name() {
    local topk="$1"
    local checkpoint_name
    checkpoint_name="$(basename "${CHECKPOINT_PATH}")"
    local name="InfoseekNew-LLaVA-Llama3-BAPE-BEFT-${checkpoint_name}-K=${topk}"

    if [[ "${HIDDEN_STATE_OFFSET}" != "0" ]]; then
        name="${name}-h${HIDDEN_STATE_OFFSET}"
    fi

    if [[ "${INCLUDE_Z0_IN_ENSEMBLE}" == "true" ]]; then
        name="${name}-withZ0"
    fi

    if [[ "${ENSURE_GT_PASSAGE_IN_ENSEMBLE}" == "true" ]]; then
        name="${name}-hasGTdoc"
    fi

    name="${name}-prior=${PASSAGE_PRIOR}"
    name="${name}-${RETRIEVE_FIELD}"
    name="${name}-TakeN=${TAKE_N}"
    name="${name}-Offset=${DS_OFFSET}"
    echo "${name}"
}

echo "Running chunk ${CHUNK_INDEX} with offset=${DS_OFFSET}, take_n=${TAKE_N}"

for retrieval_topk in "${RETRIEVAL_TOPK_LIST[@]}"; do
    full_exp_name="$(generate_exp_name "${retrieval_topk}")"
    dynamic_k_top_p="$(awk -v k="${retrieval_topk}" 'BEGIN { printf "%.12g", 1 / (2 * k) }')"

    echo "--------------------------------"
    echo "Running inference for ${full_exp_name}"
    echo "Base model path: ${BASE_MODEL_PATH}"
    echo "Processor path: ${PROCESSOR_PATH}"
    echo "Adapter path: ${ADAPTER_PATH}"
    echo "Prior head path: ${PRIOR_HEAD_PATH}"
    echo "Retrieval topk: ${retrieval_topk}"
    echo "Retrieval field: ${RETRIEVE_FIELD}"
    echo "Chunk offset: ${DS_OFFSET}"
    echo "Chunk size: ${TAKE_N}"
    echo "Include Z0 in ensemble: ${INCLUDE_Z0_IN_ENSEMBLE}"
    echo "Ensure GT passage in ensemble: ${ENSURE_GT_PASSAGE_IN_ENSEMBLE}"
    echo "Passage prior: ${PASSAGE_PRIOR}"
    echo "Dynamic K top-p threshold: ${dynamic_k_top_p}"
    echo "Attention implementation: ${ATTN_IMPLEMENTATION}"

    args=(
        --retrieval_ds_path "${RETRIEVAL_DS_PATH}"
        --dataset_name "InfoseekNew_FullPassage"
        --split "valid"
        --take_n "${TAKE_N}"
        --offset "${DS_OFFSET}"
        --img_basedir "."
        --retrieval_field "${RETRIEVE_FIELD}"
        --retrieval_topk "${retrieval_topk}"
        --model_path "${BASE_MODEL_PATH}"
        --processor_path "${PROCESSOR_PATH}"
        --adapter_name_or_path "${ADAPTER_PATH}"
        --prompt_template ""
        --seed 0
        --batch_size 1
        --exp_name "${OUTPUT_ROOT}/${full_exp_name}"
        --passage_prior "${PASSAGE_PRIOR}"
        --max_batch_size_per_forward "${MAX_BATCH_SIZE_PER_FORWARD}"
        --max_words_per_evidence "${MAX_WORDS_PER_EVIDENCE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --dynamic_k_top_p "${dynamic_k_top_p}"
        --attn_implementation "${ATTN_IMPLEMENTATION}"
    )

    if [[ "${PASSAGE_PRIOR}" == "prior_head" ]]; then
        args+=(--prior_head_path "${PRIOR_HEAD_PATH}")
        args+=(--prior_head_modeling "${PRIOR_HEAD_MODELING}")
        args+=(--prior_head_num_layers "${PRIOR_HEAD_NUM_LAYERS}")
        args+=(--hidden_state_offset "${HIDDEN_STATE_OFFSET}")
    fi

    if [[ "${DO_EVAL}" == "true" ]]; then
        args+=(--do_eval)
    fi

    if [[ "${USE_CACHE}" == "true" ]]; then
        args+=(--use_cache)
    fi

    if [[ "${PREFILL_ANS_TOKEN}" == "true" ]]; then
        args+=(--prefill_ans_token)
    fi

    if [[ "${INCLUDE_Z0_IN_ENSEMBLE}" == "true" ]]; then
        args+=(--include_z0_in_ensemble)
    fi

    if [[ "${ENSURE_GT_PASSAGE_IN_ENSEMBLE}" == "true" ]]; then
        args+=(--ensure_gt_passage_in_ensemble)
    fi

    echo "Args: ${args[*]}"
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/bape_vqa_inference.py "${args[@]}"

    echo "Finished inference for ${full_exp_name}"
    echo "--------------------------------"
done
