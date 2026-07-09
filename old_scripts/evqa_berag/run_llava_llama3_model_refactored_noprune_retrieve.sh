#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=32:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

set -euo pipefail

# Usage:
#   CHECKPOINT_PATH=/path/to/checkpoint bash scripts/evqa_bape/run_llava_llama3_model_refactored_noprune_retrieve.sh
#
# The checkpoint should be a LLaMA-Factory LoRA checkpoint containing adapter_model.*
# and, for PASSAGE_PRIOR=prior_head, prior_head.pt.

BASE_MODEL_PATH="${BASE_MODEL_PATH:-xtuner/llava-llama-3-8b-v1_1-transformers}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-third_party/LLaMA-Factory-2502/saves/llava_llama3_8b_v1_1_transformers/lora/evqa_full/beft/rag2-k2-prior=mlp-lr1e-6-h4-r64-size0-max2048/checkpoint-20833}"
ADAPTER_PATH="${ADAPTER_PATH:-${CHECKPOINT_PATH}}"
PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${CHECKPOINT_PATH}/prior_head.pt}"

if [[ -f "${CHECKPOINT_PATH}/processor_config.json" || -f "${CHECKPOINT_PATH}/preprocessor_config.json" ]]; then
    PROCESSOR_PATH="${PROCESSOR_PATH:-${CHECKPOINT_PATH}}"
else
    PROCESSOR_PATH="${PROCESSOR_PATH:-${BASE_MODEL_PATH}}"
fi

INCLUDE_Z0_IN_ENSEMBLE="${INCLUDE_Z0_IN_ENSEMBLE:-false}"
ENSURE_GT_PASSAGE_IN_ENSEMBLE="${ENSURE_GT_PASSAGE_IN_ENSEMBLE:-false}"
TAKE_N="${TAKE_N:-256}"
DS_OFFSET="${DS_OFFSET:-0}"
RETRIEVAL_TOPK_LIST=(1 3 5 10 15 20 30 50)
PASSAGE_PRIOR="${PASSAGE_PRIOR:-prior_head}"
RETRIEVE_FIELD="${RETRIEVE_FIELD:-retrieved_passage}"
RETRIEVAL_DS_PATH="${RETRIEVAL_DS_PATH:-outputs/0jingbiao_mei/EVQA-testfull-with-retrieval-rerank7B-step4000_post_reranked}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/0526/EVQA/LLaVA-Llama3/BAPE}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_BATCH_SIZE_PER_FORWARD="${MAX_BATCH_SIZE_PER_FORWARD:-5}"
MAX_WORDS_PER_EVIDENCE="${MAX_WORDS_PER_EVIDENCE:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HIDDEN_STATE_OFFSET="${HIDDEN_STATE_OFFSET:-4}"
PRIOR_HEAD_MODELING="${PRIOR_HEAD_MODELING:-mlp_head}"
PRIOR_HEAD_NUM_LAYERS="${PRIOR_HEAD_NUM_LAYERS:-2}"
USE_CACHE="${USE_CACHE:-false}"
DO_EVAL="${DO_EVAL:-true}"
PREFILL_ANS_TOKEN="${PREFILL_ANS_TOKEN:-false}"

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
    echo "Missing checkpoint directory: ${CHECKPOINT_PATH}" >&2
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

declare -A exp_llava_llama3_evqa_beft=(
    [model_path]="${BASE_MODEL_PATH}"
    [processor_path]="xtuner/llava-llama-3-8b-v1_1-transformers"
    [adapter_path]="third_party/LLaMA-Factory-2502/saves/llava_llama3_8b_v1_1_transformers/lora/evqa_full/beft/rag2-k2-prior=mlp-lr1e-6-h4-r64-size0-max2048/checkpoint-20833"
    [prior_head_path]="third_party/LLaMA-Factory-2502/saves/llava_llama3_8b_v1_1_transformers/lora/evqa_full/beft/rag2-k2-prior=mlp-lr1e-6-h4-r64-size0-max2048/checkpoint-20833/prior_head.pt"
    [hidden_state_offset]="4"
    [prompt_template]=""
    [retrieval_field]="retrieved_passage"
    [do_eval]="false"
    [use_cache]="false"
    [prefill_ans_token]="false"
    [prior_head_modeling]="mlp_head"
    [prior_head_num_layers]="2"
)

declare -A base_experiments=(
    ["EVQA-LLaVA-Llama3-BAPE-BEFT"]="exp_llava_llama3_evqa_beft"
)

generate_exp_name() {
    local base_name="$1"
    local topk="$2"
    local checkpoint_name
    checkpoint_name="$(basename "${CHECKPOINT_PATH}")"
    local name="${base_name}-${checkpoint_name}-K=${topk}"

    if [[ "${exp_cfg[hidden_state_offset]}" != "0" ]]; then
        name="${name}-h${exp_cfg[hidden_state_offset]}"
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
    echo "${name}"
}

for base_exp_name in "${!base_experiments[@]}"; do
    exp_ref="${base_experiments[$base_exp_name]}"
    declare -n exp_cfg="${exp_ref}"

    for retrieval_topk in "${RETRIEVAL_TOPK_LIST[@]}"; do
        full_exp_name="$(generate_exp_name "${base_exp_name}" "${retrieval_topk}")"
        retrieval_field="${exp_cfg[retrieval_field]}"
        dynamic_k_top_p="$(awk -v k="${retrieval_topk}" 'BEGIN { printf "%.12g", 1 - 1 / (2 * k) }')"

        echo "--------------------------------"
        echo "Running inference for ${full_exp_name}"
        echo "Base model path: ${exp_cfg[model_path]}"
        echo "Processor path: ${exp_cfg[processor_path]}"
        echo "Adapter path: ${exp_cfg[adapter_path]}"
        echo "Prior head path: ${exp_cfg[prior_head_path]}"
        echo "Retrieval topk: ${retrieval_topk}"
        echo "Retrieval field: ${retrieval_field}"
        echo "Include Z0 in ensemble: ${INCLUDE_Z0_IN_ENSEMBLE}"
        echo "Ensure GT passage in ensemble: ${ENSURE_GT_PASSAGE_IN_ENSEMBLE}"
        echo "Passage prior: ${PASSAGE_PRIOR}"
        echo "Dynamic K top-p threshold: ${dynamic_k_top_p}"
        echo "Attention implementation: ${ATTN_IMPLEMENTATION}"

        args=(
            --retrieval_ds_path "${RETRIEVAL_DS_PATH}"
            --dataset_name "EVQA"
            --take_n "${TAKE_N}"
            --img_basedir "."
            --retrieval_field "${retrieval_field}"
            --retrieval_topk "${retrieval_topk}"
            --model_path "${exp_cfg[model_path]}"
            --processor_path "${exp_cfg[processor_path]}"
            --adapter_name_or_path "${exp_cfg[adapter_path]}"
            --prompt_template "${exp_cfg[prompt_template]}"
            --seed 0
            --batch_size 1
            --exp_name "${OUTPUT_ROOT}/${full_exp_name}"
            --passage_prior "${PASSAGE_PRIOR}"
            --max_batch_size_per_forward "${MAX_BATCH_SIZE_PER_FORWARD}"
            --max_words_per_evidence "${MAX_WORDS_PER_EVIDENCE}"
            --max_new_tokens "${MAX_NEW_TOKENS}"
            --offset "${DS_OFFSET}"
            --dynamic_k_top_p "${dynamic_k_top_p}"
            --attn_implementation "${ATTN_IMPLEMENTATION}"
        )

        if [[ "${PASSAGE_PRIOR}" == "prior_head" ]]; then
            args+=(--prior_head_path "${exp_cfg[prior_head_path]}")
            args+=(--prior_head_modeling "${exp_cfg[prior_head_modeling]}")
            args+=(--prior_head_num_layers "${exp_cfg[prior_head_num_layers]}")
            args+=(--hidden_state_offset "${exp_cfg[hidden_state_offset]}")
        fi

        if [[ "${exp_cfg[do_eval]}" == "true" ]]; then
            args+=(--do_eval)
        fi

        if [[ "${exp_cfg[use_cache]}" == "true" ]]; then
            args+=(--use_cache)
        fi

        if [[ "${exp_cfg[prefill_ans_token]}" == "true" ]]; then
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
done
