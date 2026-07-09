#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

DATE=$(date +%m%d)

# Global configuration variables
ENSURE_GT_PASSAGE_IN_ENSEMBLE="false"
# RETRIEVAL_TOPK_LIST=(7 10 12 15 20 25)
# RETRIEVAL_TOPK_LIST=(1 2 3 5)
# RETRIEVAL_TOPK_LIST=(3 5 7 10 12)
# RETRIEVAL_TOPK_LIST=(15 20 30)
RETRIEVAL_TOPK_LIST=(20)
RETRIEVE_FIELD="retrieved_passage"
TAKE_N=0
BATCH_SIZE=256

# Simplified experiment configurations (without retrieval_topk)
declare -A base_experiments=(
    # ["EVQA-VLLM-SFT-GT"]="exp_sft_gt"
    # ["EVQA-VLLM-SFT-RAG-K=5"]="exp_rag_sft_k5"
    # ["EVQA-VLLM-SFT-RAG-K=5_r64_bs8_epoch1"]="exp_rag_sft_k5_r64_bs8_epoch1"
    # ["EVQA-VLLM-DPO-RAG-K=5"]="exp_rag_dpo_k5"
    # ["EVQA-VLLM-EFT-K=2"]="exp_eft_k2"
    # ["EVQA-VLLM-EFT-K=4"]="exp_eft_k4"
    # ["EVQA-VLLM-APFT-K=2"]="exp_apft_k2"
    # ["EVQA-VLLM-APFT-K=4"]="exp_apft_k4"
    # ["EVQA-VLLM-Base"]="exp_base"
    # ["EVQA-VLLM-Reranker2b"]="exp_reranker_2b"
    # ["7B-EVQA-VLLM-SFT-RAG-K=5"]="exp_7B_rag_sft_k5"
    ["7B-EVQA-VLLM-DPO-RAG-K=5"]="exp_7B_rag_dpo_k5"
    # ["7B-EVQA-VLLM-Base"]="exp_7B_base"
)

declare -A exp_7B_base=(
    [adapter_path]="Qwen/Qwen2-VL-7B-Instruct"
    [base_model_path]=""
    [do_eval]="true"
)

declare -A exp_7B_rag_sft_k5=(
    [adapter_path]=""
    [base_model_path]="data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT"
    [do_eval]="true"
)

declare -A exp_7B_rag_dpo_k5=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.5"
    [base_model_path]="data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT"
    [do_eval]="true"
)

declare -A exp_rag_sft_k5_r64_bs8_epoch1=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-sft_r64_b8-lr1e-5-max=4096"
    [base_model_path]=""
    [do_eval]="true"
)


declare -A exp_sft_gt=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag-answer-sft-size=64000-max=2048"
    [base_model_path]=""
    [do_eval]="true"
)

declare -A exp_apft_k2=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[joint]-size=64000-max=2048"
    [base_model_path]=""
    [do_eval]="true"
)

declare -A exp_apft_k4=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag4-answer-ppl[joint]-size=64000-max=2048"
    [base_model_path]=""
    [do_eval]="true"
)

declare -A exp_rag_sft_k5=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [base_model_path]=""
    [do_eval]="true"
)

declare -A exp_eft_k2=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-size=64000-max=2048/checkpoint-1992"
    [base_model_path]=""
    [do_eval]="true"
)

declare -A exp_eft_k4=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag4-answer-ppl[ensemble]-size=64000-max=2048/checkpoint-1983"
    [base_model_path]=""
    [do_eval]="true"
)

declare -A exp_rag_dpo_k5=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.7"
    [base_model_path]="data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT"
    [do_eval]="true"
)

declare -A exp_reranker_2b=(
    [adapter_path]="~/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/checkpoints/qwen2_vl-2b/qlora/evqa/2024-12-28_doc1_verify"
    [base_model_path]="QWen/QWen2-VL-2B-Instruct"
    [do_eval]="true"
)

# Function to generate experiment name with current settings
generate_exp_name() {
    local base_name="$1"
    local topk="$2"
    local name="${base_name}-Top${topk}"
    
    if [[ "$ENSURE_GT_PASSAGE_IN_ENSEMBLE" == "true" ]]; then
        name="${name}-hasGTdoc"
    fi

    if [[ "$RETRIEVE_FIELD" == "reranked_passage" ]]; then
        name="${name}-Rerank"
    else
        name="${name}-Retrieve"
    fi
    
    name="${name}-TakeN=${TAKE_N}"
    
    echo "$name"
}

# Main execution loop
for base_exp_name in "${!base_experiments[@]}"; do
    exp_ref="${base_experiments[$base_exp_name]}"
    # Indirect reference to associative array
    declare -n exp_cfg="$exp_ref"
    
    # Iterate over retrieval_topk values
    for retrieval_topk in "${RETRIEVAL_TOPK_LIST[@]}"; do
        # Generate experiment name with current settings
        full_exp_name=$(generate_exp_name "$base_exp_name" "$retrieval_topk")
        
        adapter_path="${exp_cfg[adapter_path]}"
        
        # Use custom base_model_path if specified, otherwise use default
        if [[ -n "${exp_cfg[base_model_path]}" ]]; then
            base_model_path="${exp_cfg[base_model_path]}"
        else
            base_model_path="Qwen/Qwen2-VL-2B-Instruct"
        fi

        echo "--------------------------------"
        echo "Running inference for $full_exp_name"
        echo "Base model path: $base_model_path"
        echo "Adapter path: $adapter_path"
        echo "Retrieval topk: $retrieval_topk"
        echo "Retrieval field: $RETRIEVE_FIELD"
        echo "Take N: $TAKE_N"
        echo "Batch size: $BATCH_SIZE"

        # Build arguments array
        args=(
            --retrieval_ds_path "outputs/jinghong_chen/EVQA-testfull-with-retrieval_post_reranked"
            --dataset_name "EVQA"
            --take_n "$TAKE_N"
            --img_basedir "."
            --retrieval_field "$RETRIEVE_FIELD"
            --retrieval_topk "$retrieval_topk"
            --base_model_path "$base_model_path"
            --processor_path "Qwen/Qwen2-VL-2B-Instruct"
            --adapter_name_or_path "$adapter_path"
            --seed 0
            --batch_size "$BATCH_SIZE"
            --exp_name "outputs/0326/VLLM/${full_exp_name}"
            --use_cache
        )

        # Conditionally add store_true flags
        if [[ "${exp_cfg[do_eval]}" == "true" ]]; then
            args+=(--do_eval)
        fi

        if [[ "$ENSURE_GT_PASSAGE_IN_ENSEMBLE" == "true" ]]; then
            args+=(--ensure_gt_passage_in_ensemble)
        fi

        echo "Args: ${args[@]}"

        # Run the command
        CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/vllm_vqa_inference.py "${args[@]}"

        echo "Finished inference for $full_exp_name"
        echo "--------------------------------"
    done
done
