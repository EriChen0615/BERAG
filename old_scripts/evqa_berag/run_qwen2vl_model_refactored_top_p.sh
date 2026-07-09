#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

DATE=$(date +%m%d)

# Global configuration variables
INCLUDE_Z0_IN_ENSEMBLE="false"
ENSURE_GT_PASSAGE_IN_ENSEMBLE="false"
TAKE_N=256
DYNAMIC_K_TOP_P=0.9
# TAKE_N=16
# RETRIEVAL_TOPK_LIST=(1 2 3 5)
RETRIEVAL_TOPK_LIST=(1 2 3 5 7 10 12 15 20 25)
# RETRIEVAL_TOPK_LIST=(7 10 12 15 20)
# PASSAGE_PRIOR="uniform"
PASSAGE_PRIOR="prior_head"
RETRIEVE_FIELD="retrieved_passage"
# RETRIEVE_FIELD="retrieved_passage"

# Simplified experiment configurations (without retrieval_topk, include_z0_in_ensemble, ensure_gt_passage_in_ensemble)
declare -A base_experiments=(
    # ["EVQA-BAPE-Base"]="exp_base"
    # ["EVQA-BAPE-RAG2_PPL[Joint]"]="exp_rag2_joint"
    # ["EVQA-BAPE-RAG4_PPL[Joint]"]="exp_rag4_joint"
    # ["EVQA-BAPE-GT_SFT"]="exp_gt_sft"
    # ["EVQA-BAPE-RAG2_PPL[Ensemble]"]="exp_rag2_ensemble"
    # ["EVQA-BAPE-RAG4_PPL[Ensemble]"]="exp_rag4_ensemble"
    # ["EVQA-BAPE-RAG2_PPL[Ensemble]-wPrior"]="exp_rag2_ensemble_with_prior"
    # ["EVQA-BAPE-RAG4_PPL[Ensemble]-wPrior"]="exp_rag4_ensemble_with_prior"
    ["EVQA-BAPE-RAG2_PPL[Ensemble]-wPrior-FullEPL1"]="exp_rag2_ensemble_with_prior_fullep1"
    ["EVQA-7B-BAPE-RAG2_PPL[Ensemble]-wPrior-7B-FullEPL1"]="exp_7B_rag2_ensemble_with_prior_fullep1"
)

declare -A exp_base=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag2_joint=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[joint]-size=64000-max=2048"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag4_joint=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag4-answer-ppl[joint]-size=64000-max=2048"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_gt_sft=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag-answer-sft-size=64000-max=2048"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag2_ensemble=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-size=64000-max=2048/checkpoint-1992"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag4_ensemble=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag4-answer-ppl[ensemble]-size=64000-max=2048/checkpoint-1983"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag2_ensemble_with_prior=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior-size=64000-max=2048/checkpoint-1992"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior-size=64000-max=2048/checkpoint-1992/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag4_ensemble_with_prior=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag4-answer-ppl[ensemble]-wprior-size=64000-max=2048/checkpoint-1983"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag4-answer-ppl[ensemble]-wprior-size=64000-max=2048/checkpoint-1983/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag2_ensemble_with_prior_fullep1=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior-size=0-max=2048/checkpoint-5204"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior-size=0-max=2048/checkpoint-5204/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_7B_rag2_ensemble_with_prior_fullep1=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior-size=0-max=2048/checkpoint-4500"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior-size=0-max=2048/checkpoint-4500/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

# Function to generate experiment name with current settings
generate_exp_name() {
    local base_name="$1"
    local topk="$2"
    local name="${base_name}-K=${topk}"
    
    if [[ "$INCLUDE_Z0_IN_ENSEMBLE" == "true" ]]; then
        name="${name}-withZ0"
    fi
    
    if [[ "$ENSURE_GT_PASSAGE_IN_ENSEMBLE" == "true" ]]; then
        name="${name}-hasGTdoc"
    fi

    name="${name}-prior=${PASSAGE_PRIOR}"
    name="${name}-${RETRIEVE_FIELD}"
    name="${name}-TakeN=${TAKE_N}"
    name="${name}-DynamicKTopP=${DYNAMIC_K_TOP_P}"
    
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
        retrieval_field="${exp_cfg[retrieval_field]}"

        echo "--------------------------------"
        echo "Running inference for $full_exp_name"
        echo "Adapter path: $adapter_path"
        echo "Retrieval topk: $retrieval_topk"
        echo "Retrieval field: $retrieval_field"
        echo "Prompt template: ${exp_cfg[prompt_template]}"
        echo "Include Z0 in ensemble: $INCLUDE_Z0_IN_ENSEMBLE"
        echo "Ensure GT passage in ensemble: $ENSURE_GT_PASSAGE_IN_ENSEMBLE"
        echo "Passage prior: $PASSAGE_PRIOR"
        # Build arguments array
        args=(
            --retrieval_ds_path "outputs/jinghong_chen/EVQA-testfull-with-retrieval_post_reranked"
            --dataset_name "EVQA"
            --take_n "$TAKE_N"
            --img_basedir "."
            --retrieval_field "$retrieval_field"
            --retrieval_topk "$retrieval_topk"
            --model_path "${exp_cfg[model_path]}"
            --processor_path "${exp_cfg[processor_path]}"
            --adapter_name_or_path "$adapter_path"
            --prompt_template "${exp_cfg[prompt_template]}"
            --seed 0
            --batch_size 1
            --exp_name "outputs/1025/BAPE/Qwen2-VL-2B-Instruct-${full_exp_name}"
            --prior_head_path "${exp_cfg[prior_head_path]}"
            --passage_prior "$PASSAGE_PRIOR"
            --dynamic_k_top_p "$DYNAMIC_K_TOP_P"
        )

        # Conditionally add store_true flags
        if [[ "${exp_cfg[do_eval]}" == "true" ]]; then
            args+=(--do_eval)
        fi

        if [[ "${exp_cfg[use_cache]}" == "true" ]]; then
            args+=(--use_cache)
        fi

        if [[ "${exp_cfg[prefill_ans_token]}" == "true" ]]; then
            args+=(--prefill_ans_token)
        fi

        if [[ "${exp_cfg[include_gt_passage_only]}" == "true" ]]; then
            args+=(--include_gt_passage_only)
        fi

        # Use global variables for ensemble settings
        if [[ "$INCLUDE_Z0_IN_ENSEMBLE" == "true" ]]; then
            args+=(--include_z0_in_ensemble)
        fi

        if [[ "$ENSURE_GT_PASSAGE_IN_ENSEMBLE" == "true" ]]; then
            args+=(--ensure_gt_passage_in_ensemble)
        fi

        echo "Args: ${args[@]}"

        # Run the command
        CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/bape_vqa_inference.py "${args[@]}"

        echo "Finished inference for $full_exp_name"
        echo "--------------------------------"
    done
done
