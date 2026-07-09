#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=32:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

DATE=$(date +%m%d)

# Global configuration variables
# INCLUDE_Z0_IN_ENSEMBLE="false"
ENSURE_GT_PASSAGE_IN_ENSEMBLE="false"
# TAKE_N=0
# TAKE_N=16
TAKE_N=0
# RETRIEVAL_TOPK_LIST=(1 2 3 5)
# RETRIEVAL_TOPK_LIST=(1 2 3 5 7 10 12 15 20 25)
# RETRIEVAL_TOPK_LIST=(30 40 50)
# RETRIEVAL_TOPK_LIST=(15 20)
# RETRIEVAL_TOPK_LIST=(10 15 20 30)
# RETRIEVAL_TOPK_LIST=(50)
# RETRIEVAL_TOPK_LIST=(1 2 3 5 10)
# RETRIEVAL_TOPK_LIST=(20 30 50)
RETRIEVAL_TOPK_LIST=(2)
# RETRIEVAL_TOPK_LIST=(20 30)
# RETRIEVAL_TOPK_LIST=(10)
# RETRIEVAL_TOPK_LIST=(3 5 7 10)
# RETRIEVAL_TOPK_LIST=(1 3 5 7 10 15 20)
# RETRIEVAL_TOPK_LIST=(12 15 20)
# RETRIEVAL_TOPK_LIST=(15 20 25 30)
# RETRIEVAL_TOPK_LIST=(1 2 3 5)
# RETRIEVAL_TOPK_LIST=(5)
# RETRIEVAL_TOPK_LIST=(7 10 12 15 20)
# PASSAGE_PRIOR="uniform"
PASSAGE_PRIOR="prior_head"
RETRIEVE_FIELD="retrieved_passage"
RETRIEVAL_DS_PATH="outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B_post_reranked"
# RETRIEVE_FIELD="reranked_passage"

# Simplified experiment configurations (without retrieval_topk, include_z0_in_ensemble, ensure_gt_passage_in_ensemble)
declare -A base_experiments=(
    # ["InfoseekNew-BAPE-BEFT[K=2]"]="exp_beft_k2"
    # ["InfoseekNew-BAPE-BEFT[K=2]-data=64000"]="exp_beft_k2_data_64000"
    ["InfoseekNew-BAPE-BEFT[K=2]-l0h4-data=64000"]="exp_beft_k2_l0h4_data_64000"
    ["InfoseekNew-BAPE-BEFT[K=2]-z0-data=64000"]="exp_beft_k2_z0_data_64000"
    # ["InfoseekNew-BAPE-BEFT[K=2]-cont-z0-data=64000"]="exp_beft_k2_cont_z0_data_64000"
    # ["EVQA-BAPE-GT_SFT"]="exp_gt_sft"
    # ["EVQA-BAPE-BEPO[K*=2]-prior=mlp_lr1e-6-l1h4-beta=0.7"]="exp_rag2_beft_k2_prior_mlp_lr1e_6_l1h4_bepo_beta07"
)

declare -A exp_beft_k2_l0h4_data_64000=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=64000-max=4096/checkpoint-7950"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=64000-max=4096/checkpoint-7950/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
    [hidden_state_offset]=4
    [include_z0_in_ensemble]="false"
)

declare -A exp_beft_k2_z0_data_64000=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft-z0[K=2*]-prior=mlp-lr1e-6-l1h4-r64-size=64000-max=4096/checkpoint-7947"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft-z0[K=2*]-prior=mlp-lr1e-6-l1h4-r64-size=64000-max=4096/checkpoint-7947/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
    [hidden_state_offset]=4
    [include_z0_in_ensemble]="true"
)

declare -A exp_beft_k2_cont_z0_data_64000=(
    [model_path]="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-BEFT[K=2]-data=64000-LoRA"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/cont-beft-z0[K=2*]-prior=mlp-lr1e-6-l1h4-r64-deflection-size=64000-max=4096/checkpoint-7947"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/cont-beft-z0[K=2*]-prior=mlp-lr1e-6-l1h4-r64-deflection-size=64000-max=4096/checkpoint-7947/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
    [hidden_state_offset]=4
    [include_z0_in_ensemble]="true"
)

declare -A exp_beft_k2=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft[K=2*]-prior=mlp-lr1e-6-l1h4-r64-size=64000-max=4096/checkpoint-2500"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft[K=2*]-prior=mlp-lr1e-6-l1h4-r64-size=64000-max=4096/checkpoint-2500/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
    [hidden_state_offset]=4
)

declare -A exp_beft_k2_data_64000=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft[K=2*]-prior=mlp-lr1e-6-l1h4-r64-size=64000-max=4096/checkpoint-7950"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft[K=2*]-prior=mlp-lr1e-6-l1h4-r64-size=64000-max=4096/checkpoint-7950/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
    [hidden_state_offset]=4
)

# declare -A exp_gt_sft=(
#     [model_path]="Qwen/Qwen2-VL-2B-Instruct"
#     [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
#     [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag-answer-sft-size=64000-max=2048"
#     [prompt_template]=""
#     [retrieval_field]="$RETRIEVE_FIELD"
#     [do_eval]="true"
#     [use_cache]="true"
# )


# Function to generate experiment name with current settings
generate_exp_name() {
    local base_name="$1"
    local topk="$2"
    local name="${base_name}-K=${topk}"
    if [[ "${exp_cfg[hidden_state_offset]}" != "0" ]]; then
        name="${name}-h${exp_cfg[hidden_state_offset]}"
    fi
    
    if [[ "${exp_cfg[include_z0_in_ensemble]}" == "true" ]]; then
        name="${name}-withZ0"
    fi
    
    if [[ "$ENSURE_GT_PASSAGE_IN_ENSEMBLE" == "true" ]]; then
        name="${name}-hasGTdoc"
    fi

    name="${name}-prior=${PASSAGE_PRIOR}"
    name="${name}-${RETRIEVE_FIELD}"
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
        retrieval_field="${exp_cfg[retrieval_field]}"

        echo "--------------------------------"
        echo "Running inference for $full_exp_name"
        echo "Adapter path: $adapter_path"
        echo "Retrieval topk: $retrieval_topk"
        echo "Retrieval field: $retrieval_field"
        echo "Prompt template: ${exp_cfg[prompt_template]}"
        echo "Include Z0 in ensemble: ${exp_cfg[include_z0_in_ensemble]:-false}"
        echo "Ensure GT passage in ensemble: $ENSURE_GT_PASSAGE_IN_ENSEMBLE"
        echo "Passage prior: $PASSAGE_PRIOR"
        # Build arguments array
        args=(
            --retrieval_ds_path "$RETRIEVAL_DS_PATH"
            --dataset_name "InfoseekNew_FullPassage"
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
            --exp_name "outputs/0326/Infoseek/BAPE/${full_exp_name}"
            --prior_head_path "${exp_cfg[prior_head_path]}"
            --passage_prior "$PASSAGE_PRIOR"
            --max_batch_size_per_forward 5
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

        if [[ "${exp_cfg[hidden_state_offset]}" != "0" ]]; then
            args+=(--hidden_state_offset "${exp_cfg[hidden_state_offset]}")
        fi

        if [[ -n "${exp_cfg[prior_head_modeling]}" ]]; then
            args+=(--prior_head_modeling "${exp_cfg[prior_head_modeling]}")
        fi

        if [[ -n "${exp_cfg[prior_head_num_layers]}" ]]; then
            args+=(--prior_head_num_layers "${exp_cfg[prior_head_num_layers]}")
        fi

        # Use experiment config for ensemble settings
        if [[ "${exp_cfg[include_z0_in_ensemble]}" == "true" ]]; then
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
