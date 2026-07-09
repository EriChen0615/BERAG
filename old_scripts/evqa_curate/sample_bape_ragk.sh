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
# TAKE_N=64000
TAKE_N=64000
# TAKE_N=20
# TAKE_N=128
# TAKE_N=16
# RETRIEVAL_TOPK_LIST=(1 2 3 5)
# RETRIEVAL_TOPK_LIST=(1 2 3 5 7 10 12 15 20 25)
# RETRIEVAL_TOPK_LIST=(30 40 50)
# RETRIEVAL_TOPK_LIST=(1 2 3 5 7 10 12 15 20 25 30 40 50)
# RETRIEVAL_TOPK_LIST=(7 10 12 15 20 25 30)
RETRIEVAL_TOPK_LIST=(2)
# RETRIEVAL_TOPK_LIST=(1 2 3 5)
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
    # ["EVQA-BAPE-RAG2_PPL[Ensemble]-wPrior-FullEPL1"]="exp_rag2_ensemble_with_prior_fullep1"
    # ["EVQA-7B-BAPE-RAG2_PPL[Ensemble]-wPrior-7B-FullEPL1"]="exp_7B_rag2_ensemble_with_prior_fullep1"
    # ["EVQA-BAPE-RAG2_PPL[Ensemble]-wPrior5-FullEPL1"]="exp_rag2_ensemble_with_prior5_fullep1"
    # ["EVQA-BAPE-RAG2_PPL[Ensemble]-wPrior10-FullEPL1"]="exp_rag2_ensemble_with_prior10_fullep1"
    # ["EVQA-BAPE-RAG2_PPL[Ensemble]-wPrior_dyn_token-FullEPL1"]="exp_rag2_ensemble_with_prior_dyn_token_fullep1"
    # ["EVQA-BAPE-RAG2_BEFT2[K*=2]"]="exp_rag2_beft2_k2"
    # ["EVQA-BAPE-RAG2_BEFT2[K*=4]"]="exp_rag2_beft2_k4"
    # ["EVQA-BAPE-BEFT[K*=2]-l1h4"]="exp_rag2_beft_k2_l1h4"
    # ["EVQA-BAPE-BEFT[K*=2]-l0h4"]="exp_rag2_beft_k2_l0h4"
    # ["EVQA-BAPE-BEFT[K*=2]-cont-[K*=4]-l1h4-n4beams"]="exp_rag2_beft_k2_cont_k4_l1h4_n4beams"
    ["EVQA-BAPE-BEFT[K*=2]-1epoch-prior=mlp_lr1e-6-n4beams"]="exp_rag2_beft_k2_prior_mlp_lr1e_6_n4beams_size0"
)

declare -A exp_rag2_beft_k2_prior_mlp_lr1e_6_n4beams_size0=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-BEFT[K=2*]-prior=mlp[lr1e-6]-l1h4-r64_bs8-epoch1-max=2048"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [prior_head_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-BEFT[K=2*]-prior=mlp[lr1e-6]-l1h4-r64_bs8-epoch1-max=2048/prior_head.pt"
    [hidden_state_offset]=4
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
    [inference_engine_version]="v2"
    [return_n_sequences]=4
    [num_beams]=4
)

declare -A exp_rag2_beft_k2_cont_k4_l1h4_n4beams=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-BEFT[K=2*]-l0h4-size=64000-max=2048"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/beft/beft[K=2*]-cont-[K=4*]-prior=l1h4-size=64000-max=2048"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/beft/beft[K=2*]-cont-[K=4*]-prior=l1h4-size=64000-max=2048/prior_head.pt"
    [hidden_state_offset]=4
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="false"
    [inference_engine_version]="v2"
    [return_n_sequences]=4
    [num_beams]=4
    [offset]=128000
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


declare -A exp_rag2_ensemble_with_prior5_fullep1=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior5-size=0-max=2048"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior5-size=0-max=2048/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag2_ensemble_with_prior10_fullep1=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior10-size=0-max=2048"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior10-size=0-max=2048/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag2_ensemble_with_prior_dyn_token_fullep1=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior-dyn-token-size=0-max=2048"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[ensemble]-wprior-dyn-token-size=0-max=2048/prior_head.pt"
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

declare -A exp_rag2_beft2_k2=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG2_LoRA-BEFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/beft-stage2/beft2[K=2*]-wprior-size=64000-max=2048/checkpoint-1992/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag2_beft2_k4=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG2_LoRA-BEFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/beft-stage2/beft2[K=4*]-wprior-size=64000-max=2048/checkpoint-1983/prior_head.pt"
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag2_beft_k2_l0h4=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/beft/beft[K=2*]-prior=l0h4-size=64000-max=2048"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/beft/beft[K=2*]-prior=l0h4-size=64000-max=2048/prior_head.pt"
    [hidden_state_offset]=4
    [prompt_template]=""
    [retrieval_field]="$RETRIEVE_FIELD"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp_rag2_beft_k2_l1h4=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/beft/beft[K=2*]-prior=l1h4-size=64000-max=2048"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/beft/beft[K=2*]-prior=l1h4-size=64000-max=2048/prior_head.pt"
    [hidden_state_offset]=4
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
    if [[ "${exp_cfg[hidden_state_offset]}" != "0" ]]; then
        name="${name}-h${exp_cfg[hidden_state_offset]}"
    fi
    
    if [[ "$INCLUDE_Z0_IN_ENSEMBLE" == "true" ]]; then
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
        echo "Include Z0 in ensemble: $INCLUDE_Z0_IN_ENSEMBLE"
        echo "Ensure GT passage in ensemble: $ENSURE_GT_PASSAGE_IN_ENSEMBLE"
        echo "Passage prior: $PASSAGE_PRIOR"
        # Build arguments array
        args=(
            --retrieval_ds_path "outputs/jinghong_chen/EVQA-with-retrieval"
            --dataset_name "EVQA"
            --take_n "$TAKE_N"
            --img_basedir "."
            --retrieval_field "$retrieval_field"
            --retrieval_topk "$retrieval_topk"
            --model_path "${exp_cfg[model_path]}"
            --processor_path "${exp_cfg[processor_path]}"
            --adapter_name_or_path "$adapter_path"
            --prompt_template "${exp_cfg[prompt_template]}"
            --seed 42
            --batch_size 1
            --exp_name "outputs/1125/BAPE_sample/${full_exp_name}"
            --prior_head_path "${exp_cfg[prior_head_path]}"
            --passage_prior "$PASSAGE_PRIOR"
            --return_n_sequences "${exp_cfg[return_n_sequences]}"
            --num_beams "${exp_cfg[num_beams]}"
            --inference_engine_version "${exp_cfg[inference_engine_version]}"
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

        # Use global variables for ensemble settings
        if [[ "$INCLUDE_Z0_IN_ENSEMBLE" == "true" ]]; then
            args+=(--include_z0_in_ensemble)
        fi

        if [[ "$ENSURE_GT_PASSAGE_IN_ENSEMBLE" == "true" ]]; then
            args+=(--ensure_gt_passage_in_ensemble)
        fi

        if [[ "${exp_cfg[offset]}" ]]; then
            args+=(--offset "${exp_cfg[offset]}")
        fi

        echo "Args: ${args[@]}"

        # Run the command
        CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/bape_vqa_inference.py "${args[@]}"

        echo "Finished inference for $full_exp_name"
        echo "--------------------------------"
    done
done
