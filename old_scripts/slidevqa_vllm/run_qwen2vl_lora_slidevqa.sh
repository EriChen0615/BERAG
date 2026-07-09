#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

DATE=$(date +%m%d)

# Global configuration variables
TAKE_N=0
OFFSET=0
BATCH_SIZE=64
SPLIT="test"  # or "train", "validation"

# Simplified experiment configurations
declare -A base_experiments=(
    # ["SlideVQA-VLLM-Base"]="exp_base"
    # ["SlideVQA-VLLM-Oracle-Base"]="exp_oracle_base"
    # ["SlideVQA-VLLM-SFT"]="exp_sft"
    ["SlideVQA-VLLM-Oracle-SFT"]="exp_oracle_sft"
    # ["SlideVQA-VLLM-LoRA"]="exp_lora"
)

declare -A exp_base=(
    [adapter_path]=""
    [base_model_path]=""
    [do_eval]="true"
    [use_oracle_slides]="false"
    [use_bem]="true"
)

declare -A exp_oracle_base=(
    [adapter_path]=""
    [base_model_path]=""
    [do_eval]="true"
    [use_oracle_slides]="true"
    [use_bem]="true"
)

declare -A exp_sft=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/sft/rag4-slidevqa-sft-r64-bs8-size=0-max=8192/checkpoint-2500"
    [base_model_path]=""
    [do_eval]="true"
    [use_oracle_slides]="false"
    [use_bem]="true"
)

declare -A exp_oracle_sft=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/sft/rag4-slidevqa-sft-r64-bs8-size=0-max=8192/checkpoint-2500"
    [base_model_path]=""
    [do_eval]="true"
    [use_oracle_slides]="true"
    [use_bem]="true"
)

declare -A exp_lora=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/slidevqa/rag4_answer-sft_r64_b8-lr1e-5-max=4096"
    [base_model_path]=""
    [do_eval]="true"
)

# Function to generate experiment name with current settings
generate_exp_name() {
    local base_name="$1"
    local exp_ref="$2"
    local name="${base_name}"
    
    # Check if oracle mode is enabled
    declare -n exp_cfg_local="$exp_ref"
    if [[ "${exp_cfg_local[use_oracle_slides]}" == "true" ]]; then
        name="${name}-Oracle"
    fi
    
    name="${name}-TakeN=${TAKE_N}"
    name="${name}-Split=${SPLIT}"
    
    echo "$name"
}

# Main execution loop
for base_exp_name in "${!base_experiments[@]}"; do
    exp_ref="${base_experiments[$base_exp_name]}"
    # Indirect reference to associative array
    declare -n exp_cfg="$exp_ref"
    
    # Generate experiment name with current settings
    full_exp_name=$(generate_exp_name "$base_exp_name" "$exp_ref")
    
    adapter_path="${exp_cfg[adapter_path]}"
    
    # Use custom base_model_path if specified, otherwise use default
    if [[ -n "${exp_cfg[base_model_path]}" ]]; then
        base_model_path="${exp_cfg[base_model_path]}"
    else
        base_model_path="Qwen/Qwen2-VL-7B-Instruct"
    fi

    echo "--------------------------------"
    echo "Running inference for $full_exp_name"
    echo "Base model path: $base_model_path"
    echo "Adapter path: $adapter_path"
    echo "Split: $SPLIT"
    echo "Take N: $TAKE_N"
    echo "Offset: $OFFSET"
    echo "Batch size: $BATCH_SIZE"

    # Build arguments array
    args=(
        --hf_dataset_path "NTT-hil-insight/SlideVQA"
        --split "$SPLIT"
        --take_n "$TAKE_N"
        --offset "$OFFSET"
        --img_basedir "../../shared_space/vqa_data/KBVQA_data/SlideVQA"
        --base_model_path "$base_model_path"
        --processor_path "Qwen/Qwen2-VL-7B-Instruct"
        --seed 0
        --batch_size "$BATCH_SIZE"
        --exp_name "outputs/1225/VLLM/SlideVQA-Qwen2-VL-7B-Instruct-${full_exp_name}"
        --use_cache
    )

    # Add adapter if specified
    if [[ -n "$adapter_path" ]]; then
        args+=(--adapter_name_or_path "$adapter_path")
    fi

    # Conditionally add store_true flags
    if [[ "${exp_cfg[do_eval]}" == "true" ]]; then
        args+=(--do_eval)
    fi

    if [[ "${exp_cfg[use_oracle_slides]}" == "true" ]]; then
        args+=(--use_oracle_slides)
    fi

    if [[ "${exp_cfg[use_bem]}" == "true" ]]; then
        args+=(--use_bem)
    fi

    echo "Args: ${args[@]}"

    # Run the command
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/vllm_slidevqa_inference.py "${args[@]}"

    echo "Finished inference for $full_exp_name"
    echo "--------------------------------"
done

