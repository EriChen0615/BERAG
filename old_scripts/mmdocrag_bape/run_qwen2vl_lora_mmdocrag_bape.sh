#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

set -euo pipefail

# source /home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/scripts/hpc_activate_env.sh

SETTING=${1:-15}
TAKE_N=5

DATASET_DIR="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/MMDocRAG/dataset"
PROMPT_TEMPLATE="third_party/MMDocRAG/prompt_bank/multimodal_infer.txt"

OUTPUT_DIR="outputs/0226/MMDocRAG"

declare -A base_experiments=(
    # ["MMDocRAG-BAPE-Qwen2.5VL-3B-base"]="exp_mmdocrag_base"
    ["MMDocRAG-BAPE-Qwen2.5VL-3B-lora"]="exp_mmdocrag_eval15_lora"
)

declare -A exp_mmdocrag_base=(
    [model_path]="Qwen/Qwen2.5-VL-3B-Instruct"
    [processor_path]="Qwen/Qwen2.5-VL-3B-Instruct"
    [adapter_path]=""
    [prior_head_path]=""
    [hidden_state_offset]=4
    [passage_prior]="uniform"
)

declare -A exp_mmdocrag_eval15_lora=(
    [model_path]="Qwen/Qwen2.5-VL-3B-Instruct"
    [processor_path]="Qwen/Qwen2.5-VL-3B-Instruct"
    [adapter_path]="third_party/LLaMA-Factory-2502/saves/qwen2_5_vl-3b/lora/mmdocrag/beft/rag8-mmdocrag-size=0-multimodal-max_len=2048/checkpoint-428"
    [prior_head_path]="third_party/LLaMA-Factory-2502/saves/qwen2_5_vl-3b/lora/mmdocrag/beft/rag8-mmdocrag-size=0-multimodal-max_len=2048/checkpoint-428/prior_head.pt"
    [hidden_state_offset]=4
    [passage_prior]="prior_head"
)

for base_exp_name in "${!base_experiments[@]}"; do
    exp_ref="${base_experiments[$base_exp_name]}"
    declare -n exp_cfg="$exp_ref"
    
    output_jsonl="${OUTPUT_DIR}/mmdocrag_bape_${base_exp_name// /_}_quotes${SETTING}_response_N${TAKE_N}.jsonl"

    args=(
        --setting "${SETTING}"
        --dataset_dir "${DATASET_DIR}"
        --prompt_template "${PROMPT_TEMPLATE}"
        --output_path "${output_jsonl}"
        --take_n "${TAKE_N}"
        --model_path "${exp_cfg[model_path]}"
        --processor_path "${exp_cfg[processor_path]}"
        --hidden_state_offset "${exp_cfg[hidden_state_offset]}"
        --max_new_tokens 512
        --passage_prior "${exp_cfg[passage_prior]}"
        --max_batch_size_per_forward 5
        --prior_head_modeling "mlp_head"
        --prior_head_num_layers 2
        --prior_head_proj_dim 1024
    )
        # --attn_implementation "sdpa"

    if [[ -n "${exp_cfg[adapter_path]}" ]]; then
        args+=(--adapter_name_or_path "${exp_cfg[adapter_path]}")
    fi

    if [[ -n "${exp_cfg[prior_head_path]}" ]]; then
        args+=(--prior_head_path "${exp_cfg[prior_head_path]}")
    fi

    echo "--------------------------------"
    echo "Running inference for $base_exp_name"
    echo "Output: $output_jsonl"
    echo "Args: ${args[@]}"

    python3 src/bape_mmdocrag_inference.py "${args[@]}"
done
