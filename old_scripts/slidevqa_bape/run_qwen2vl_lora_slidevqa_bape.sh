#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

DATE=$(date +%m%d)

# Global configuration variables
TAKE_N=0
# For SlideVQA, we always use all 20 slides (no TopK concept)
# RETRIEVAL_TOPK is set to 20 to use all available slides
RETRIEVAL_TOPK=20
PASSAGE_PRIOR="prior_head"
# Posterior traces are saved by src/bape_slidevqa_inference.py during fresh inference.
# Do not reuse older cached CSVs, since they may not contain the posterior columns.
FORCE_RERUN_FOR_POSTERIORS=true

# Simplified experiment configurations for SlideVQA
declare -A base_experiments=(
    ["SlideVQA-BAPE-BEFT[K=4*]-prior=mlp-lr1e-6-l1h4-r64-epoch1"]="exp_slidevqa_beft_k4_prior_mlp"
    # ["SlideVQA-BAPE-BEFT[K=4*]-prior=mlp-lr1e-6-l1h4-r64-epoch1-subdivide=4"]="exp_slidevqa_beft_k4_prior_mlp_subdivide4"
    # ["SlideVQA-BAPE-BEFT[K=4*]-prior=mlp-lr1e-6-l1h4-r64-epoch1-oracle"]="exp_slidevqa_beft_k4_prior_mlp_oracle"
    # ["SlideVQA-BAPE-BEFT[K=4*]-prior=mlp-lr1e-6-l1h4-r64-epoch1-da=subdivide4"]="exp_slidevqa_beft_k4_prior_mlp_da_subdivide4"
    # ["SlideVQA-BAPE-BEFT[K=4*]-prior=mlp-lr1e-6-l1h4-r64-epoch1-da=subdivide4-oracle"]="exp_slidevqa_beft_k4_prior_mlp_da_subdivide4_oracle"
)


declare -A exp_slidevqa_beft_k4_prior_mlp_da_subdivide4_oracle=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-da=subdivide4-max=2048/checkpoint-5000"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-da=subdivide4-max=2048/checkpoint-5000/prior_head.pt"
    [hidden_state_offset]=4
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [use_oracle_slides]="true"
    [use_bem]="true"
    [subdivide_image_into_parts]=4
)

declare -A exp_slidevqa_beft_k4_prior_mlp_da_subdivide4=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-da=subdivide4-max=2048/checkpoint-5000"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-da=subdivide4-max=2048/checkpoint-5000/prior_head.pt"
    [hidden_state_offset]=4
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [use_oracle_slides]="false"
    [use_bem]="true"
)

declare -A exp_slidevqa_beft_k4_prior_mlp=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-max=2048/checkpoint-2500"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-max=2048/checkpoint-2500/prior_head.pt"
    [hidden_state_offset]=4
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [use_oracle_slides]="false"
    [use_bem]="false"
)

declare -A exp_slidevqa_beft_k4_prior_mlp_oracle=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-max=2048/checkpoint-2500"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-max=2048/checkpoint-2500/prior_head.pt"
    [hidden_state_offset]=4
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [use_oracle_slides]="true"
    [use_bem]="true"
)

declare -A exp_slidevqa_beft_k4_prior_mlp_subdivide4=(
    [model_path]="Qwen/Qwen2-VL-7B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-max=2048/checkpoint-2500"
    [prior_head_path]="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-max=2048/checkpoint-2500/prior_head.pt"
    [hidden_state_offset]=4
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [use_oracle_slides]="true"
    [use_bem]="true"
    [subdivide_image_into_parts]=4
)

# Function to generate experiment name with current settings
generate_exp_name() {
    local base_name="$1"
    local name="${base_name}"
    
    if [[ "${exp_cfg[hidden_state_offset]}" != "0" ]]; then
        name="${name}-h${exp_cfg[hidden_state_offset]}"
    fi
    
    if [[ "${exp_cfg[use_oracle_slides]}" == "true" ]]; then
        name="${name}-oracle"
    fi
    
    if [[ -n "${exp_cfg[subdivide_image_into_parts]}" ]]; then
        name="${name}-subdivide=${exp_cfg[subdivide_image_into_parts]}"
    fi
    
    name="${name}-prior=${PASSAGE_PRIOR}"
    name="${name}-K=${RETRIEVAL_TOPK}"
    name="${name}-TakeN=${TAKE_N}"
    
    echo "$name"
}

# Main execution loop
for base_exp_name in "${!base_experiments[@]}"; do
    exp_ref="${base_experiments[$base_exp_name]}"
    # Indirect reference to associative array
    declare -n exp_cfg="$exp_ref"
    
    # Generate experiment name with current settings
    full_exp_name=$(generate_exp_name "$base_exp_name")
    
    adapter_path="${exp_cfg[adapter_path]}"
    hf_dataset_path="${exp_cfg[hf_dataset_path]:-NTT-hil-insight/SlideVQA}"
    split="${exp_cfg[split]:-test}"

    echo "--------------------------------"
    echo "Running inference for $full_exp_name"
    echo "Adapter path: $adapter_path"
    echo "Prior head path: ${exp_cfg[prior_head_path]}"
    echo "Retrieval topk (slides): $RETRIEVAL_TOPK"
    echo "Prompt template: ${exp_cfg[prompt_template]}"
    echo "Use oracle slides: ${exp_cfg[use_oracle_slides]}"
    echo "Use BEM evaluation: ${exp_cfg[use_bem]}"
    echo "Subdivide image into parts: ${exp_cfg[subdivide_image_into_parts]:-none}"
    echo "Passage prior: $PASSAGE_PRIOR"
    echo "Split: $split"
    
    # Build arguments array
    args=(
        --hf_dataset_path "$hf_dataset_path"
        --split "$split"
        --take_n "$TAKE_N"
        --img_basedir "../../shared_space/vqa_data/KBVQA_data/SlideVQA"
        --retrieval_topk "$RETRIEVAL_TOPK"
        --model_path "${exp_cfg[model_path]}"
        --processor_path "${exp_cfg[processor_path]}"
        --adapter_name_or_path "$adapter_path"
        --prompt_template "${exp_cfg[prompt_template]}"
        --seed 42
        --batch_size 1
        --exp_name "outputs/0526/SlideVQA/BAPE/${full_exp_name}"
        --prior_head_path "${exp_cfg[prior_head_path]}"
        --passage_prior "$PASSAGE_PRIOR"
        --prior_head_modeling "mlp_head"
        --prior_head_num_layers 2
        --prior_head_proj_dim 1024
        --hidden_state_offset "${exp_cfg[hidden_state_offset]}"
        --max_new_tokens 128
        --max_batch_size_per_forward 5
        --inference_engine_version "v1"
    )

    # Conditionally add store_true flags
    if [[ "${exp_cfg[do_eval]}" == "true" ]]; then
        args+=(--do_eval)
    fi

    if [[ "${exp_cfg[use_cache]}" == "true" ]]; then
        args+=(--use_cache)
    fi

    if [[ "${exp_cfg[use_oracle_slides]}" == "true" ]]; then
        args+=(--use_oracle_slides)
    fi

    if [[ "${exp_cfg[prefill_ans_token]}" == "true" ]]; then
        args+=(--prefill_ans_token)
    fi

    if [[ "${exp_cfg[use_bem]}" == "true" ]]; then
        args+=(--use_bem)
    fi

    if [[ -n "${exp_cfg[subdivide_image_into_parts]}" ]]; then
        args+=(--subdivide_image_into_parts "${exp_cfg[subdivide_image_into_parts]}")
    fi

    echo "Args: ${args[@]}"

    # Run the command
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/bape_slidevqa_inference.py "${args[@]}"
    result_csv="outputs/0526/SlideVQA/BAPE/${full_exp_name}/inference_results.csv"
    python - "$result_csv" <<'PY'
import csv
import sys

result_csv = sys.argv[1]
required_columns = {"log_posterior_over_steps", "log_document_posterior"}

with open(result_csv, newline="") as f:
    header = set(next(csv.reader(f)))

missing_columns = sorted(required_columns - header)
if missing_columns:
    raise SystemExit(f"Missing posterior columns in {result_csv}: {missing_columns}")

print(f"Verified posterior columns in {result_csv}")
PY

    echo "Finished inference for $full_exp_name"
    echo "--------------------------------"
done



declare -A exp_slidevqa_beft_k4_prior_mlp_fusedgt=(
    [model_name]="Qwen/Qwen2-VL-7B-Instruct"
    [adapter_path]="third_party/LLaMA-Factory-2502/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-fusedgt-lr1e-6-l1h4-r64-size=0-max=2048/checkpoint-2500"
    [prior_head_path]="third_party/LLaMA-Factory-2502/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-fusedgt-lr1e-6-l1h4-r64-size=0-max=2048/checkpoint-2500/prior_head.pt"
    [topk_docs]=4
)
