#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH -J mmneedle_bape_qwen2b_z0
set -euo pipefail

TAKE_N="${TAKE_N:-8}"
OFFSET="${OFFSET:-0}"
N_GRID="${N_GRID:-1}"
HAYSTACK_M="${HAYSTACK_M:-10}"
NEEDLES_PER_QUERY="${NEEDLES_PER_QUERY:-1}"
ADD_Z0="${ADD_Z0:-true}"  # z0 checkpoint by default
PASSAGE_PRIOR="${PASSAGE_PRIOR:-prior_head}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MMNEEDLE_DATA_ROOT="${MMNEEDLE_DATA_ROOT:-/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/MMNeedle/data}"
MMNEEDLE_CAPTION_PATH="${MMNEEDLE_CAPTION_PATH:-/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/MMNeedle/data/file_to_caption.json}"

# Trained with-z0 checkpoint (provided by user).
CHECKPOINT_DIR="${CHECKPOINT_DIR:-third_party/LLaMA-Factory-2502/saves/qwen2_vl-2b/lora/mmneedle/beft/qwen2_vl-2b-rag10-n1x1-k2-z0-size10000-offset0/checkpoint-1242}"

declare -A base_experiments=(
  ["MMNeedle-BAPE-BEFT-Qwen2VL2B-z0"]="exp_mmneedle_qwen2vl2b_z0_beft"
)

declare -A exp_mmneedle_qwen2vl2b_z0_beft=(
  [model_path]="Qwen/Qwen2-VL-2B-Instruct"
  [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
  [adapter_path]="${CHECKPOINT_DIR}"
  [prior_head_path]="${CHECKPOINT_DIR}/prior_head.pt"
  [hidden_state_offset]="4"
  [prior_head_modeling]="mlp_head"
  [prior_head_num_layers]="2"
  [prior_head_proj_dim]="1024"
  [prompt_template]=""
  [do_eval]="true"
  [use_cache]="false"
)

generate_exp_name() {
  local base_name="$1"
  local name="${base_name}"
  name="${name}-N=${N_GRID}x${N_GRID}"
  name="${name}-M=${HAYSTACK_M}"
  name="${name}-needle=${NEEDLES_PER_QUERY}"
  if [[ "${ADD_Z0}" == "true" ]]; then
    name="${name}-z0"
  fi
  name="${name}-prior=${PASSAGE_PRIOR}"
  name="${name}-TakeN=${TAKE_N}"
  echo "${name}"
}

for base_exp_name in "${!base_experiments[@]}"; do
  exp_ref="${base_experiments[$base_exp_name]}"
  declare -n exp_cfg="${exp_ref}"

  full_exp_name=$(generate_exp_name "${base_exp_name}")
  output_dir="outputs/0426/BAPE/MMNeedle/${full_exp_name}"

  echo "--------------------------------"
  echo "Running MMNeedle inference for ${full_exp_name}"
  echo "Output dir: ${output_dir}"
  echo "Model path: ${exp_cfg[model_path]}"
  echo "Adapter path: ${exp_cfg[adapter_path]}"
  echo "Prior head path: ${exp_cfg[prior_head_path]}"
  echo "--------------------------------"

  args=(
    --hf_dataset_path "${MMNEEDLE_DATA_ROOT}"
    --split "test"
    --force_local_mmneedle
    --mmneedle_data_root "${MMNEEDLE_DATA_ROOT}"
    --mmneedle_caption_path "${MMNEEDLE_CAPTION_PATH}"
    --n_grid "${N_GRID}"
    --haystack_m "${HAYSTACK_M}"
    --needles_per_query "${NEEDLES_PER_QUERY}"
    --take_n "${TAKE_N}"
    --offset "${OFFSET}"
    --seed 42
    --model_path "${exp_cfg[model_path]}"
    --processor_path "${exp_cfg[processor_path]}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --passage_prior "${PASSAGE_PRIOR}"
    --prior_head_modeling "${exp_cfg[prior_head_modeling]}"
    --prior_head_num_layers "${exp_cfg[prior_head_num_layers]}"
    --prior_head_proj_dim "${exp_cfg[prior_head_proj_dim]}"
    --hidden_state_offset "${exp_cfg[hidden_state_offset]}"
    --max_batch_size_per_forward 5
    --num_beams 1
    --exp_name "${output_dir}"
  )

  if [[ -n "${exp_cfg[adapter_path]}" ]]; then
    args+=(--adapter_name_or_path "${exp_cfg[adapter_path]}")
  fi
  if [[ -n "${exp_cfg[prior_head_path]}" ]]; then
    args+=(--prior_head_path "${exp_cfg[prior_head_path]}")
  fi
  if [[ -n "${exp_cfg[prompt_template]}" ]]; then
    args+=(--prompt_template "${exp_cfg[prompt_template]}")
  fi
  if [[ "${exp_cfg[do_eval]}" == "true" ]]; then
    args+=(--do_eval)
  fi
  if [[ "${ADD_Z0}" == "true" ]]; then
    args+=(--add_z0)
  fi
  if [[ "${exp_cfg[use_cache]}" == "true" ]]; then
    args+=(--use_cache)
  fi

  CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/bape_mmneedle_inference.py "${args[@]}"
  echo "Finished ${full_exp_name}"
done

