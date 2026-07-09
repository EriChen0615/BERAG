#!/usr/bin/env bash
set -euo pipefail

TAKE_N="${TAKE_N:-256}"
OFFSET="${OFFSET:-0}"
N_GRID="${N_GRID:-2}"
HAYSTACK_M="${HAYSTACK_M:-40}"
NEEDLES_PER_QUERY="${NEEDLES_PER_QUERY:-1}"
ADD_Z0="${ADD_Z0:-false}"  # no-z0 models
PASSAGE_PRIOR="${PASSAGE_PRIOR:-prior_head}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MMNEEDLE_DATA_ROOT="${MMNEEDLE_DATA_ROOT:-/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/MMNeedle/test_val2014}"
MMNEEDLE_CAPTION_PATH="${MMNEEDLE_CAPTION_PATH:-/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/MMNeedle/data/file_to_caption.json}"

case "${N_GRID}" in
  1)
    CHECKPOINT_DIR_DEFAULT="third_party/LLaMA-Factory-2502/saves/llava_llama3_8b_v1_1_transformers/lora/mmneedle/beft/rag10-n1x1-k2-size10000-offset0/checkpoint-1242"
    BASE_NAME="MMNeedle-BAPE-BEFT-LLaVA-Llama3-8B-v1_1-no_z0-n1x1"
    ;;
  2)
    CHECKPOINT_DIR_DEFAULT="third_party/LLaMA-Factory-2502/saves/llava_llama3_8b_v1_1_transformers/lora/mmneedle/beft/llava_llama3_8b_v1_1_transformers-rag10-n2x2-k2-no_z0-size10000-offset0/checkpoint-1242"
    BASE_NAME="MMNeedle-BAPE-BEFT-LLaVA-Llama3-8B-v1_1-no_z0-n2x2"
    ;;
  4)
    CHECKPOINT_DIR_DEFAULT="third_party/LLaMA-Factory-2502/saves/llava_llama3_8b_v1_1_transformers/lora/mmneedle/beft/llava_llama3_8b_v1_1_transformers-rag10-n4x4-k2-no_z0-size10000-offset0/checkpoint-1242"
    BASE_NAME="MMNeedle-BAPE-BEFT-LLaVA-Llama3-8B-v1_1-no_z0-n4x4"
    ;;
  8)
    CHECKPOINT_DIR_DEFAULT="third_party/LLaMA-Factory-2502/saves/llava_llama3_8b_v1_1_transformers/lora/mmneedle/beft/llava_llama3_8b_v1_1_transformers-rag10-n8x8-k2-no_z0-size10000-offset0/checkpoint-1242"
    BASE_NAME="MMNeedle-BAPE-BEFT-LLaVA-Llama3-8B-v1_1-no_z0-n8x8"
    ;;
  *)
    echo "Unsupported N_GRID=${N_GRID}. Expected one of: 1, 2, 4, 8"
    exit 2
    ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_DIR_DEFAULT}}"
PRIOR_HEAD_PATH="${PRIOR_HEAD_PATH:-${CHECKPOINT_DIR}/prior_head.pt}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}"
  exit 1
fi
if [[ ! -f "${PRIOR_HEAD_PATH}" ]]; then
  echo "Missing prior head file: ${PRIOR_HEAD_PATH}"
  exit 1
fi

full_exp_name="${BASE_NAME}-N=${N_GRID}x${N_GRID}-M=${HAYSTACK_M}-needle=${NEEDLES_PER_QUERY}-prior=${PASSAGE_PRIOR}-TakeN=${TAKE_N}"
output_dir="outputs/0426/BAPE/MMNeedle/${full_exp_name}"

echo "--------------------------------"
echo "Running MMNeedle inference for ${full_exp_name}"
echo "Output dir: ${output_dir}"
echo "Model path: xtuner/llava-llama-3-8b-v1_1-transformers"
echo "Adapter path: ${CHECKPOINT_DIR}"
echo "Prior head path: ${PRIOR_HEAD_PATH}"
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
  --model_path "xtuner/llava-llama-3-8b-v1_1-transformers"
  --processor_path "xtuner/llava-llama-3-8b-v1_1-transformers"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --passage_prior "${PASSAGE_PRIOR}"
  --prior_head_modeling "mlp_head"
  --prior_head_num_layers "2"
  --prior_head_proj_dim "1024"
  --hidden_state_offset "4"
  --max_batch_size_per_forward 5
  --num_beams 1
  --exp_name "${output_dir}"
  --adapter_name_or_path "${CHECKPOINT_DIR}"
  --prior_head_path "${PRIOR_HEAD_PATH}"
  --do_eval
)

if [[ "${ADD_Z0}" == "true" ]]; then
  args+=(--add_z0)
fi

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/bape_mmneedle_inference.py "${args[@]}"
echo "Finished ${full_exp_name}"

