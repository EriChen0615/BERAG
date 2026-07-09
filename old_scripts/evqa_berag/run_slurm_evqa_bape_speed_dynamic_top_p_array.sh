#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --array=0-4
#SBATCH --output=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_evqa_bape_speed_dynamic_%A_%a.out
#SBATCH --error=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_evqa_bape_speed_dynamic_%A_%a.err

set -euo pipefail

ROOT="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
# K_VALUES=(5 10 20 30 50)
K_VALUES=(25 30 50)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RETRIEVAL_TOPK="${K_VALUES[$TASK_ID]}"
# TAKE_N=512
TAKE_N=8
MAX_BATCH_SIZE_PER_FORWARD="${MAX_BATCH_SIZE_PER_FORWARD:-25}"
DYNAMIC_K_TOP_P="$(awk -v k="${RETRIEVAL_TOPK}" 'BEGIN { printf "%.6f", 1/(2*k) }')"

RETRIEVAL_DS_PATH="${ROOT}/outputs/0jingbiao_mei/EVQA-testfull-with-retrieval-rerank7B-step4000_post_reranked"
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
PROCESSOR_PATH="Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH="${ROOT}/third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=0-max=2048/checkpoint-20833"
PRIOR_HEAD_PATH="${ADAPTER_PATH}/prior_head.pt"

EXP_DIR="${ROOT}/outputs/0426/EVQA-inference-analysis/BAPE-dynamicTopP/K=${RETRIEVAL_TOPK}-TopP=${DYNAMIC_K_TOP_P}-TakeN=${TAKE_N}-MBPF=${MAX_BATCH_SIZE_PER_FORWARD}"
mkdir -p "${EXP_DIR}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python "${ROOT}/src/bape_vqa_inference.py" \
  --retrieval_ds_path "${RETRIEVAL_DS_PATH}" \
  --dataset_name "EVQA" \
  --take_n "${TAKE_N}" \
  --img_basedir "${ROOT}" \
  --retrieval_field "retrieved_passage" \
  --retrieval_topk "${RETRIEVAL_TOPK}" \
  --model_path "${MODEL_PATH}" \
  --processor_path "${PROCESSOR_PATH}" \
  --adapter_name_or_path "${ADAPTER_PATH}" \
  --prompt_template "" \
  --seed 0 \
  --batch_size 1 \
  --exp_name "${EXP_DIR}" \
  --prior_head_path "${PRIOR_HEAD_PATH}" \
  --passage_prior "prior_head" \
  --max_batch_size_per_forward "${MAX_BATCH_SIZE_PER_FORWARD}" \
  --hidden_state_offset 4 \
  --dynamic_k_top_p "${DYNAMIC_K_TOP_P}" \
  --use_cache \
  --do_eval
