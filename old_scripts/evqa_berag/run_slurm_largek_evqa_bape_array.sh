#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --array=0-8
#SBATCH --output=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_evqa_largek_%A_%a.out
#SBATCH --error=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_evqa_largek_%A_%a.err

set -euo pipefail

REPO_DIR="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
cd "${REPO_DIR}"
# source "/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/scripts/hpc_activate_env_py310_infer.sh"

# K sweep requested by user.
K_VALUES=(1 5 10 20 30 50 100 150 200)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RETRIEVAL_TOPK="${K_VALUES[$TASK_ID]}"

TAKE_N=256
PASSAGE_PRIOR="prior_head"
RETRIEVE_FIELD="retrieved_passage"

DATASET_NAME="EVQA"
RETRIEVAL_DS_PATH="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/outputs/0jingbiao_mei/EVQA-testfull-with-retrieval-rerank7B-step4000_post_reranked"
IMG_BASEDIR="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
OUTPUT_BASE="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/outputs/0426/LargeK/EVQA"

# 7B BEFT checkpoint with prior head (l0h4) from existing EVQA scripts.
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
PROCESSOR_PATH="Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=0-max=2048/checkpoint-20833"
PRIOR_HEAD_PATH="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=0-max=2048/checkpoint-20833/prior_head.pt"

EXP_NAME="EVQA-BAPE-LargeK-7B-l0h4-prior_head-K=${RETRIEVAL_TOPK}-TakeN=${TAKE_N}"
EXP_DIR="${OUTPUT_BASE}/${EXP_NAME}"
mkdir -p "${EXP_DIR}"

echo "[EVQA-LargeK] SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "[EVQA-LargeK] RETRIEVAL_TOPK=${RETRIEVAL_TOPK}"
echo "[EVQA-LargeK] EXP_DIR=${EXP_DIR}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python "/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/src/bape_vqa_inference.py" \
  --retrieval_ds_path "${RETRIEVAL_DS_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --take_n "${TAKE_N}" \
  --img_basedir "${IMG_BASEDIR}" \
  --retrieval_field "${RETRIEVE_FIELD}" \
  --retrieval_topk "${RETRIEVAL_TOPK}" \
  --model_path "${MODEL_PATH}" \
  --processor_path "${PROCESSOR_PATH}" \
  --adapter_name_or_path "${ADAPTER_PATH}" \
  --prompt_template "" \
  --seed 0 \
  --batch_size 1 \
  --exp_name "${EXP_DIR}" \
  --prior_head_path "${PRIOR_HEAD_PATH}" \
  --passage_prior "${PASSAGE_PRIOR}" \
  --max_batch_size_per_forward 5 \
  --hidden_state_offset 4 \
  --use_cache \
  --do_eval

