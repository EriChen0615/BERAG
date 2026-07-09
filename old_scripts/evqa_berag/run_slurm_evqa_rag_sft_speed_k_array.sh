#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --array=0-4
#SBATCH --output=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_evqa_rag_sft_speed_k_%A_%a.out
#SBATCH --error=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_evqa_rag_sft_speed_k_%A_%a.err

set -euo pipefail

ROOT="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
# K_VALUES=(5 10 20 30 50)
K_VALUES=(25)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RETRIEVAL_TOPK="${K_VALUES[$TASK_ID]}"
# TAKE_N=512
TAKE_N=16

RETRIEVAL_DS_PATH="${ROOT}/outputs/0jingbiao_mei/EVQA-testfull-with-retrieval-rerank7B-step4000_post_reranked"
SFT_MODEL_PATH="${ROOT}/data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT"
PROCESSOR_PATH="Qwen/Qwen2-VL-7B-Instruct"

EXP_DIR="${ROOT}/outputs/0426/EVQA-inference-analysis/RAG-SFT/K=${RETRIEVAL_TOPK}-TakeN=${TAKE_N}"
mkdir -p "${EXP_DIR}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python "${ROOT}/src/rag_vqa_inference.py" \
  --retrieval_ds_path "${RETRIEVAL_DS_PATH}" \
  --dataset_name "EVQA" \
  --take_n "${TAKE_N}" \
  --img_basedir "${ROOT}" \
  --retrieval_field "retrieved_passage" \
  --retrieval_topk "${RETRIEVAL_TOPK}" \
  --model_path "${SFT_MODEL_PATH}" \
  --processor_path "${PROCESSOR_PATH}" \
  --prompt_template "" \
  --seed 0 \
  --exp_name "${EXP_DIR}" \
  --use_cache \
  --do_eval
