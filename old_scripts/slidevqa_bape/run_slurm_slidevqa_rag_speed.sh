#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --output=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_slidevqa_rag_speed.out
#SBATCH --error=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_slidevqa_rag_speed.err

set -euo pipefail

ROOT="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
SLIDEVQA_IMG_BASEDIR="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/SlideVQA"
TAKE_N=512

BASE_MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
PROCESSOR_PATH="Qwen/Qwen2-VL-7B-Instruct"
# Default to SlideVQA RAG-SFT checkpoint; allow override via env.
ADAPTER_PATH="${ADAPTER_PATH:-${ROOT}/third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/sft/rag4-slidevqa-sft-r64-bs8-size=0-max=8192/checkpoint-2500}"
MAX_BATCH_SIZE_PER_FORWARD=1

EXP_DIR="${ROOT}/outputs/0426/SlideVQA-inference-analysis/RAG/all-slides-TakeN=${TAKE_N}"
mkdir -p "${EXP_DIR}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python "${ROOT}/src/rag_slidevqa_inference.py" \
  --hf_dataset_path "NTT-hil-insight/SlideVQA" \
  --split "test" \
  --take_n "${TAKE_N}" \
  --img_basedir "${SLIDEVQA_IMG_BASEDIR}" \
  --model_path "${BASE_MODEL_PATH}" \
  --processor_path "${PROCESSOR_PATH}" \
  --adapter_name_or_path "${ADAPTER_PATH}" \
  --retrieval_topk 20 \
  --prompt_template "" \
  --seed 0 \
  --max_batch_size_per_forward "${MAX_BATCH_SIZE_PER_FORWARD}" \
  --exp_name "${EXP_DIR}" \
  --use_cache \
  --do_eval \
  --use_bem
