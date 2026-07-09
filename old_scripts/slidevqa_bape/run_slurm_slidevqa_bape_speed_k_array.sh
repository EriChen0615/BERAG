#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --output=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_slidevqa_bape_speed.out
#SBATCH --error=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_slidevqa_bape_speed.err

set -euo pipefail

ROOT="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
SLIDEVQA_IMG_BASEDIR="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/SlideVQA"
RETRIEVAL_TOPK=20
TAKE_N=512

MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
PROCESSOR_PATH="Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH="${ROOT}/third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/slidevqa/beft/beft[K=4*]-prior=mlp-lr1e-6-l1h4-r64-size=0-da=subdivide4-max=2048/checkpoint-5000"
PRIOR_HEAD_PATH="${ADAPTER_PATH}/prior_head.pt"

EXP_DIR="${ROOT}/outputs/0426/SlideVQA-inference-analysis/BAPE/all-slides-K=${RETRIEVAL_TOPK}-TakeN=${TAKE_N}"
mkdir -p "${EXP_DIR}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python "${ROOT}/src/bape_slidevqa_inference.py" \
  --hf_dataset_path "NTT-hil-insight/SlideVQA" \
  --split "test" \
  --take_n "${TAKE_N}" \
  --img_basedir "${SLIDEVQA_IMG_BASEDIR}" \
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
  --max_batch_size_per_forward 5 \
  --hidden_state_offset 4 \
  --use_cache \
  --do_eval \
  --use_bem
