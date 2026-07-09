#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --array=0-2
#SBATCH --output=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_infoseek_bape_speed_k_%A_%a.out
#SBATCH --error=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_infoseek_bape_speed_k_%A_%a.err

set -euo pipefail

ROOT="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
K_VALUES=(5 10 15)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RETRIEVAL_TOPK="${K_VALUES[$TASK_ID]}"
TAKE_N=512

RETRIEVAL_DS_PATH="${ROOT}/outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B_post_reranked"
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
PROCESSOR_PATH="Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH="${ROOT}/third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=64000-max=4096/checkpoint-7950"
PRIOR_HEAD_PATH="${ADAPTER_PATH}/prior_head.pt"

EXP_DIR="${ROOT}/outputs/0426/Infoseek-inference-analysis/BAPE/K=${RETRIEVAL_TOPK}-TakeN=${TAKE_N}"
mkdir -p "${EXP_DIR}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python "${ROOT}/src/bape_vqa_inference.py" \
  --retrieval_ds_path "${RETRIEVAL_DS_PATH}" \
  --dataset_name "InfoseekNew_FullPassage" \
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
  --max_batch_size_per_forward 5 \
  --hidden_state_offset 4 \
  --use_cache \
  --do_eval
