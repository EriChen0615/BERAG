#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --array=0-8
#SBATCH --output=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_infoseek_largek_pp_%A_%a.out
#SBATCH --error=/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/logs/slurm_infoseek_largek_pp_%A_%a.err

set -euo pipefail

REPO_DIR="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
cd "${REPO_DIR}"
# NOTE: per project rule, do not activate env inside sbatch scripts.

K_VALUES=(1 5 10 20 30 50 100 150 200)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RETRIEVAL_TOPK="${K_VALUES[$TASK_ID]}"

TAKE_N=256
PASSAGE_PRIOR="prior_head"
RETRIEVE_FIELD="retrieved_passage"
DATASET_NAME="InfoseekNew_FullPassage"
RETRIEVAL_DS_PATH="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B_post_reranked"
IMG_BASEDIR="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
OUTPUT_BASE="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/outputs/0426/LargeK/InfoSeek"

MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
PROCESSOR_PATH="Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=64000-max=4096/checkpoint-7950"
PRIOR_HEAD_PATH="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/beft/beft[K=2*]-prior=mlp-lr1e-6-l0h4-r64-size=64000-max=4096/checkpoint-7950/prior_head.pt"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
if [[ "${NUM_GPUS}" == "(null)" || -z "${NUM_GPUS}" ]]; then
  NUM_GPUS=4
fi

EXP_NAME="InfoSeek-BAPE-LargeK-PP-7B-l0h4-prior_head-K=${RETRIEVAL_TOPK}-TakeN=${TAKE_N}-gpus=${NUM_GPUS}"
EXP_DIR="${OUTPUT_BASE}/${EXP_NAME}"
mkdir -p "${EXP_DIR}"

echo "[InfoSeek-LargeK-PP] SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "[InfoSeek-LargeK-PP] RETRIEVAL_TOPK=${RETRIEVAL_TOPK}"
echo "[InfoSeek-LargeK-PP] NUM_GPUS=${NUM_GPUS}"
echo "[InfoSeek-LargeK-PP] EXP_DIR=${EXP_DIR}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt torchrun \
  --standalone \
  --nproc_per_node "${NUM_GPUS}" \
  "/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/src/bape_vqa_inference.py" \
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
  --do_eval \
  --inference_engine_version v1 \
  --passage_parallel \
  --pp_backend nccl

