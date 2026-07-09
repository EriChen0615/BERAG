#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH -J evqa_pos_vllm_7b
set -euo pipefail

# Activate your env before sbatch (e.g. source scripts/hpc_activate_env_py310_infer.sh).

ROOT="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
cd "${ROOT}"

OUT_ROOT="${ROOT}/outputs/0426/EVQA-positional-analysis"
CURATE_ROOT="${ROOT}/analysis/EVQA-gtdoc-position-datasets"
# Default N matches curator output prefix EVQA-${N}-gtdoc_at_* ; override if you used --n_sample.
N="${N:-256}"

VARIANT_TAGS=(
  "gtdoc_at_1-4"
  "gtdoc_at_5-8"
  "gtdoc_at_9-12"
  "gtdoc_at_13-16"
  "gtdoc_at_17-20"
)

RETRIEVAL_TOPK=20
RETRIEVE_FIELD="retrieved_passage"
PROC="Qwen/Qwen2-VL-7B-Instruct"

for tag in "${VARIANT_TAGS[@]}"; do
  DS_PATH="${CURATE_ROOT}/EVQA-${N}-${tag}"
  if [[ ! -d "${DS_PATH}" ]]; then
    echo "ERROR: missing curated dataset: ${DS_PATH}" >&2
    exit 1
  fi

  # Base 7B (no LoRA)
  EXP_BASE="${OUT_ROOT}/${tag}__7B-base__${RETRIEVE_FIELD}-Top${RETRIEVAL_TOPK}"
  echo "=== ${EXP_BASE}"
  CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/vllm_vqa_inference.py \
    --retrieval_ds_path "${DS_PATH}" \
    --dataset_name "EVQA" \
    --take_n 0 \
    --img_basedir "." \
    --retrieval_field "${RETRIEVE_FIELD}" \
    --retrieval_topk "${RETRIEVAL_TOPK}" \
    --base_model_path "${PROC}" \
    --processor_path "${PROC}" \
    --seed 0 \
    --batch_size 256 \
    --exp_name "${EXP_BASE}" \
    --do_eval \
    --use_cache

  # SFT merged weights (no adapter)
  EXP_SFT="${OUT_ROOT}/${tag}__7B-SFT__${RETRIEVE_FIELD}-Top${RETRIEVAL_TOPK}"
  echo "=== ${EXP_SFT}"
  CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/vllm_vqa_inference.py \
    --retrieval_ds_path "${DS_PATH}" \
    --dataset_name "EVQA" \
    --take_n 0 \
    --img_basedir "." \
    --retrieval_field "${RETRIEVE_FIELD}" \
    --retrieval_topk "${RETRIEVAL_TOPK}" \
    --base_model_path "${ROOT}/data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT" \
    --processor_path "${PROC}" \
    --seed 0 \
    --batch_size 256 \
    --exp_name "${EXP_SFT}" \
    --do_eval \
    --use_cache

  # DPO LoRA on SFT base
  DPO_ADAPTER="${ROOT}/third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.5"
  EXP_DPO="${OUT_ROOT}/${tag}__7B-DPO__${RETRIEVE_FIELD}-Top${RETRIEVAL_TOPK}"
  echo "=== ${EXP_DPO}"
  CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/vllm_vqa_inference.py \
    --retrieval_ds_path "${DS_PATH}" \
    --dataset_name "EVQA" \
    --take_n 0 \
    --img_basedir "." \
    --retrieval_field "${RETRIEVE_FIELD}" \
    --retrieval_topk "${RETRIEVAL_TOPK}" \
    --base_model_path "${ROOT}/data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT" \
    --processor_path "${PROC}" \
    --adapter_name_or_path "${DPO_ADAPTER}" \
    --seed 0 \
    --batch_size 256 \
    --exp_name "${EXP_DPO}" \
    --do_eval \
    --use_cache
done
