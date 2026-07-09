#!/usr/bin/env bash
set -euo pipefail

# EVQA K=20
# EVQA Retrieval dataset; field=retrieval_passage
RETRIEVAL_DB="outputs/0jingbiao_mei/EVQA-testfull-with-retrieval_post_reranked"
# Base
BASE_CSV="outputs/0326/EVQA/Qwen2-VL-7B-Instruct-retrieved_passage-Top20/marked_inference_results.csv"
# SFT
SFT_CSV="outputs/1125/VLLM/Qwen2-VL-2B-Instruct-7B-EVQA-VLLM-SFT-RAG-K=5-Top20-Retrieve-TakeN=0/marked_inference_results.csv"
# DPO
DPO_CSV="outputs/1125/VLLM/Qwen2-VL-2B-Instruct-EVQA-VLLM-DPO-RAG-K=5-Top20-Retrieve-TakeN=0/marked_inference_results.csv"
# BEFT
BEFT_CSV="outputs/1125-v3/BAPE/7B-EVQA-BAPE-BEFT[K*=2]-prior=mlp_lr1e-6-l0h4-lora_r64_bs8-epoch1-K=20-h4-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"

# source scripts/hpc_activate_env_py310_infer.sh

python analysis/positional_invariance.py \
  --retrieval_db "$RETRIEVAL_DB" \
  --base_csv "$BASE_CSV" \
  --sft_csv "$SFT_CSV" \
  --dpo_csv "$DPO_CSV" \
  --beft_csv "$BEFT_CSV"

