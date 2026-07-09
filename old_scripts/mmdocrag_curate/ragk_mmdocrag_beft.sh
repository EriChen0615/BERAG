#!/bin/bash
set -euo pipefail

SAMPLE_SIZE=0
SAMPLE_OFFSET=0
TOPK_DOCS=8
SEED=42
DO_MULTIMODAL_TRAINING=true

DATASET_DIR="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/MMDocRAG/dataset"
OUTPUT_DIR="third_party/LLaMA-Factory-2502/data/jinghong_chen/mmdocrag/rag${TOPK_DOCS}-mmdocrag-beft-size=${SAMPLE_SIZE}-offset=${SAMPLE_OFFSET}"
if [ "${DO_MULTIMODAL_TRAINING}" = "true" ]; then
  OUTPUT_DIR="${OUTPUT_DIR}-multimodal"
fi

python src/curate/ragk_mmdocrag.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --topk_docs "${TOPK_DOCS}" \
  --seed "${SEED}" \
  --sample_size "${SAMPLE_SIZE}" \
  --sample_offset "${SAMPLE_OFFSET}" \
  $( [ "${DO_MULTIMODAL_TRAINING}" = "true" ] && echo "--do_multimodal_training" ) \
  --sanity_check_count 10
