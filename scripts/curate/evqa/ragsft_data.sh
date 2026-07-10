#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

HF_DATASET_PATH="${HF_DATASET_PATH:-outputs/jinghong_chen/EVQA-with-retrieval}"
LLAMAFACTORY_DATA_DIR="${LLAMAFACTORY_DATA_DIR:-src/train/LlamaFactory-0.9.5-beft/data}"
IMG_BASEDIR="${IMG_BASEDIR:-data/EVQA}"

TOPK_DOCS="${TOPK_DOCS:-5}"
DROP_MAX_TOKENS="${DROP_MAX_TOKENS:-4096}"
SAMPLE_SIZE="${SAMPLE_SIZE:-64000}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-${LLAMAFACTORY_DATA_DIR}/jinghong_chen/evqa/rag${TOPK_DOCS}-answer-sft-size=${SAMPLE_SIZE}-max=${DROP_MAX_TOKENS}}"

python src/curate/ragk_answer_ppl.py \
    --hf_dataset_path "${HF_DATASET_PATH}" \
    --passage_set_name "EVQA" \
    --mode "sft" \
    --topk_docs "${TOPK_DOCS}" \
    --sample_size "${SAMPLE_SIZE}" \
    --sample_offset "${SAMPLE_OFFSET}" \
    --img_basedir "${IMG_BASEDIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --drop_max_tokens "${DROP_MAX_TOKENS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --ensure_gt_passage_in_topk
