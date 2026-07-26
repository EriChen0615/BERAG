#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-src/train/LlamaFactory-0.9.5-beft/.venv/bin/python}"
HF_DATASET_PATH="${HF_DATASET_PATH:-outputs/jinghong_chen/Infoseek-train64000-with-retrieval}"
LLAMAFACTORY_DATA_DIR="${LLAMAFACTORY_DATA_DIR:-src/train/LlamaFactory-0.9.5-beft/data}"
IMG_BASEDIR="${IMG_BASEDIR:-${LLAMAFACTORY_DATA_DIR}/Infoseek/infoseek_train_images}"

TOPK_DOCS="${TOPK_DOCS:-5}"
INFOSEEK_RETRIEVED_CHUNK_LOOKUP_DEPTH="${INFOSEEK_RETRIEVED_CHUNK_LOOKUP_DEPTH:-50}"
DROP_MAX_TOKENS="${DROP_MAX_TOKENS:-4096}"
SAMPLE_SIZE="${SAMPLE_SIZE:-128}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-${LLAMAFACTORY_DATA_DIR}/jinghong_chen/infoseek/rag${TOPK_DOCS}-answer-sft-size=64000-max=${DROP_MAX_TOKENS}}"

"${PYTHON_BIN}" src/curate/ragk_answer_ppl.py \
    --hf_dataset_path "${HF_DATASET_PATH}" \
    --passage_set_name "InfoseekNew_FullPassage" \
    --mode "sft" \
    --output_style "inline_sft" \
    --topk_docs "${TOPK_DOCS}" \
    --infoseek_retrieved_chunk_lookup_depth "${INFOSEEK_RETRIEVED_CHUNK_LOOKUP_DEPTH}" \
    --sample_size "${SAMPLE_SIZE}" \
    --sample_offset "${SAMPLE_OFFSET}" \
    --img_basedir "${IMG_BASEDIR}" \
    --img_path_mode "basename" \
    --output_dir "${OUTPUT_DIR}" \
    --drop_max_tokens "${DROP_MAX_TOKENS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --map_infoseek_chunks_to_full_documents \
    --ensure_gt_passage_in_topk
