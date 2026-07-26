#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/src/train/LlamaFactory-0.9.5-beft/.venv/bin/python}"
TOP_K="${TOP_K:-2}"
INPUT_DATASET="${INPUT_DATASET:-${ROOT_DIR}/outputs/jinghong_chen/Infoseek-train64000-echosight-top${TOP_K}-with-retrieval}"
KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-/workspace/projects/Infoseek-retrieval/data/echosight/infoseek_100k_wiki/wiki_100_dict_v4.json}"
PASSAGE_SET_NAME="${PASSAGE_SET_NAME:-EchoSightKB:${KNOWLEDGE_BASE}}"
IMG_BASEDIR="${IMG_BASEDIR:-${ROOT_DIR}/src/train/LlamaFactory-0.9.5-beft/data/Infoseek/infoseek_train_images}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/src/train/LlamaFactory-0.9.5-beft/data/jinghong_chen/infoseek/echosight-rag${TOP_K}-answer-sft-size=64000-max=32768}"
DROP_MAX_TOKENS="${DROP_MAX_TOKENS:-32768}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-100}"

"${PYTHON_BIN}" src/curate/ragk_answer_ppl.py \
  --hf_dataset_path "${INPUT_DATASET}" \
  --passage_set_name "${PASSAGE_SET_NAME}" \
  --mode sft \
  --img_basedir "${IMG_BASEDIR}" \
  --img_path_mode basename \
  --output_dir "${OUTPUT_DIR}" \
  --topk_docs "${TOP_K}" \
  --drop_max_tokens "${DROP_MAX_TOKENS}" \
  --count_all_passages_for_token_filter \
  --passage_format dict \
  --num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}"

echo "SFT output: ${OUTPUT_DIR}"
