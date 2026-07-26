#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/src/train/LlamaFactory-0.9.5-beft/.venv/bin/python}"
RETRIEVAL_JSONL="${RETRIEVAL_JSONL:-/workspace/projects/Infoseek-retrieval/outputs/infoseek_train_evaclip_top100.sample64000.seed42.jsonl}"
ECHOSIGHT_TRAIN_CSV="${ECHOSIGHT_TRAIN_CSV:-/workspace/projects/Infoseek-retrieval/data/echosight/infoseek_train_filtered.csv}"
KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-/workspace/projects/Infoseek-retrieval/data/echosight/infoseek_100k_wiki/wiki_100_dict_v4.json}"
TOP_K="${TOP_K:-2}"
MIN_MAPPED_DOCS="${MIN_MAPPED_DOCS:-${TOP_K}}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/outputs/jinghong_chen/Infoseek-train64000-echosight-top${TOP_K}}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-${OUTPUT_BASE}-with-retrieval}"
"${PYTHON_BIN}" src/curate/convert_infoseek_echosight_retrieval_to_hf.py \
  --retrieval_jsonl "${RETRIEVAL_JSONL}" \
  --echosight_train_csv "${ECHOSIGHT_TRAIN_CSV}" \
  --knowledge_base "${KNOWLEDGE_BASE}" \
  --output_data_dir "${OUTPUT_DATA_DIR}" \
  --top_k "${TOP_K}" \
  --min_mapped_docs "${MIN_MAPPED_DOCS}" \
  --image_path_mode basename

echo "Converted data: ${OUTPUT_DATA_DIR}"
