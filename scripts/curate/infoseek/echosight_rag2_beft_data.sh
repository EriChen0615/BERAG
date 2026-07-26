#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-src/train/LlamaFactory-0.9.5-beft/.venv/bin/python}"
LLAMAFACTORY_DATA_DIR="${LLAMAFACTORY_DATA_DIR:-src/train/LlamaFactory-0.9.5-beft/data}"
IMG_BASEDIR="${IMG_BASEDIR:-${LLAMAFACTORY_DATA_DIR}/Infoseek/infoseek_train_images}"

# Inputs from the standalone /workspace/projects/Infoseek-retrieval pipeline.
RETRIEVAL_JSONL="${RETRIEVAL_JSONL:-/workspace/projects/Infoseek-retrieval/outputs/infoseek_train_evaclip_top100.sample64000.seed42.jsonl}"
ECHOSIGHT_TRAIN_CSV="${ECHOSIGHT_TRAIN_CSV:-/workspace/projects/Infoseek-retrieval/data/echosight/infoseek_train_filtered.csv}"
ECHOSIGHT_KB_JSON="${ECHOSIGHT_KB_JSON:-/workspace/projects/Infoseek-retrieval/data/echosight/infoseek_100k_wiki/wiki_100_dict_v4.json}"

# Convert top-5 retrieval once; BEFT then keeps K=2: one GT doc + one random non-GT hard negative from top-5.
CONVERTED_TOPK="${CONVERTED_TOPK:-5}"
TOPK_DOCS="${TOPK_DOCS:-2}"
RETRIEVAL_POOL_SIZE="${RETRIEVAL_POOL_SIZE:-${CONVERTED_TOPK}}"
MIN_MAPPED_DOCS="${MIN_MAPPED_DOCS:-${CONVERTED_TOPK}}"
DROP_MAX_TOKENS="${DROP_MAX_TOKENS:-32768}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-100}"
SEED="${SEED:-42}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
ADD_EXAMPLES_WITH_PSEUDODOCS="${ADD_EXAMPLES_WITH_PSEUDODOCS:-0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --add_examples_with_pseudodocs)
            ADD_EXAMPLES_WITH_PSEUDODOCS="$2"
            shift 2
            ;;
        --add_examples_with_pseudodocs=*)
            ADD_EXAMPLES_WITH_PSEUDODOCS="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

CONVERTED_DATASET="${CONVERTED_DATASET:-outputs/jinghong_chen/Infoseek-train64000-echosight-top${CONVERTED_TOPK}-with-retrieval}"
OUTPUT_DIR="${OUTPUT_DIR:-${LLAMAFACTORY_DATA_DIR}/jinghong_chen/infoseek/echosight-rag${TOPK_DOCS}-hardneg-beft-size=${SAMPLE_SIZE}-max=${DROP_MAX_TOKENS}}"

if [[ "${FORCE_CONVERT}" == "1" || ! -d "${CONVERTED_DATASET}" ]]; then
    "${PYTHON_BIN}" src/curate/convert_infoseek_echosight_retrieval_to_hf.py \
        --retrieval_jsonl "${RETRIEVAL_JSONL}" \
        --echosight_train_csv "${ECHOSIGHT_TRAIN_CSV}" \
        --knowledge_base "${ECHOSIGHT_KB_JSON}" \
        --output_data_dir "${CONVERTED_DATASET}" \
        --top_k "${CONVERTED_TOPK}" \
        --min_mapped_docs "${MIN_MAPPED_DOCS}" \
        --image_path_mode basename
else
    echo "Using existing converted HF dataset: ${CONVERTED_DATASET}"
fi

"${PYTHON_BIN}" src/curate/ragk_answer_ppl.py \
    --hf_dataset_path "${CONVERTED_DATASET}" \
    --passage_set_name "EchoSightKB:${ECHOSIGHT_KB_JSON}" \
    --mode "sft" \
    --output_style "beft" \
    --topk_docs "${TOPK_DOCS}" \
    --retrieval_pool_size "${RETRIEVAL_POOL_SIZE}" \
    --passage_selection_strategy "gt_plus_random_hard_negative" \
    --img_basedir "${IMG_BASEDIR}" \
    --img_path_mode "basename" \
    --output_dir "${OUTPUT_DIR}" \
    --drop_max_tokens "${DROP_MAX_TOKENS}" \
    --count_all_passages_for_token_filter \
    --passage_format dict \
    --add_examples_with_pseudodocs "${ADD_EXAMPLES_WITH_PSEUDODOCS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --sample_size "${SAMPLE_SIZE}"


echo "Converted HF dataset: ${CONVERTED_DATASET}"
echo "Pseudo-doc examples requested: ${ADD_EXAMPLES_WITH_PSEUDODOCS}"
echo "BEFT ShareGPT output: ${OUTPUT_DIR}/train_sharegpt.json"
