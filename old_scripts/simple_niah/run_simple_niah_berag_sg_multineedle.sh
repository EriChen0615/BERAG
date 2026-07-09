#!/usr/bin/env bash
# Multi-needle NIAH with BERAG-SG (e.g. pizza example: goat cheese and figs).
# Usage (from repo root):
#   ./scripts/simple_niah/run_simple_niah_berag_sg_multineedle.sh
#   MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct ./scripts/simple_niah/run_simple_niah_berag_sg_multineedle.sh
#
# Required: MODEL_PATH (or set default below). Optional: ADAPTER_PATH, HAYSTACK_DIR, PYTHON,
# NEEDLES_JSON, RETRIEVAL_QUESTION, DECODE_SUFFIX, SEGMENT_LENGTH, BERAG_SG_BEAM_SIZE, MAX_COMPOSITE_SIZE,
# BERAG_SG_TOP_P, NUM_OF_CHUNKS, CHUNK_SIZE (same env vars as run_simple_niah_berag_sg.sh).

set -e

# Reduce CUDA allocator fragmentation (helps with OOM when reserved memory is large)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Error: MODEL_PATH is not set. Set it to your HF model path, e.g.:" >&2
  echo "  export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct" >&2
  echo "  $0" >&2
  exit 1
fi

# Python: prefer python3 for portability
if [[ -n "${PYTHON}" ]]; then
  PYTHON_CMD="${PYTHON}"
elif command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

# Multi-needle: max composite size (e.g. 2 for two needles so composites can combine both)
MAX_COMPOSITE_SIZE="${MAX_COMPOSITE_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/0226/SimpleNIAH_BERAGSG_MultiNeedle}"
DECODE_SUFFIX="${DECODE_SUFFIX:-}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-25}"
BERAG_SG_BEAM_SIZE="${BERAG_SG_BEAM_SIZE:-4}"
BERAG_SG_TOP_P="${BERAG_SG_TOP_P:-0.99}"
BERAG_SG_TEMPERATURE="${BERAG_SG_TEMPERATURE:-0.1}"
BERAG_SG_BEAM_PRUNE="${BERAG_SG_BEAM_PRUNE:-diverse_beam_search}"
# BERAG_SG_BEAM_PRUNE="${BERAG_SG_BEAM_PRUNE:-top_b}"
NUM_OF_CHUNKS="${NUM_OF_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
USE_OUTPUT_DIR_FOR_RESULTS="${USE_OUTPUT_DIR_FOR_RESULTS:-true}"

# Pizza example: two needles (goat cheese, figs)
# NEEDLES_JSON="${NEEDLES_JSON:-[\"Goat cheese is one of the secret ingredients needed to build the perfect pizza.\", \"Figs are one of the secret ingredients needed to build the perfect pizza.\"]}"
# RETRIEVAL_QUESTION="${RETRIEVAL_QUESTION:-What are the secret ingredients needed to build the perfect pizza?}"

NEEDLES_JSON="${NEEDLES_JSON:-[\"One secret number is 615.\", \"One secret number is 926.\"]}"
RETRIEVAL_QUESTION="${RETRIEVAL_QUESTION:-What are the secret numbers?}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/NeedleInAHaystack:${PYTHONPATH:-}"

"${PYTHON_CMD}" scripts/simple_niah/run_simple_niah_berag_sg.py \
  --model_path "${MODEL_PATH}" \
  --processor_path "${PROCESSOR_PATH:-${MODEL_PATH}}" \
  ${ADAPTER_PATH:+--adapter_path "${ADAPTER_PATH}"} \
  --haystack_dir "${HAYSTACK_DIR:-PaulGrahamEssays}" \
  --retrieval_question "${RETRIEVAL_QUESTION}" \
  --multi_needle true \
  --needles "${NEEDLES_JSON}" \
  --context_lengths_min 30000 \
  --context_lengths_max 30000 \
  --context_lengths_num_intervals 1 \
  --document_depth_percent_min 10 \
  --document_depth_percent_max 10 \
  --document_depth_percent_intervals 1 \
  --save_results true \
  --save_contexts true \
  --segment_length "${SEGMENT_LENGTH}" \
  --berag_sg_beam_size "${BERAG_SG_BEAM_SIZE}" \
  --max_composite_size "${MAX_COMPOSITE_SIZE}" \
  --berag_sg_top_p "${BERAG_SG_TOP_P}" \
  --berag_sg_temperature "${BERAG_SG_TEMPERATURE}" \
  --berag_sg_beam_prune "${BERAG_SG_BEAM_PRUNE}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_output_dir_for_results "${USE_OUTPUT_DIR_FOR_RESULTS}" \
  $([[ -n "${DECODE_SUFFIX}" ]] && echo "--decode_suffix ${DECODE_SUFFIX}") \
  $([[ -n "${NUM_OF_CHUNKS}" ]] && echo "--num_of_chunks ${NUM_OF_CHUNKS}" || echo "--chunk_size ${CHUNK_SIZE}") \
  --max_new_tokens 128 \
  --model_name simple_niah_berag_sg_multineedle \
  --debug "${DEBUG:-true}" \
  "$@"
