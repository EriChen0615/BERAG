#!/usr/bin/env bash
# Simple NIAH with BERAG-SG inference engine (segment-level beam search + composite states).
# Usage (from repo root):
#   ./scripts/simple_niah/run_simple_niah_berag_sg.sh
#   MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct ./scripts/simple_niah/run_simple_niah_berag_sg.sh
#
# Required: MODEL_PATH (or set default below). Optional: ADAPTER_PATH, HAYSTACK_DIR, PYTHON.

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

# Python: prefer python3 for portability (e.g. RHEL/CentOS where python is not in PATH)
if [[ -n "${PYTHON}" ]]; then
  PYTHON_CMD="${PYTHON}"
elif command -v python3 &>/dev/null; then
  PYTHON_CMD=python3
else
  PYTHON_CMD=python
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/0226/SimpleNIAH_BERAGSG}"
# Decode suffix: auto from args (segment_length, beam_size, top_p, etc.) unless DECODE_SUFFIX is set
DECODE_SUFFIX="${DECODE_SUFFIX:-}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-10}"
BERAG_SG_BEAM_SIZE="${BERAG_SG_BEAM_SIZE:-2}"
MAX_COMPOSITE_SIZE="${MAX_COMPOSITE_SIZE:-2}"
BERAG_SG_TOP_P="${BERAG_SG_TOP_P:-0.95}"
BERAG_SG_TEMPERATURE="${BERAG_SG_TEMPERATURE:-0.5}"
BERAG_SG_BEAM_PRUNE="${BERAG_SG_BEAM_PRUNE:-diverse_beam_search}"
NUM_OF_CHUNKS="${NUM_OF_CHUNKS:-5}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
USE_OUTPUT_DIR_FOR_RESULTS="${USE_OUTPUT_DIR_FOR_RESULTS:-true}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/NeedleInAHaystack:${PYTHONPATH:-}"

"${PYTHON_CMD}" scripts/simple_niah/run_simple_niah_berag_sg.py \
  --model_path "${MODEL_PATH}" \
  --processor_path "${PROCESSOR_PATH:-${MODEL_PATH}}" \
  ${ADAPTER_PATH:+--adapter_path "${ADAPTER_PATH}"} \
  --haystack_dir "${HAYSTACK_DIR:-PaulGrahamEssays}" \
  --needle "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day." \
  --retrieval_question "What is the best thing to do in San Francisco?" \
  --context_lengths_min 10000 \
  --context_lengths_max 10000 \
  --context_lengths_num_intervals 1 \
  --document_depth_percent_min 48 \
  --document_depth_percent_max 48 \
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
  --debug true \
  "$@"
