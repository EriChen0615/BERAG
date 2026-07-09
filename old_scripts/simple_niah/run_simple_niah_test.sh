#!/usr/bin/env bash
# Test run for simple_niah with BERAG ESF.
# Usage:
#   ./scripts/simple_niah/run_test.sh
#   MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct ./scripts/simple_niah/run_test.sh
#
# Set MODEL_PATH (required). Optional: ADAPTER_PATH, HAYSTACK_DIR.

set -e

# Reduce CUDA allocator fragmentation (helps with OOM when reserved memory is large)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Error: MODEL_PATH is not set. Set it to your HF model path, e.g.:" >&2
  echo "  export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct" >&2
  echo "  $0" >&2
  exit 1
fi

# Optional: transition kernel (identity | log_linear). identity = no support change, avoids KV cache mismatch.
TRANSITION_KERNEL="${TRANSITION_KERNEL:-identity}"
# Optional: output dir for inference logs (and responses). Default: outputs/0226/SimpleNIAH
OUTPUT_DIR="${OUTPUT_DIR:-outputs/0226/SimpleNIAH}"
# Optional: beam width for segment_beam decode mode
BEAM_WIDTH="${BEAM_WIDTH:-1}"
# Optional: number of chunks to split context into (chunk_size derived per context). Unset = use CHUNK_SIZE.
NUM_OF_CHUNKS=20
# Optional: token chunk size (only used when NUM_OF_CHUNKS is unset)
CHUNK_SIZE="${CHUNK_SIZE:-512}"
# Optional: write results/contexts under OUTPUT_DIR (true = use outputs/0226/SimpleNIAH/results and .../contexts)
USE_OUTPUT_DIR_FOR_RESULTS="${USE_OUTPUT_DIR_FOR_RESULTS:-true}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/NeedleInAHaystack:${PYTHONPATH:-}"

# Quick test: few context lengths and depth intervals
python scripts/run_simple_niah_esf.py \
  --model_path "${MODEL_PATH}" \
  --processor_path "${PROCESSOR_PATH:-${MODEL_PATH}}" \
  ${ADAPTER_PATH:+--adapter_path "${ADAPTER_PATH}"} \
  --haystack_dir "${HAYSTACK_DIR:-PaulGrahamEssays}" \
  --needle "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day." \
  --retrieval_question "What is the best thing to do in San Francisco?" \
  --context_lengths_min 60000 \
  --context_lengths_max 60000 \
  --context_lengths_num_intervals 1 \
  --document_depth_percent_min 48 \
  --document_depth_percent_max 48 \
  --document_depth_percent_intervals 1 \
  --save_results true \
  --save_contexts true \
  --segment_size 1 \
  --debug true \
  --state_explore_TopK 10 \
  --state_explore_mode TopP_capped \
  --top_p 0.99 \
  --decode_mode segment_beam \
  --beam_width "${BEAM_WIDTH}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_output_dir_for_results "${USE_OUTPUT_DIR_FOR_RESULTS}" \
  --transition_kernel "${TRANSITION_KERNEL}" \
  $([[ -n "${NUM_OF_CHUNKS}" ]] && echo "--num_of_chunks ${NUM_OF_CHUNKS}" || echo "--chunk_size ${CHUNK_SIZE}") \
  --max_new_tokens 128 \
  "$@"
