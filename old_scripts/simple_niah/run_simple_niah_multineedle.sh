#!/usr/bin/env bash
# Multi-needle NIAH with BERAG ESF (e.g. pizza example: goat cheese and figs).
# Usage:
#   ./scripts/simple_niah/run_simple_niah_multineedle.sh
#   MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct ./scripts/simple_niah/run_simple_niah_multineedle.sh
#
# Set MODEL_PATH (required). Optional: ADAPTER_PATH, HAYSTACK_DIR.

set -e

# Reduce CUDA allocator fragmentation (helps with OOM when reserved memory is large)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Error: MODEL_PATH is not set. Set it to your HF model path, e.g.:" >&2
  echo "  export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct" >&2
  echo "  $0" >&2
  exit 1
fi

# Max allowed composite state size (e.g. 2 for two needles)
MAX_STATE_SIZE="${MAX_STATE_SIZE:-2}"
# Output dir for multi-needle runs
OUTPUT_DIR="${OUTPUT_DIR:-outputs/0226/SimpleNIAH_MultiNeedle}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
NUM_OF_CHUNKS="${NUM_OF_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
USE_OUTPUT_DIR_FOR_RESULTS="${USE_OUTPUT_DIR_FOR_RESULTS:-true}"
STATE_EXPLORE_TOPK="${STATE_EXPLORE_TOPK:-10}"

# Transition: mass_threshold_kernel (combine when both b(s)>1/K+epsilon; T(s'|s)=mass_at_destination/Z)
# Alternative: multiplicative_threshold (t_s=2/K, t_w=0.5)
TRANSITION_KERNEL="${TRANSITION_KERNEL:-mass_threshold_kernel}"
MASS_THRESHOLD_EPSILON="${MASS_THRESHOLD_EPSILON:-0.0}"
THRESHOLD_SINGLE="${THRESHOLD_SINGLE:-}"
THRESHOLD_COMBINED="${THRESHOLD_COMBINED:-0.5}"

# Pizza example: two needles (goat cheese, figs)
NEEDLES_JSON='["Goat cheese is one of the secret ingredients needed to build the perfect pizza.", "Figs are one of the secret ingredients needed to build the perfect pizza."]'
RETRIEVAL_QUESTION="${RETRIEVAL_QUESTION:-What are the secret ingredients needed to build the perfect pizza?}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/NeedleInAHaystack:${PYTHONPATH:-}"

python scripts/run_simple_niah_esf.py \
  --model_path "${MODEL_PATH}" \
  --processor_path "${PROCESSOR_PATH:-${MODEL_PATH}}" \
  ${ADAPTER_PATH:+--adapter_path "${ADAPTER_PATH}"} \
  --haystack_dir "${HAYSTACK_DIR:-PaulGrahamEssays}" \
  --retrieval_question "${RETRIEVAL_QUESTION}" \
  --multi_needle true \
  --needles "${NEEDLES_JSON}" \
  --max_state_size "${MAX_STATE_SIZE}" \
  --transition_kernel "${TRANSITION_KERNEL}" \
  --mass_threshold_epsilon "${MASS_THRESHOLD_EPSILON}" \
  ${THRESHOLD_SINGLE:+--threshold_single "${THRESHOLD_SINGLE}"} \
  --threshold_combined "${THRESHOLD_COMBINED}" \
  --context_lengths_min 10000 \
  --context_lengths_max 10000 \
  --context_lengths_num_intervals 1 \
  --document_depth_percent_min 30 \
  --document_depth_percent_max 30 \
  --document_depth_percent_intervals 1 \
  --save_results true \
  --save_contexts true \
  --segment_size 2 \
  --debug true \
  --state_explore_TopK "${STATE_EXPLORE_TOPK}" \
  --state_explore_mode TopP_capped \
  --top_p 0.99 \
  --decode_mode segment_beam \
  --beam_width "${BEAM_WIDTH}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_output_dir_for_results "${USE_OUTPUT_DIR_FOR_RESULTS}" \
  $([[ -n "${NUM_OF_CHUNKS}" ]] && echo "--num_of_chunks ${NUM_OF_CHUNKS}" || echo "--chunk_size ${CHUNK_SIZE}") \
  --max_new_tokens 128 \
  --model_name simple_niah_berag_esf_multineedle \
  "$@"
