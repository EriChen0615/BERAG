#!/bin/bash
# Start BERAG-DPP server in background and wait until it is healthy.
# Usage:
#   bash scripts/ruler_infer/start_berag_dpp_server.sh MODEL_PATH TOKENIZER_PATH PORT VLLM_PORT CHUNK_SIZE DEBUG NUM_SUBSET_SAMPLES SEGMENT_LENGTH NUM_LOOK_AHEAD LOOKAHEAD_ROLLOUT MAX_SUBSET_SIZE BETA ADD_ANSWER_PREFIX

set -euo pipefail

MODEL_PATH="${1:?model path required}"
TOKENIZER_PATH="${2:-${MODEL_PATH}}"
PORT="${3:-5001}"
VLLM_PORT="${4:-5000}"
CHUNK_SIZE="${5:-512}"
DEBUG="${6:-${BERAG_DPP_DEBUG:-}}"
NUM_SUBSET_SAMPLES="${7:-${BERAG_DPP_NUM_SUBSET_SAMPLES:-4}}"
SEGMENT_LENGTH="${8:-${BERAG_DPP_SEGMENT_LENGTH:-8}}"
NUM_LOOK_AHEAD="${9:-${BERAG_DPP_NUM_LOOK_AHEAD:-8}}"
LOOKAHEAD_ROLLOUT="${10:-${BERAG_DPP_LOOKAHEAD_ROLLOUT:-1}}"
MAX_SUBSET_SIZE="${11:-${BERAG_DPP_MAX_SUBSET_SIZE:-2}}"
BETA="${12:-${BERAG_DPP_BETA:-1.0}}"
ADD_ANSWER_PREFIX="${13:-${BERAG_DPP_ADD_ANSWER_PREFIX:-0}}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RULER_SCRIPTS_DIR="${REPO_ROOT}/third_party/RULER/scripts"
cd "${RULER_SCRIPTS_DIR}"

echo "[start_berag_dpp_server] Starting BERAG-DPP in background from ${RULER_SCRIPTS_DIR}..."
DEBUG_FLAG=""
if [ -n "${DEBUG}" ] && [ "${DEBUG}" != "0" ] && [ "${DEBUG}" != "false" ]; then
  DEBUG_FLAG="--debug"
fi

ADD_ANSWER_PREFIX_FLAG=""
if [ -n "${ADD_ANSWER_PREFIX}" ] && [ "${ADD_ANSWER_PREFIX}" != "0" ] && [ "${ADD_ANSWER_PREFIX}" != "false" ]; then
  ADD_ANSWER_PREFIX_FLAG="--add_answer_prefix"
fi

python pred/serve_berag_dpp.py \
  --port "${PORT}" \
  --vllm-host 127.0.0.1 \
  --vllm-port "${VLLM_PORT}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --chunk_size "${CHUNK_SIZE}" \
  --segment_length "${SEGMENT_LENGTH}" \
  --num_look_ahead "${NUM_LOOK_AHEAD}" \
  --lookahead_rollout "${LOOKAHEAD_ROLLOUT}" \
  --num_subset_samples "${NUM_SUBSET_SAMPLES}" \
  --max_subset_size "${MAX_SUBSET_SIZE}" \
  --beta "${BETA}" \
  ${DEBUG_FLAG} \
  ${ADD_ANSWER_PREFIX_FLAG} &

BERAG_DPP_PID=$!
echo "[start_berag_dpp_server] BERAG-DPP PID: ${BERAG_DPP_PID}"

for i in $(seq 1 120); do
  if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q 200; then
    echo "[start_berag_dpp_server] BERAG-DPP server is ready (after ${i}s)."
    exit 0
  fi
  if ! kill -0 "${BERAG_DPP_PID}" 2>/dev/null; then
    echo "[start_berag_dpp_server] BERAG-DPP process exited unexpectedly."
    wait "${BERAG_DPP_PID}" || true
    exit 1
  fi
  sleep 1
done

echo "[start_berag_dpp_server] Timed out waiting for BERAG-DPP server."
kill "${BERAG_DPP_PID}" 2>/dev/null || true
exit 1
