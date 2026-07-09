#!/bin/bash
# Start BERAG server in background and wait until it is healthy.
# Usage: start_berag_server.sh MODEL_PATH [PORT] [CHUNK_SIZE] [DEBUG] [BEAM_WIDTH] [SEGMENT_LENGTH] [MAX_COMPOSITE_SIZE] [SEGMENT_GEN_BATCH_SIZE] [BEAM_SCORE_MODE]
#   MODEL_PATH            path to the model (required)
#   PORT                  server port (default 5000)
#   CHUNK_SIZE            token chunk size for long context (default 512)
#   DEBUG                 non-empty to enable --debug (e.g. 1 or yes); or set BERAG_DEBUG=1
#   BEAM_WIDTH            beam width for BERAG-SG segment-level beam search (default 1 = greedy)
#   SEGMENT_LENGTH        segment length for BERAG-SG (default 4)
#   MAX_COMPOSITE_SIZE    max composite chunk size (default 1 = no composite states)
#   SEGMENT_GEN_BATCH_SIZE batch size for batched segment generation (default 4)
#   BEAM_SCORE_MODE       BERAG-SG beam ranking mode: marginal or proposal_chunk (default marginal)
#
# Optional env: BERAG_DEBUG=1, BERAG_BEAM_WIDTH=N, BERAG_SEGMENT_LENGTH=N, BERAG_MAX_COMPOSITE_SIZE=N, BERAG_SEGMENT_GEN_BATCH_SIZE=N, BERAG_BEAM_SCORE_MODE=proposal_chunk.

set -e

if [ -z "${1}" ]; then
    echo "Usage: $0 MODEL_PATH [PORT] [CHUNK_SIZE] [DEBUG] [BEAM_WIDTH] [SEGMENT_LENGTH] [MAX_COMPOSITE_SIZE] [SEGMENT_GEN_BATCH_SIZE] [BEAM_SCORE_MODE]"
    exit 1
fi

MODEL_PATH="${1}"
PORT="${2:-5000}"
CHUNK_SIZE="${3:-512}"
# 4th arg or BERAG_DEBUG env enables debug
DEBUG="${4:-${BERAG_DEBUG:-}}"
# 5th arg or BERAG_BEAM_WIDTH env
BEAM_WIDTH="${5:-${BERAG_BEAM_WIDTH:-1}}"
# 6th arg or BERAG_SEGMENT_LENGTH env
SEGMENT_LENGTH="${6:-${BERAG_SEGMENT_LENGTH:-4}}"
# 7th arg or BERAG_MAX_COMPOSITE_SIZE env
MAX_COMPOSITE_SIZE="${7:-${BERAG_MAX_COMPOSITE_SIZE:-1}}"
# 8th arg or BERAG_SEGMENT_GEN_BATCH_SIZE env
SEGMENT_GEN_BATCH_SIZE="${8:-${BERAG_SEGMENT_GEN_BATCH_SIZE:-4}}"
# 9th arg or BERAG_BEAM_SCORE_MODE env
BEAM_SCORE_MODE="${9:-${BERAG_BEAM_SCORE_MODE:-marginal}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RULER_SCRIPTS_DIR="${REPO_ROOT}/third_party/RULER/scripts"

echo "[start_berag_server] Model: ${MODEL_PATH}, port: ${PORT}, chunk_size: ${CHUNK_SIZE}, debug: ${DEBUG:+yes}, beam_width: ${BEAM_WIDTH}, segment_length: ${SEGMENT_LENGTH}, max_composite_size: ${MAX_COMPOSITE_SIZE}, segment_gen_batch_size: ${SEGMENT_GEN_BATCH_SIZE}, beam_score_mode: ${BEAM_SCORE_MODE}"
echo "[start_berag_server] Starting BERAG in background from ${RULER_SCRIPTS_DIR}..."

EXTRA_ARGS=()
if [ -n "${DEBUG}" ]; then
    EXTRA_ARGS+=(--debug)
fi

cd "${RULER_SCRIPTS_DIR}"
python pred/serve_berag.py \
    --model="${MODEL_PATH}" \
    --port "${PORT}" \
    --chunk_size "${CHUNK_SIZE}" \
    --beam_width "${BEAM_WIDTH}" \
    --segment_length "${SEGMENT_LENGTH}" \
    --max_composite_size "${MAX_COMPOSITE_SIZE}" \
    --segment_gen_batch_size "${SEGMENT_GEN_BATCH_SIZE}" \
    --beam_score_mode "${BEAM_SCORE_MODE}" \
    "${EXTRA_ARGS[@]}" \
    &

BERAG_PID=$!
echo "[start_berag_server] BERAG PID: ${BERAG_PID}"

if [ -n "${BERAG_PID_FILE:-}" ]; then
    PID_DIR="$(dirname "${BERAG_PID_FILE}")"
    mkdir -p "${PID_DIR}" 2>/dev/null || true
    echo "${BERAG_PID}" > "${BERAG_PID_FILE}"
    echo "[start_berag_server] Wrote BERAG PID to ${BERAG_PID_FILE}"
fi

echo "[start_berag_server] Waiting for server to be ready (up to 60 min)..."
for i in $(seq 1 3600); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q 200; then
        echo "[start_berag_server] BERAG server is ready (after ${i}s)."
        exit 0
    fi
    if ! kill -0 "${BERAG_PID}" 2>/dev/null; then
        echo "[start_berag_server] BERAG process exited unexpectedly."
        exit 1
    fi
    sleep 1
done

echo "[start_berag_server] Timeout waiting for server (60 min)."
kill "${BERAG_PID}" 2>/dev/null || true
exit 1
