#!/bin/bash
# Start vLLM server in background and wait until it is healthy.
# Usage: start_vllm_server.sh MODEL_PATH [PORT] [GPUS]
#   MODEL_PATH  path to the model (required)
#   PORT        server port (default 5000)
#   GPUS        tensor-parallel size (default 1)

set -e

if [ -z "${1}" ]; then
    echo "Usage: $0 MODEL_PATH [PORT] [GPUS]"
    exit 1
fi

MODEL_PATH="${1}"
PORT="${2:-5000}"
GPUS="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RULER_SCRIPTS_DIR="${REPO_ROOT}/third_party/RULER/scripts"

echo "[start_vllm_server] Model: ${MODEL_PATH}, port: ${PORT}, GPUs: ${GPUS}"
echo "[start_vllm_server] Starting vLLM in background from ${RULER_SCRIPTS_DIR}..."

cd "${RULER_SCRIPTS_DIR}"
python pred/serve_vllm.py \
    --model="${MODEL_PATH}" \
    --tensor-parallel-size="${GPUS}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    &

VLLM_PID=$!
echo "[start_vllm_server] vLLM PID: ${VLLM_PID}"

echo "[start_vllm_server] Waiting for server to be ready (up to 60 min)..."
for i in $(seq 1 3600); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q 200; then
        echo "[start_vllm_server] vLLM server is ready (after ${i}s)."
        exit 0
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[start_vllm_server] vLLM process exited unexpectedly."
        exit 1
    fi
    sleep 1
done

echo "[start_vllm_server] Timeout waiting for server (60 min)."
kill "${VLLM_PID}" 2>/dev/null || true
exit 1
