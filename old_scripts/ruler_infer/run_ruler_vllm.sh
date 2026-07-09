#!/bin/bash
# Entry point: run from repo root. Uses scripts/ruler_infer/run.sh (vLLM check + inference).
# Config (model, tasks, seq lengths) is in scripts/ruler_infer/run.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
echo "[run_ruler_vllm] Running from $(pwd)"
bash "${SCRIPT_DIR}/run.sh"
echo "[run_ruler_vllm] run.sh finished with exit code $?"
