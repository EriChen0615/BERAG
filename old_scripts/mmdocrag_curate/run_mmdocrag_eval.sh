#!/bin/bash
set -euo pipefail

SETTING=${1:-20}
RESPONSE_JSONL=${2:-"third_party/MMDocRAG/response/mmdocrag_berag_quotes${SETTING}_response.jsonl"}

python third_party/MMDocRAG/eval_all.py \
  --path "${RESPONSE_JSONL}" \
  --setting "${SETTING}"
