#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LLAMAFACTORY_DIR=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

CONFIG_PATH="${LLAMAFACTORY_DIR}/my_configs/evqa/qwen3-vl-8B/beft/export_beft_k2_prior_mlp_checkpoint_7901.yaml"
ADAPTER_DIR="${LLAMAFACTORY_DIR}/saves/qwen3-vl-8b/lora/evqa/beft/beft-k2-prior-mlp-r64-bs8-sample64000/checkpoint-7901"
EXPORT_DIR="/workspace/projects/BERAG/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-EVQA64000"
PRIOR_HEAD_PATH="${ADAPTER_DIR}/prior_head.pt"

if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "Missing adapter_config.json in ${ADAPTER_DIR}" >&2
  exit 1
fi

if [[ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" && ! -f "${ADAPTER_DIR}/adapter_model.bin" ]]; then
  echo "Missing adapter weights in ${ADAPTER_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PRIOR_HEAD_PATH}" ]]; then
  echo "Missing BEFT prior head: ${PRIOR_HEAD_PATH}" >&2
  exit 1
fi

if [[ -e "${EXPORT_DIR}" && "${OVERWRITE_EXPORT:-0}" != "1" ]]; then
  echo "Export directory already exists: ${EXPORT_DIR}" >&2
  echo "Set OVERWRITE_EXPORT=1 to allow LlamaFactory to overwrite it." >&2
  exit 1
fi

cd "${LLAMAFACTORY_DIR}"

if command -v llamafactory-cli >/dev/null 2>&1; then
  llamafactory-cli export "${CONFIG_PATH}"
else
  PYTHONPATH="${LLAMAFACTORY_DIR}/src:${PYTHONPATH:-}" python -m llamafactory.cli export "${CONFIG_PATH}"
fi

cp -f "${PRIOR_HEAD_PATH}" "${EXPORT_DIR}/prior_head.pt"

cat > "${EXPORT_DIR}/beft_prior_config.json" <<'JSON'
{
  "prior_head_path": "prior_head.pt",
  "prior_modeling": "mlp_head",
  "prior_head_num_layers": 2,
  "prior_head_proj_dim": 1024,
  "beft_hidden_state_offset": 4,
  "default_prior_token_offset": -4
}
JSON

test -f "${EXPORT_DIR}/config.json"
test -f "${EXPORT_DIR}/prior_head.pt"

echo "Merged model exported to ${EXPORT_DIR}"
echo "Copied BEFT prior head to ${EXPORT_DIR}/prior_head.pt"
