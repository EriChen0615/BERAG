#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
LLAMAFACTORY_DIR="${REPO_ROOT}/src/train/LlamaFactory-0.9.5-beft"
EXPORT_MODE="${EXPORT_MODE:-all}"

SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:-${LLAMAFACTORY_DIR}/saves/qwen3-vl-8b/lora/infoseek/sft/rag2-answer-sft-r64-bs8-64000-max35k/checkpoint-7992}"
BEFT_ADAPTER_DIR="${BEFT_ADAPTER_DIR:-${LLAMAFACTORY_DIR}/saves/qwen3-vl-8b/lora/infoseek/beft/beft-k2-prior-mlp-r64-bs8-sample64000/checkpoint-8792}"
SFT_EXPORT_DIR="${SFT_EXPORT_DIR:-${REPO_ROOT}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-SFT-Infoseek64000}"
BEFT_EXPORT_DIR="${BEFT_EXPORT_DIR:-${REPO_ROOT}/outputs/jinghong_chen/Qwen3-VL-8B-Instruct-BEFT-Infoseek-ckpt8792}"

SFT_CONFIG="${LLAMAFACTORY_DIR}/my_configs/infoseek/qwen3-vl-8B/sft/export_rag2_answer_sft_checkpoint_7992.yaml"
BEFT_CONFIG="${LLAMAFACTORY_DIR}/my_configs/infoseek/qwen3-vl-8B/beft/export_beft_k2_prior_mlp_checkpoint_8792.yaml"

usage() {
    echo "Usage: EXPORT_MODE=all|sft|beft bash $0" >&2
}

case "${EXPORT_MODE}" in
    all|sft|beft) ;;
    *) usage; exit 2 ;;
esac

require_adapter() {
    local adapter_dir="$1"
    [[ -f "${adapter_dir}/adapter_config.json" ]] || {
        echo "Missing adapter config: ${adapter_dir}/adapter_config.json" >&2
        exit 1
    }
    [[ -f "${adapter_dir}/adapter_model.safetensors" || -f "${adapter_dir}/adapter_model.bin" ]] || {
        echo "Missing adapter weights in ${adapter_dir}" >&2
        exit 1
    }
}

run_export() {
    local label="$1"
    local config_path="$2"
    local adapter_dir="$3"
    local export_dir="$4"

    require_adapter "${adapter_dir}"
    if [[ -e "${export_dir}" && "${OVERWRITE_EXPORT:-0}" != "1" ]]; then
        echo "Export directory already exists: ${export_dir}" >&2
        echo "Set OVERWRITE_EXPORT=1 to allow overwriting it." >&2
        exit 1
    fi

    echo "[InfoSeek ${label}] Exporting ${adapter_dir}"
    if command -v llamafactory-cli >/dev/null 2>&1; then
        llamafactory-cli export "${config_path}"
    else
        PYTHONPATH="${LLAMAFACTORY_DIR}/src:${PYTHONPATH:-}" \
            python -m llamafactory.cli export "${config_path}"
    fi
    [[ -f "${export_dir}/config.json" ]] || {
        echo "Export did not create ${export_dir}/config.json" >&2
        exit 1
    }
}

cd "${LLAMAFACTORY_DIR}"

if [[ "${EXPORT_MODE}" == "all" || "${EXPORT_MODE}" == "sft" ]]; then
    run_export "SFT" "${SFT_CONFIG}" "${SFT_ADAPTER_DIR}" "${SFT_EXPORT_DIR}"
fi

if [[ "${EXPORT_MODE}" == "all" || "${EXPORT_MODE}" == "beft" ]]; then
    [[ -f "${BEFT_ADAPTER_DIR}/prior_head.pt" ]] || {
        echo "Missing BEFT prior head: ${BEFT_ADAPTER_DIR}/prior_head.pt" >&2
        exit 1
    }
    run_export "BEFT" "${BEFT_CONFIG}" "${BEFT_ADAPTER_DIR}" "${BEFT_EXPORT_DIR}"
    cp -f "${BEFT_ADAPTER_DIR}/prior_head.pt" "${BEFT_EXPORT_DIR}/prior_head.pt"
    [[ -f "${BEFT_EXPORT_DIR}/prior_head.pt" ]] || {
        echo "Failed to copy the BEFT prior head." >&2
        exit 1
    }
fi

echo "InfoSeek model export complete."
