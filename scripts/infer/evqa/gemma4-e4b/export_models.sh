#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/projects/BERAG}"
LLAMAFACTORY_DIR="${REPO_ROOT}/src/train/LlamaFactory-0.9.5-beft"
MODEL_KIND="${1:-all}"

SFT_ADAPTER_DIR="${LLAMAFACTORY_DIR}/saves/gemma4-e4b/lora/evqa/sft/rag5-answer-sft-r64-bs8-sample64000-max8192/checkpoint-7992"
SFT_EXPORT_DIR="${REPO_ROOT}/outputs/jinghong_chen/Gemma-4-E4B-it-SFT-EVQA"
SFT_CONFIG="${LLAMAFACTORY_DIR}/my_configs/evqa/gemma4-e4b/sft/export_rag5_answer_sft_checkpoint_7992.yaml"

BEFT_ADAPTER_DIR="${LLAMAFACTORY_DIR}/saves/gemma4-e4b/lora/evqa/beft/beft-k2-prior-mlp-r64-bs8-full-epoch/checkpoint-7992"
BEFT_EXPORT_DIR="${REPO_ROOT}/outputs/jinghong_chen/Gemma-4-E4B-it-BEFT-EVQA"
BEFT_CONFIG="${LLAMAFACTORY_DIR}/my_configs/evqa/gemma4-e4b/beft/export_beft_k2_prior_mlp_checkpoint_7992.yaml"

usage() {
    echo "Usage: $0 [sft|beft|all]" >&2
}

validate_adapter() {
    local adapter_dir="$1"
    test -f "${adapter_dir}/adapter_config.json" || {
        echo "Missing adapter config: ${adapter_dir}/adapter_config.json" >&2
        exit 1
    }
    if [[ ! -f "${adapter_dir}/adapter_model.safetensors" && ! -f "${adapter_dir}/adapter_model.bin" ]]; then
        echo "Missing adapter weights in ${adapter_dir}" >&2
        exit 1
    fi
}

run_export() {
    local label="$1"
    local adapter_dir="$2"
    local export_dir="$3"
    local config_path="$4"
    local prior_head="${5:-}"

    validate_adapter "${adapter_dir}"
    if [[ -n "${prior_head}" && ! -f "${prior_head}" ]]; then
        echo "Missing BEFT prior head: ${prior_head}" >&2
        exit 1
    fi

    if [[ -e "${export_dir}" ]]; then
        if [[ -f "${export_dir}/config.json" && ( -z "${prior_head}" || -f "${export_dir}/prior_head.pt" ) ]]; then
            echo "[${label}] Export already complete; skipping: ${export_dir}"
            return
        fi
        echo "[${label}] Export path exists but is incomplete: ${export_dir}" >&2
        echo "Move or remove that directory explicitly, then rerun this script." >&2
        exit 1
    fi

    echo "[${label}] Merging ${adapter_dir}"
    (
        cd "${LLAMAFACTORY_DIR}"
        if command -v llamafactory-cli >/dev/null 2>&1; then
            llamafactory-cli export "${config_path}"
        else
            PYTHONPATH="${LLAMAFACTORY_DIR}/src:${PYTHONPATH:-}" \
                python -m llamafactory.cli export "${config_path}"
        fi
    )

    test -f "${export_dir}/config.json"
    if [[ -n "${prior_head}" ]]; then
        cp "${prior_head}" "${export_dir}/prior_head.pt"
        printf '%s\n' \
            '{' \
            '  "prior_head_path": "prior_head.pt",' \
            '  "prior_modeling": "mlp_head",' \
            '  "prior_head_num_layers": 2,' \
            '  "prior_head_proj_dim": 1024,' \
            '  "beft_hidden_state_offset": 4,' \
            '  "default_prior_token_offset": -5' \
            '}' > "${export_dir}/beft_prior_config.json"
    fi
    echo "[${label}] Exported to ${export_dir}"
}

case "${MODEL_KIND}" in
    sft)
        run_export "Gemma-4-E4B SFT" "${SFT_ADAPTER_DIR}" "${SFT_EXPORT_DIR}" "${SFT_CONFIG}"
        ;;
    beft)
        run_export "Gemma-4-E4B BEFT" "${BEFT_ADAPTER_DIR}" "${BEFT_EXPORT_DIR}" "${BEFT_CONFIG}" \
            "${BEFT_ADAPTER_DIR}/prior_head.pt"
        ;;
    all)
        run_export "Gemma-4-E4B SFT" "${SFT_ADAPTER_DIR}" "${SFT_EXPORT_DIR}" "${SFT_CONFIG}"
        run_export "Gemma-4-E4B BEFT" "${BEFT_ADAPTER_DIR}" "${BEFT_EXPORT_DIR}" "${BEFT_CONFIG}" \
            "${BEFT_ADAPTER_DIR}/prior_head.pt"
        ;;
    *)
        usage
        exit 2
        ;;
esac
