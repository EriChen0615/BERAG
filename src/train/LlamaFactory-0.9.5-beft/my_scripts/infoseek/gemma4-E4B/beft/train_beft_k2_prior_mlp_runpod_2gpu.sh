#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${LLAMAFACTORY_ROOT}/../../.." && pwd)}"
cd "${LLAMAFACTORY_ROOT}"

source "${LLAMAFACTORY_ROOT}/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=2
export TOKENIZERS_PARALLELISM=false
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-InfoSeek-Gemma4-E4B-BEFT-K2}"

CONFIG="my_configs/infoseek/gemma4-E4B/beft/beft_k2_prior_mlp.yaml"
TOKENIZED_PATH="${REPO_ROOT}/outputs/tokenized/infoseek_beft_gemma4n_e2b_sample64000_cutoff35k"

[[ -f "${CONFIG}" ]] || {
    echo "Training config not found: ${LLAMAFACTORY_ROOT}/${CONFIG}" >&2
    exit 1
}
[[ -f "${TOKENIZED_PATH}/dataset_dict.json" ]] || {
    echo "Completed tokenized dataset not found: ${TOKENIZED_PATH}" >&2
    exit 1
}

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
[[ "${GPU_COUNT}" -eq 2 ]] || {
    echo "Expected exactly 2 visible GPUs, found ${GPU_COUNT}: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
}

echo "Python: $(command -v python)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}; workers: ${NPROC_PER_NODE}"
echo "Effective batch size: 1 per GPU x 4 accumulation x 2 GPUs = 8"
echo "Config: ${CONFIG}"

llamafactory-cli train "${CONFIG}"
