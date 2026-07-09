#!/bin/bash
# Create conda environment A_RAVQA-py310 for inference (Qwen2.5-VL, flash-attn 2, transformers>=4.53).
# Intended to be run as a Slurm CPU job: sbatch scripts/create_A_RAVQA_py310_env.sh
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake

set -euo pipefail

ENV_PATH="/rds/project/rds-hirYTW1FQIw/shared_space/envs/A_RAVQA-py310"
PYTHON_VERSION="3.10"

# Dao-AILab flash-attention v2.8.3 prebuilt wheel: Python 3.10, torch 2.4, CUDA 12, linux x86_64 (cxx11abi FALSE = default for pip torch)
FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "Creating environment at ${ENV_PATH} (Python ${PYTHON_VERSION})"

# # Initialize conda for this shell if needed
# if ! command -v conda &>/dev/null; then
#     if [[ -f "${CONDA_PREFIX:-/none}/etc/profile.d/conda.sh" ]]; then
#         source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
#     elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
#         source "$HOME/miniconda3/etc/profile.d/conda.sh"
#     elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
#         source "$HOME/anaconda3/etc/profile.d/conda.sh"
#     else
#         echo "ERROR: conda not found. Load conda module or source your conda install." >&2
#         exit 1
#     fi
# fi

# Create env if it does not exist
if [[ ! -d "${ENV_PATH}" ]]; then
    conda create -y -p "${ENV_PATH}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_PATH}"

# PyTorch 2.4 with CUDA 12.1 (matches flash-attn cu12torch2.4 wheel)
pip install --upgrade pip
pip install "torch==2.4.0" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121

# Transformers >= 4.53.2 for Qwen2_5_VLForConditionalGeneration
pip install "transformers>=4.53.2" "accelerate" "sentencepiece"

# Flash Attention 2 via prebuilt wheel (no GPU needed for install)
pip install "${FLASH_ATTN_WHEEL_URL}"

# Inference / MMDocRAG / BAPE dependencies
pip install "qwen-vl-utils" "tqdm" "numpy"

echo "Verifying installation..."
python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
import transformers
print('transformers:', transformers.__version__)
from transformers import Qwen2_5_VLForConditionalGeneration
print('Qwen2_5_VLForConditionalGeneration: OK')
try:
    import flash_attn
    print('flash_attn:', flash_attn.__version__)
except Exception as e:
    print('flash_attn import (may fail on CPU-only node):', e)
"
echo "Done. Activate with: conda activate ${ENV_PATH}"
