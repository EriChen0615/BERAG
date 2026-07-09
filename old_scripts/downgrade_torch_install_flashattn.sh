#!/bin/bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake

set -euo pipefail


# Downgrade torch to 2.4.0 (CUDA 12.1)
pip uninstall -y torch torchvision torchaudio
pip install --upgrade \
  "torch==2.4.0+cu121" \
  "torchvision==0.19.0+cu121" \
  "torchaudio==2.4.0+cu121" \
  --index-url https://download.pytorch.org/whl/cu121

# Determine CXX11 ABI flag used by torch
CXX11_ABI=$(python - <<'PY'
import torch
print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")
PY
)

# Install prebuilt flash-attn wheel from official release
FLASH_ATTN_VER="2.8.3"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/flash_attn-${FLASH_ATTN_VER}+cu12torch2.4cxx11abi${CXX11_ABI}-cp310-cp310-linux_x86_64.whl"

pip install --no-build-isolation "${WHEEL_URL}"
