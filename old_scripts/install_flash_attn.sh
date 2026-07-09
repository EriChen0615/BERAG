#!/bin/bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake

set -euo pipefail

# Activate environment
# source /home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/scripts/hpc_activate_env.sh

# Ensure CUDA 12.1
export CUDA_HOME=/usr/local/software/cuda/12.1
export LD_LIBRARY_PATH=/usr/local/software/cuda/12.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/software/cuda/12.1/bin:$PATH

# Verify
which nvcc
nvcc -V
# python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Limit build parallelism and target arch
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="8.0"

# Install flash-attn from source
pip install -U ninja packaging psutil

WORKDIR="${WORKDIR:-../tmp/flash-attn-src}"
# rm -rf "${WORKDIR}"
# git clone --recursive https://github.com/Dao-AILab/flash-attention.git "${WORKDIR}"
cd "${WORKDIR}"
# git submodule update --init --recursive

TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 \
pip install -v --no-build-isolation .