#!/bin/bash
#SBATCH -J is_g4e4b_sft
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${LLAMAFACTORY_ROOT}"

source "${LLAMAFACTORY_ROOT}/.venv/bin/activate"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-InfoSeek-Gemma4-E4B-RAG2-SFT}"

which python
llamafactory-cli train my_configs/infoseek/gemma4n-E4B/sft/rag2_answer_sft.yaml

