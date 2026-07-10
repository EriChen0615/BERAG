#!/bin/bash
#SBATCH -J evqa_q3vl8b_rag5_sft
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${LLAMAFACTORY_ROOT}"

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-EVQA-Qwen3VL-RAG5-SFT}"
which python

llamafactory-cli train my_configs/evqa/qwen3-vl-8B/sft/rag5_answer_sft.yaml
