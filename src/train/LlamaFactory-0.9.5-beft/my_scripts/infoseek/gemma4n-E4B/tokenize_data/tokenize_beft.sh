#!/bin/bash
#SBATCH -J is_g4e4b_tok_beft
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${LLAMAFACTORY_ROOT}"

source "${LLAMAFACTORY_ROOT}/.venv/bin/activate"
export TOKENIZERS_PARALLELISM=false

which python
llamafactory-cli train my_configs/infoseek/gemma4n-E4B/tokenize_data/beft/beft_k2_prior_mlp.yaml

