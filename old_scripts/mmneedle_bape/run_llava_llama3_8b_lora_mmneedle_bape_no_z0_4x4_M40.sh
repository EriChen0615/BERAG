#!/usr/bin/env bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH -J mmneedle_bape_llava8b_4x4_M40
set -euo pipefail

N_GRID=4 \
ADD_Z0=false \
HAYSTACK_M=40 \
TAKE_N=256 \
bash scripts/mmneedle_bape/run_llava_llama3_8b_lora_mmneedle_bape_no_z0_nxn_M40.sh
