#!/bin/bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake

set -euo pipefail

source scripts/hpc_activate_env.sh

bash scripts/mmdocrag_curate/ragk_mmdocrag_beft.sh
