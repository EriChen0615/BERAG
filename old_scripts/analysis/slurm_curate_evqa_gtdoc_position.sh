#!/usr/bin/env bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake
#SBATCH -J evqa_gtdoc_curate
set -euo pipefail

# Activate your Python env before sbatch (e.g. source scripts/hpc_activate_env_py310_infer.sh).
# No conda activation is embedded here by design.

ROOT="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
cd "${ROOT}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python scripts/analysis/curate_evqa_gtdoc_position.py "$@"
