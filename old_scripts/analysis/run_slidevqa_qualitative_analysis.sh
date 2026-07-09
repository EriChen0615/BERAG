#!/usr/bin/env bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake
#SBATCH -J slidevqa_qual
set -euo pipefail

# Activate your Python env before sbatch (e.g. source scripts/hpc_activate_env_py310_infer.sh).
# No environment activation is embedded here by design.
# Assume this script is submitted from repo root:
#   /home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA

CSV_PATH="${CSV_PATH:-outputs/1225/BAPE/SlideVQA/SlideVQA-BAPE-BEFT[K=4*]-prior=mlp-lr1e-6-l1h4-r64-epoch1-h4-prior=prior_head-K=20-TakeN=0/marked_inference_results.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/output/slidevqa_qualitative}"
TOP_N="${TOP_N:-5}"
SEED="${SEED:-0}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python analysis/slidevqa_qualitative_analysis.py \
  --csv_path "${CSV_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --top_n "${TOP_N}" \
  --seed "${SEED}"
