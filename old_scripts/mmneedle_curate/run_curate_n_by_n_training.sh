#!/usr/bin/env bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake
#SBATCH -J mmneedle_curate
set -euo pipefail

# Example:
#   N_GRID=1 TAKE_N=500 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_curate_n_by_n_training.sh
#   N_GRID=2 TAKE_N=500 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_curate_n_by_n_training.sh
#   RUN_BOTH=true TAKE_N=500 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_curate_n_by_n_training.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# if [[ -f "scripts/hpc_activate_env_py310_infer.sh" ]]; then
#   # shellcheck disable=SC1091
#   source scripts/hpc_activate_env_py310_infer.sh
# fi

METADATA_ROOT="${METADATA_ROOT:-../vqa_data/MMNeedle/train/metadata_stitched}"
CAPTIONS_JSON="${CAPTIONS_JSON:-../vqa_data/MSCOCO2014/annotations/captions_train2014.json}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
# TAKE_N="${TAKE_N:-10000}"
TAKE_N="${TAKE_N:-10000}"
OFFSET="${OFFSET:-0}"
SEED="${SEED:-0}"
RUN_BOTH="${RUN_BOTH:-false}"
N_GRID="${N_GRID:-1}"
K="${K:-2}"
ADD_Z0="${ADD_Z0:-true}"

run_one() {
  local n="$1"
  python scripts/mmneedle_curate/curate_n_by_n_training.py \
    --n_grid "${n}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --k "${K}" \
    --take_n "${TAKE_N}" \
    --offset "${OFFSET}" \
    --seed "${SEED}" \
    --metadata_root "${METADATA_ROOT}" \
    --captions_json "${CAPTIONS_JSON}" \
    $( [[ "${ADD_Z0}" == "true" ]] && echo "--add_z0" )
}

if [[ "${RUN_BOTH}" == "true" ]]; then
  run_one 1
  run_one 2
else
  run_one "${N_GRID}"
fi

echo ""
echo "Stage 3 complete. Please confirm curated outputs before launching training."

