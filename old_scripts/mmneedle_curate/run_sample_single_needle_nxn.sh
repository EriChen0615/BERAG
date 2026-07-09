#!/usr/bin/env bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake
#SBATCH -J mmneedle_needle
set -euo pipefail

# Example:
#   N_GRID=1 NUM_SEQUENCES=100 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_sample_single_needle_nxn.sh
#   N_GRID=2 NUM_SEQUENCES=100 SEQUENCE_LENGTH=10 bash scripts/mmneedle_curate/run_sample_single_needle_nxn.sh
#   RUN_BOTH=true NUM_SEQUENCES=200 bash scripts/mmneedle_curate/run_sample_single_needle_nxn.sh

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# cd "${ROOT_DIR}"

# if [[ -f "scripts/hpc_activate_env_py310_infer.sh" ]]; then
#   # shellcheck disable=SC1091
#   source scripts/hpc_activate_env_py310_infer.sh
# fi

COCO_TRAIN_DIR="${COCO_TRAIN_DIR:-/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/MSCOCO2014/train2014}"
IMAGES_ROOT="${IMAGES_ROOT:-../vqa_data/MMNeedle/train/images_stitched}"
METADATA_ROOT="${METADATA_ROOT:-../vqa_data/MMNeedle/train/metadata_stitched}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
NUM_SEQUENCES="${NUM_SEQUENCES:-10000}"
POSITIVE_RATIO="${POSITIVE_RATIO:-0.5}"
SEED="${SEED:-0}"
RUN_BOTH="${RUN_BOTH:-false}"
N_GRID="${N_GRID:-1}"

run_one() {
  local n="$1"
  python scripts/mmneedle_curate/sample_single_needle_nxn.py \
    --n_grid "${n}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --num_sequences "${NUM_SEQUENCES}" \
    --positive_ratio "${POSITIVE_RATIO}" \
    --seed "${SEED}" \
    --images_root "${IMAGES_ROOT}" \
    --metadata_root "${METADATA_ROOT}" \
    --coco_train_dir "${COCO_TRAIN_DIR}"
}

if [[ "${RUN_BOTH}" == "true" ]]; then
  run_one 1
  run_one 2
else
  run_one "${N_GRID}"
fi

echo ""
echo "Stage 2 complete. Please inspect annotation outputs before proceeding:"
echo "  ${METADATA_ROOT}/annotations_${SEQUENCE_LENGTH}_1_1.json"
echo "  ${METADATA_ROOT}/annotations_${SEQUENCE_LENGTH}_2_2.json (if generated)"

