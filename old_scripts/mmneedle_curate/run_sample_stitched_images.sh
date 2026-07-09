#!/usr/bin/env bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake
#SBATCH -J mmneedle_stitched
set -euo pipefail

# Example:
#   N_GRID=1 NUM_IMAGES=50 bash scripts/mmneedle_curate/run_sample_stitched_images.sh
#   N_GRID=2 NUM_IMAGES=50 bash scripts/mmneedle_curate/run_sample_stitched_images.sh
#   RUN_BOTH=true NUM_IMAGES=100 bash scripts/mmneedle_curate/run_sample_stitched_images.sh

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# cd "${ROOT_DIR}"

# if [[ -f "scripts/hpc_activate_env_py310_infer.sh" ]]; then
#   # shellcheck disable=SC1091
#   source scripts/hpc_activate_env_py310_infer.sh
# fi

COCO_TRAIN_DIR="${COCO_TRAIN_DIR:-/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/MSCOCO2014/train2014}"
IMAGES_ROOT="${IMAGES_ROOT:-../vqa_data/MMNeedle/train/images_stitched}"
METADATA_ROOT="${METADATA_ROOT:-../vqa_data/MMNeedle/train/metadata_stitched}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
SUBIMAGE_RES="${SUBIMAGE_RES:-256}"
SEED="${SEED:-0}"
RUN_BOTH="${RUN_BOTH:-false}"
N_GRID="${N_GRID:-1}"

run_one() {
  local n="$1"
  python scripts/mmneedle_curate/sample_stitched_images.py \
    --coco_train_dir "${COCO_TRAIN_DIR}" \
    --n_grid "${n}" \
    --num_images "${NUM_IMAGES}" \
    --subimage_res "${SUBIMAGE_RES}" \
    --seed "${SEED}" \
    --images_root "${IMAGES_ROOT}" \
    --metadata_root "${METADATA_ROOT}"
}

if [[ "${RUN_BOTH}" == "true" ]]; then
  run_one 1
  run_one 2
else
  run_one "${N_GRID}"
fi

echo ""
echo "Stage 1 complete. Please inspect stitched outputs before proceeding:"
echo "  ${IMAGES_ROOT}/1_1, ${IMAGES_ROOT}/2_2 (if generated)"
echo "  ${METADATA_ROOT}/1_1.json, ${METADATA_ROOT}/2_2.json (if generated)"

