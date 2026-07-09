#!/usr/bin/env bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake
#SBATCH -J mmneedle_full_curate
set -euo pipefail

# End-to-end MMNeedle curation pipeline on CPU for all grids:
#   1) create stitched images
#   2) sample single-needle annotations
#   3) curate train_sharegpt (both no-z0 and z0)
#
# This script writes a summary text report listing generated files.

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# cd "${ROOT_DIR}"

# if [[ -f "scripts/hpc_activate_env_py310_infer.sh" ]]; then
#   # shellcheck disable=SC1091
#   source scripts/hpc_activate_env_py310_infer.sh
# fi

COCO_TRAIN_DIR="${COCO_TRAIN_DIR:-/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/MSCOCO2014/train2014}"
CAPTIONS_JSON="${CAPTIONS_JSON:-../vqa_data/MSCOCO2014/annotations/captions_train2014.json}"

IMAGES_ROOT="${IMAGES_ROOT:-../vqa_data/MMNeedle/train/images_stitched}"
METADATA_ROOT="${METADATA_ROOT:-../vqa_data/MMNeedle/train/metadata_stitched}"
CURATED_ROOT="${CURATED_ROOT:-../vqa_data/MMNeedle/train/curated}"

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
NUM_SEQUENCES="${NUM_SEQUENCES:-10000}"
POSITIVE_RATIO="${POSITIVE_RATIO:-0.5}"
SUBIMAGE_RES="${SUBIMAGE_RES:-256}"
K="${K:-2}"
TAKE_N="${TAKE_N:-10000}"
OFFSET="${OFFSET:-0}"
SEED="${SEED:-0}"

# Space-separated list, e.g. "1 2 4 8"
GRIDS="${GRIDS:-2 4 8}"

mkdir -p "${CURATED_ROOT}"
SUMMARY_FILE="${SUMMARY_FILE:-${CURATED_ROOT}/full_curate_summary_$(date +%Y%m%d_%H%M%S).txt}"

{
  echo "MMNeedle Full Curation Summary"
  echo "Generated at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo ""
  echo "Parameters:"
  echo "  COCO_TRAIN_DIR=${COCO_TRAIN_DIR}"
  echo "  CAPTIONS_JSON=${CAPTIONS_JSON}"
  echo "  IMAGES_ROOT=${IMAGES_ROOT}"
  echo "  METADATA_ROOT=${METADATA_ROOT}"
  echo "  CURATED_ROOT=${CURATED_ROOT}"
  echo "  GRIDS=${GRIDS}"
  echo "  SEQUENCE_LENGTH=${SEQUENCE_LENGTH}"
  echo "  NUM_IMAGES=${NUM_IMAGES}"
  echo "  NUM_SEQUENCES=${NUM_SEQUENCES}"
  echo "  POSITIVE_RATIO=${POSITIVE_RATIO}"
  echo "  SUBIMAGE_RES=${SUBIMAGE_RES}"
  echo "  K=${K}"
  echo "  TAKE_N=${TAKE_N}"
  echo "  OFFSET=${OFFSET}"
  echo "  SEED=${SEED}"
  echo ""
} > "${SUMMARY_FILE}"

for g in ${GRIDS}; do
  echo "=== Running grid ${g}x${g} ==="

  # Stage 1: stitched images
  python scripts/mmneedle_curate/sample_stitched_images.py \
    --coco_train_dir "${COCO_TRAIN_DIR}" \
    --n_grid "${g}" \
    --num_images "${NUM_IMAGES}" \
    --subimage_res "${SUBIMAGE_RES}" \
    --seed "${SEED}" \
    --images_root "${IMAGES_ROOT}" \
    --metadata_root "${METADATA_ROOT}"

  # Stage 2: single-needle annotations
  python scripts/mmneedle_curate/sample_single_needle_nxn.py \
    --n_grid "${g}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --num_sequences "${NUM_SEQUENCES}" \
    --positive_ratio "${POSITIVE_RATIO}" \
    --seed "${SEED}" \
    --images_root "${IMAGES_ROOT}" \
    --metadata_root "${METADATA_ROOT}" \
    --coco_train_dir "${COCO_TRAIN_DIR}"

  # Stage 3A: curate without z0
  python scripts/mmneedle_curate/curate_n_by_n_training.py \
    --n_grid "${g}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --k "${K}" \
    --take_n "${TAKE_N}" \
    --offset "${OFFSET}" \
    --seed "${SEED}" \
    --metadata_root "${METADATA_ROOT}" \
    --captions_json "${CAPTIONS_JSON}"

  # Stage 3B: curate with z0
  python scripts/mmneedle_curate/curate_n_by_n_training.py \
    --n_grid "${g}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --k "${K}" \
    --add_z0 \
    --take_n "${TAKE_N}" \
    --offset "${OFFSET}" \
    --seed "${SEED}" \
    --metadata_root "${METADATA_ROOT}" \
    --captions_json "${CAPTIONS_JSON}"

  NO_Z0_DIR="${CURATED_ROOT}/rag${SEQUENCE_LENGTH}-mmneedle-n${g}x${g}-k=${K}-beft-size=${TAKE_N}-offset=${OFFSET}"
  Z0_DIR="${CURATED_ROOT}/rag${SEQUENCE_LENGTH}-mmneedle-n${g}x${g}-k=${K}-z0-beft-size=${TAKE_N}-offset=${OFFSET}"

  {
    echo "Configuration: ${g}x${g}"
    echo "  Stage 1 images dir: ${IMAGES_ROOT}/${g}_${g}/"
    echo "  Stage 1 metadata: ${METADATA_ROOT}/${g}_${g}.json"
    echo "  Stage 2 annotations: ${METADATA_ROOT}/annotations_${SEQUENCE_LENGTH}_${g}_${g}.json"
    echo "  Stage 3 (no_z0) train: ${NO_Z0_DIR}/train_sharegpt.json"
    echo "  Stage 3 (no_z0) stats: ${NO_Z0_DIR}/stats.json"
    echo "  Stage 3 (z0) train: ${Z0_DIR}/train_sharegpt.json"
    echo "  Stage 3 (z0) stats: ${Z0_DIR}/stats.json"
    echo ""
  } >> "${SUMMARY_FILE}"
done

echo "All stages completed. Summary written to:"
echo "  ${SUMMARY_FILE}"

