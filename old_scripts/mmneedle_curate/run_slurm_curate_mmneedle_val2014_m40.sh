#!/usr/bin/env bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake
#SBATCH -J mmneedle_curate_val_m40
set -euo pipefail

# Curate MMNeedle *test-style* data on COCO val2014 with sequence length 40 (haystack M=40).
#
# Stitching configs (default GRIDS="1 2 4"):
#   1x1  -> n_grid=1, images under images_stitched/1_1/,   metadata 1_1.json,   annotations_40_1_1.json
#   2x2  -> n_grid=2, images under images_stitched/2_2/,   metadata 2_2.json,   annotations_40_2_2.json
#   4x4  -> n_grid=4, images under images_stitched/4_4/,   metadata 4_4.json,   annotations_40_4_4.json
#
# Stages (controlled by env):
#   RUN_STITCH=true  -> sample_stitched_images.py for each of 1x1, 2x2, 4x4 (needs >= N*N val images per stitch)
#   RUN_STITCH=false -> skip stitching; expects existing IMAGES_ROOT + metadata_stitched/{N}_{N}.json
#   Always runs sample_single_needle_nxn.py per grid -> annotations_${SEQUENCE_LENGTH}_{N}_{N}.json
#
# Activate your Python env before sbatch (e.g. source scripts/hpc_activate_env_py310_infer.sh).
# No conda activation is embedded here.
#
# Example:
#   sbatch scripts/mmneedle_curate/run_slurm_curate_mmneedle_val2014_m40.sh
#
# Example (stitch already built; only annotations):
#   RUN_STITCH=false sbatch scripts/mmneedle_curate/run_slurm_curate_mmneedle_val2014_m40.sh
#
# Point BAPE at the same tree (metadata_stitched + images_stitched) via MMNEEDLE_DATA_ROOT, and
# supply file_to_caption.json that covers val image basenames (from captions_val2014).

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# cd "${ROOT_DIR}"

VQA_ROOT="${VQA_ROOT:-/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data}"
COCO_VAL_DIR="${COCO_VAL_DIR:-${VQA_ROOT}/MSCOCO2014/val2014}"

# Keep val test artifacts separate from train defaults.
IMAGES_ROOT="${IMAGES_ROOT:-${VQA_ROOT}/MMNeedle/test_val2014/images_stitched}"
METADATA_ROOT="${METADATA_ROOT:-${VQA_ROOT}/MMNeedle/test_val2014/metadata_stitched}"

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-40}"
NUM_IMAGES="${NUM_IMAGES:-2000}"
NUM_SEQUENCES="${NUM_SEQUENCES:-40}"
POSITIVE_RATIO="${POSITIVE_RATIO:-0.5}"
SUBIMAGE_RES="${SUBIMAGE_RES:-256}"
SEED="${SEED:-0}"
# Space-separated n_grid values: 1=1x1, 2=2x2, 4=4x4 (default: all three stitching configs).
GRIDS="${GRIDS:-1 2 4}"
RUN_STITCH="${RUN_STITCH:-true}"

_stitching_label() {
  case "$1" in
    1) echo "1x1" ;;
    2) echo "2x2" ;;
    4) echo "4x4" ;;
    *) echo "${1}x${1}" ;;
  esac
}

if [[ ! -d "${COCO_VAL_DIR}" ]]; then
  echo "ERROR: COCO val2014 not found: ${COCO_VAL_DIR}" >&2
  exit 1
fi

# echo "ROOT_DIR=${ROOT_DIR}"
echo "COCO_VAL_DIR=${COCO_VAL_DIR}"
echo "IMAGES_ROOT=${IMAGES_ROOT}"
echo "METADATA_ROOT=${METADATA_ROOT}"
echo "SEQUENCE_LENGTH=${SEQUENCE_LENGTH} RUN_STITCH=${RUN_STITCH} GRIDS=${GRIDS}"
echo "Stitching configs: 1x1, 2x2, 4x4 (unless GRIDS overridden)"
echo "--------------------------------"

for g in ${GRIDS}; do
  cfg="$(_stitching_label "${g}")"
  echo "=== Stitching config ${cfg} (n_grid=${g}) ==="

  if [[ "${RUN_STITCH}" == "true" ]]; then
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python scripts/mmneedle_curate/sample_stitched_images.py \
      --coco_train_dir "${COCO_VAL_DIR}" \
      --n_grid "${g}" \
      --num_images "${NUM_IMAGES}" \
      --subimage_res "${SUBIMAGE_RES}" \
      --seed "${SEED}" \
      --images_root "${IMAGES_ROOT}" \
      --metadata_root "${METADATA_ROOT}"
  fi

  CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python scripts/mmneedle_curate/sample_single_needle_nxn.py \
    --n_grid "${g}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --num_sequences "${NUM_SEQUENCES}" \
    --positive_ratio "${POSITIVE_RATIO}" \
    --seed "${SEED}" \
    --images_root "${IMAGES_ROOT}" \
    --metadata_root "${METADATA_ROOT}" \
    --coco_train_dir "${COCO_VAL_DIR}"
done

echo "--------------------------------"
echo "Done. Annotation files (per grid):"
for g in ${GRIDS}; do
  echo "  ${METADATA_ROOT}/annotations_${SEQUENCE_LENGTH}_${g}_${g}.json"
done
echo ""
echo "Next: ensure file_to_caption.json for val targets; set MMNEEDLE_DATA_ROOT to a layout with"
echo "  metadata_stitched/ + images_stitched/ mirroring this tree (or symlink into MMNeedle/data)."
