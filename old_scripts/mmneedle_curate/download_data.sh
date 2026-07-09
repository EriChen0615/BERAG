#!/usr/bin/env bash
set -euo pipefail

# Download:
# 1) MS COCO 2014 train/val images (+ optional annotations)
# 2) MMNeedle dataset from Hugging Face
#
# Target directories:
#   ../vqa_data/MSCOCO2014
#   ../vqa_data/MMNeedle
#
# Run from anywhere:
#   bash scripts/mmneedle_curate/download_data.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# Activate project env if available (optional but recommended)
# if [[ -f "scripts/hpc_activate_env_py310_infer.sh" ]]; then
#   # shellcheck disable=SC1091
#   source "scripts/hpc_activate_env_py310_infer.sh"
# fi

MSCOCO_DIR="../vqa_data/MSCOCO2014"
MMNEEDLE_DIR="../vqa_data/MMNeedle"

mkdir -p "${MSCOCO_DIR}" "${MMNEEDLE_DIR}"

echo "[1/4] Downloading MS COCO 2014 train/val zips..."
cd "${MSCOCO_DIR}"
wget -c "http://images.cocodataset.org/zips/train2014.zip"
wget -c "http://images.cocodataset.org/zips/val2014.zip"

echo "[2/4] Downloading MS COCO 2014 annotations (optional but useful)..."
wget -c "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

echo "[3/4] Unzipping MS COCO files..."
unzip -q -o "train2014.zip"
unzip -q -o "val2014.zip"
unzip -q -o "annotations_trainval2014.zip"

echo "[4/4] Downloading MMNeedle from Hugging Face..."
cd "${ROOT_DIR}"
huggingface-cli download "Wang-ML-Lab/MMNeedle" \
  --repo-type dataset \
  --local-dir "${MMNEEDLE_DIR}" \
  --local-dir-use-symlinks False

echo "Done."
echo "MS COCO path:  ${MSCOCO_DIR}"
echo "MMNeedle path: ${MMNEEDLE_DIR}"

