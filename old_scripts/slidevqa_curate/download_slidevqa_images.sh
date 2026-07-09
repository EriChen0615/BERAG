#!/bin/bash

# Script to download SlideVQA images from HuggingFace and update dataset with local paths
#
# To access gated datasets, you need to authenticate with HuggingFace:
# Method 1: Set environment variable (recommended)
#   export HF_TOKEN="your_huggingface_token_here"
# Method 2: Use --hf_token argument (see below)
# Method 3: Use huggingface-cli login (if installed)
#   huggingface-cli login
#
# To get your token:
# 1. Go to https://huggingface.co/settings/tokens
# 2. Create a new token (read access is sufficient)
# 3. Copy the token and use it as above

HF_DATASET_PATH="NTT-hil-insight/SlideVQA"
OUTPUT_DIR="outputs/jinghong_chen/SlideVQA-with-local-images"
IMG_BASE_DIR="../../shared_space/vqa_data/KBVQA_data/SlideVQA"
# SPLIT="train"  # Uncomment to process specific split, or leave None for all splits
SPLIT="train"
# HF_TOKEN=""  # Uncomment and set your token here, or use HF_TOKEN environment variable
BATCH_SIZE=128 # Batch size for batched processing
NUM_PROC=16  # Number of processes for parallel processing

python src/curate/download_slidevqa_images.py \
    --hf_dataset_path "$HF_DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --img_base_dir "$IMG_BASE_DIR" \
    ${SPLIT:+--split "$SPLIT"} \
    --batch_size $BATCH_SIZE \
    --num_proc $NUM_PROC \
    ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
    --verbose \
    --check_corruption

