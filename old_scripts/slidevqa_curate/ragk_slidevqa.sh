#!/bin/bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake
#! Notes for icelake partition:
#! - Each node has 76 CPUs (cores)
#! - Each CPU is allocated 3380 MiB (~3.3 GB) of memory by default
#! - For 16 CPUs: default memory = 16 * 3380 MiB ≈ 54 GB
#! - We request 80G to have extra headroom for image processing

DROP_MAX_TOKENS=2048
# SAMPLE_SIZE=64000
# TOPK_DOCS=2
# SAMPLE_SIZE=64000
# SAMPLE_OFFSET=64000
# TOPK_DOCS=4
SAMPLE_SIZE=0
SAMPLE_OFFSET=0
TOPK_DOCS=4  # Required for data augmentation (must be 4)
SEED=42
# SEED=615
# TOPK_DOCS=5
SPLIT="train"
# OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/slidevqa/rag${TOPK_DOCS}-slidevqa-beft-size=${SAMPLE_SIZE}-offset=${SAMPLE_OFFSET}-max=${DROP_MAX_TOKENS}"
OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/slidevqa/rag${TOPK_DOCS}-slidevqa-beft-size=${SAMPLE_SIZE}-offset=${SAMPLE_OFFSET}-max=${DROP_MAX_TOKENS}-da=subdivide4"
# OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/slidevqa/beft_K=2star_rand=Top${TOPK_DOCS}-prior=separate_prompt-size=${SAMPLE_SIZE}-max=${DROP_MAX_TOKENS}"

IMG_BASE_DIR="../../shared_space/vqa_data/KBVQA_data/SlideVQA"

python src/curate/ragk_slidevqa.py \
    --hf_dataset_path "NTT-hil-insight/SlideVQA" \
    --mode "beft" \
    --topk_docs $TOPK_DOCS \
    --sample_size $SAMPLE_SIZE \
    --sample_offset $SAMPLE_OFFSET \
    --img_basedir "$IMG_BASE_DIR" \
    --output_dir $OUTPUT_DIR \
    --drop_max_tokens $DROP_MAX_TOKENS \
    --num_workers 16 \
    --seed $SEED \
    --batch_size 64 \
    --split $SPLIT \
    --ensure_gt_passage_in_topk \
    --skip_image_path_exist_check \
    --enable_data_augmentation
    # --random_sample_1passage_from_topk
    # --add_separate_prompt_for_prior
    # Data augmentation creates separate training instances by subdividing GT slides:
    # - 2 GT slides: split each by width (left-right) → 4 sub-images
    # - 1 GT slide: split into 2x2 grid → 4 sub-images
    # - >2 GT slides: no augmentation
    # Augmented instances have empty gt_passage_idx (no GT annotation)
