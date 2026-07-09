#/bin/bash

DROP_MAX_TOKENS=2048
# SAMPLE_SIZE=64000
# TOPK_DOCS=2
# SAMPLE_SIZE=64000
# SAMPLE_OFFSET=64000
# TOPK_DOCS=4
SAMPLE_SIZE=0
SAMPLE_OFFSET=0
TOPK_DOCS=4
SEED=42
# SEED=615
# TOPK_DOCS=5
SPLIT="train"
OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/slidevqa/rag${TOPK_DOCS}-slidevqa-sft-size=${SAMPLE_SIZE}-offset=${SAMPLE_OFFSET}-max=${DROP_MAX_TOKENS}"
# OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/slidevqa/sft_K=2star_rand=Top${TOPK_DOCS}-prior=separate_prompt-size=${SAMPLE_SIZE}-max=${DROP_MAX_TOKENS}"

IMG_BASE_DIR="../../shared_space/vqa_data/KBVQA_data/SlideVQA"

python src/curate/ragk_slidevqa.py \
    --hf_dataset_path "NTT-hil-insight/SlideVQA" \
    --mode "sft" \
    --topk_docs $TOPK_DOCS \
    --sample_size $SAMPLE_SIZE \
    --sample_offset $SAMPLE_OFFSET \
    --img_basedir "$IMG_BASE_DIR" \
    --output_dir $OUTPUT_DIR \
    --drop_max_tokens $DROP_MAX_TOKENS \
    --num_workers 4 \
    --seed $SEED \
    --batch_size 128 \
    --split $SPLIT \
    --ensure_gt_passage_in_topk \
    --skip_image_path_exist_check 
    # --random_sample_1passage_from_topk
    # --add_separate_prompt_for_prior

