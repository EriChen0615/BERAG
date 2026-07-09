#/bin/bash

DROP_MAX_TOKENS=2048
# SAMPLE_SIZE=64000
# TOPK_DOCS=2
# SAMPLE_SIZE=64000
# SAMPLE_OFFSET=64000
# TOPK_DOCS=4
SAMPLE_SIZE=0
SAMPLE_OFFSET=0
TOPK_DOCS=16
SEED=42
# SEED=615
# TOPK_DOCS=5
OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/evqa/rag${TOPK_DOCS}-answer-ppl-size=${SAMPLE_SIZE}-offset=${SAMPLE_OFFSET}-max=${DROP_MAX_TOKENS}"
# OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/evqa/beft_K=2star_rand=Top${TOPK_DOCS}-prior=separate_prompt-size=${SAMPLE_SIZE}-max=${DROP_MAX_TOKENS}"

python src/curate/ragk_answer_ppl.py \
    --hf_dataset_path "outputs/jinghong_chen/EVQA-with-retrieval" \
    --passage_set_name "EVQA" \
    --topk_docs $TOPK_DOCS \
    --sample_size $SAMPLE_SIZE \
    --sample_offset $SAMPLE_OFFSET \
    --img_basedir "../.." \
    --output_dir $OUTPUT_DIR \
    --drop_max_tokens $DROP_MAX_TOKENS \
    --num_workers 8 \
    --seed $SEED \
    --batch_size 4096 \
    --ensure_gt_passage_in_topk 
    # --random_sample_1passage_from_topk
    # --add_separate_prompt_for_prior \