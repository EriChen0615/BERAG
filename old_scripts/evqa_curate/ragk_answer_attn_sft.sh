#/bin/bash

DROP_MAX_TOKENS=4096
SAMPLE_SIZE=64000
# SAMPLE_SIZE=1000
TOPK_DOCS=5
ATTN_CALIBRATION_SPAN=question_token
ATTN_SOURCE_SPAN=question
python src/curate/ragk_answer_attn_sft.py \
    --hf_dataset_path "outputs/jinghong_chen/EVQA-with-retrieval" \
    --passage_set_name "EVQA" \
    --mode "sft" \
    --topk_docs $TOPK_DOCS \
    --sample_size $SAMPLE_SIZE \
    --img_basedir "../.." \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/evqa/rag${TOPK_DOCS}-answer-attn_sft-source_span=${ATTN_SOURCE_SPAN}-cali_span=${ATTN_CALIBRATION_SPAN}-size=${SAMPLE_SIZE}-max=${DROP_MAX_TOKENS}" \
    --drop_max_tokens $DROP_MAX_TOKENS \
    --num_workers 8 \
    --batch_size 200 \
    --attn_calibration_span $ATTN_CALIBRATION_SPAN \
    --attn_source_span $ATTN_SOURCE_SPAN