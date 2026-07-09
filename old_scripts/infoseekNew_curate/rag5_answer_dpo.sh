#!/bin/bash
PROMPT_FILE="config/prompts/1003_conventional_rag.txt"
# INPUT_CSV="data/jinghong_chen/EVQA_self-generated-pairs_n=8.csv"
# INPUT_CSV="data/jinghong_chen/InfoseekNew_QWen2-VL-7B_self-generated-pairs_n=8.csv"
# OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/InfoseekNew/7B-rag5-answer-dpo_max=${DROP_MAX_TOKENS}"
DROP_MAX_TOKENS=8196
# INPUT_CSV="data/jinghong_chen/InfoseekNew_QWen2-VL-7B_self-generated-pairs_n=8_rag1.csv"
# OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/InfoseekNew/7B-rag1-answer-dpo_max=${DROP_MAX_TOKENS}"

# INPUT_CSV="data/jinghong_chen/InfoseekNew_QWen2-VL-7B_self-generated-pairs_n=8_rag3.csv"
# OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/InfoseekNew/7B-rag3-answer-dpo_max=${DROP_MAX_TOKENS}"
INPUT_CSV="data/jinghong_chen/InfoseekNew_QWen2-VL-7B_self-generated-pairs_n=8_rag1_it2.csv"
OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/InfoseekNew/7B-rag1-answer-dpo_max=${DROP_MAX_TOKENS}_it2"
# DROP_MAX_TOKENS=16384
python src/curate/rag5_answer.py \
    --input_csvfile $INPUT_CSV \
    --sample_size_train 0 \
    --output_dir $OUTPUT_DIR \
    --mode "dpo" \
    --report_token_length \
    --drop_max_tokens $DROP_MAX_TOKENS

