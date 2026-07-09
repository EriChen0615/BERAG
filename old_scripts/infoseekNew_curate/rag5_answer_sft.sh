#!/bin/bash
PROMPT_FILE="config/prompts/1003_conventional_rag.txt"
# INPUT_CSV="data/jinghong_chen/EVQA_self-generated-pairs_n=8.csv"
INPUT_CSV="data/jinghong_chen/InfoseekNew_QWen2-VL-7B_self-generated-pairs_n=8.csv"
DROP_MAX_TOKENS=16384
python src/curate/rag5_answer.py \
    --input_csvfile $INPUT_CSV \
    --sample_size_train 0 \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/InfoseekNew/7B-rag5-answer-sft_max=${DROP_MAX_TOKENS}" \
    --mode "sft" \
    --report_token_length \
    --drop_max_tokens $DROP_MAX_TOKENS

