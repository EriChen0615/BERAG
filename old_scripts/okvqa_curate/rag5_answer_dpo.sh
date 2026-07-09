#!/bin/bash
PROMPT_FILE="config/prompts/1003_conventional_rag.txt"
INPUT_CSV="data/jinghong_chen/OKVQA_self-generated-pairs_n=8.csv"
DROP_MAX_TOKENS=4096
python src/curate/rag5_answer.py \
    --input_csvfile $INPUT_CSV \
    --sample_size_train 0 \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/okvqa/rag5-answer-dpo_max=${DROP_MAX_TOKENS}" \
    --mode "dpo" \
    --report_token_length \
    --drop_max_tokens $DROP_MAX_TOKENS

