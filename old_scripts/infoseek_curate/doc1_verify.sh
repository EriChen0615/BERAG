#!/bin/bash
PROMPT_FILE="config/prompts/1111_doc1_verify.txt"
INPUT_DATASET="outputs/jinghong_chen/Infoseek-with-retrieval"
python src/curate/doc1_verify.py \
    --input_dataset $INPUT_DATASET \
    --prompt_template_file $PROMPT_FILE \
    --seed 0 \
    --sample_size_train 0 \
    --sample_size_eval 256 \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/Infoseek/doc1-verify" 
    # --img_basedir $IMG_BASEDIR \

