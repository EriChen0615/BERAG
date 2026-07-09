#!/bin/bash
PROMPT_FILE="config/prompts/1101_norag_answer.txt"
IMG_BASEDIR="/rds/project/rds-iS0FZqj9lmg/wl356/infoseek/infoseek_images/images" # with respect to LlamaFactory
python src/curate/norag_answer.py \
    --rewrite_prompt_template_file $PROMPT_FILE \
    --dataset_name "Infoseek" \
    --img_basedir $IMG_BASEDIR \
    --seed 0 \
    --sample_size_train "-1" \
    --sample_size_eval 128 \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/Infoseek/norag_answer" 

