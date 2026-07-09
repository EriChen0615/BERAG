#!/bin/bash
PROMPT_FILE="config/prompts/1101_norag_answer.txt"
IMG_BASEDIR="../../../vqa_data/KBVQA_data/ok-vqa/" # with respect to LlamaFactory
python src/curate/norag_answer.py \
    --rewrite_prompt_template_file $PROMPT_FILE \
    --dataset_name "OKVQA" \
    --img_basedir $IMG_BASEDIR \
    --seed 0 \
    --sample_size_train "-1" \
    --sample_size_eval 128 \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/okvqa/norag_answer" 

