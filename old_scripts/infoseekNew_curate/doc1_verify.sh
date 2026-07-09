#!/bin/bash
PROMPT_FILE="config/prompts/1111_doc1_verify.txt"
INPUT_DATASET="outputs/0jingbiao_mei/InfoseekNew-train64000-with-retrieval"
IMG_BASEDIR="/rds/project/rds-iS0FZqj9lmg/wl356/infoseek/infoseek_images/images" 
export HF_HOME='../jm2245/HF_HOME'
python src/curate/doc1_verify.py \
    --input_dataset $INPUT_DATASET \
    --prompt_template_file $PROMPT_FILE \
    --seed 0 \
    --sample_size_train 0 \
    --sample_size_eval 256 \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/InfoseekNew/doc1-verify" \
    #--img_basedir $IMG_BASEDIR \

