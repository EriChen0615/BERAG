#!/bin/bash
PROMPT_FILE="config/prompts/1111_doc1_verify.txt"
IMG_BASEDIR="../.." # with respect to LlamaFactory
INPUT_DATASET="outputs/jinghong_chen/OKVQA-with-retrieval"
python src/curate/doc1_verify.py \
    --input_dataset $INPUT_DATASET \
    --prompt_template_file $PROMPT_FILE \
    --img_basedir $IMG_BASEDIR \
    --seed 0 \
    --sample_size_train 0 \
    --sample_size_eval 256 \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/okvqa/doc1-verify" 

