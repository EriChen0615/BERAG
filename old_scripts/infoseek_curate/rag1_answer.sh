#!/bin/bash
PROMPT_FILE="config/prompts/1003_conventional_rag.txt"
# IMG_BASEDIR="" # with respect to LlamaFactory for Infoseek this is an absolute path
INPUT_DATASET="outputs/jinghong_chen/Infoseek-with-retrieval"
python src/curate/rag1_answer.py \
    --input_dataset $INPUT_DATASET \
    --prompt_template_file $PROMPT_FILE \
    --seed 0 \
    --sample_size_train 0 \
    --sample_size_eval 128 \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/Infoseek/rag1-answer" 
    # --img_basedir $IMG_BASEDIR \

