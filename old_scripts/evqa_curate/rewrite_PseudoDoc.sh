#!/bin/bash
PROMPT_FILE="config/prompts/1021_question_expansion_with_doc_gen.txt"
IMG_BASEDIR="../../../vqa_data/KBVQA_data/EVQA/images" # with respect to LlamaFactory
python src/curate/rewrite_PseudoDoc.py \
    --rewrite_prompt_template_file $PROMPT_FILE \
    --img_basedir $IMG_BASEDIR \
    --seed 0 \
    --sample_size_train 50000 \
    --sample_size_eval 128 \
    --output_dir "third_party/LLaMAFactory/data/jinghong_chen/rewrite_pseudo_doc" 

