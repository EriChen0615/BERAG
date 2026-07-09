#!/bin/bash

DATASET_NAME="OKVQA"
SPLIT="train"
RETRIEVER_NAME="PreFLMR-L-train"
MODE="Retrieval"
CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${RETRIEVER_NAME}.jsonnet"
IMG_BASEDIR="../vqa_data/KBVQA_data/ok-vqa/"
EXP_NAME="${DATASET_NAME}_${MODE}"
RETRIEVAL_DS_SAVEPATH="outputs/jinghong_chen/OKVQA-with-retrieval"

# export WANDB_RUN_GROUP="3090"

python src/retriever_inference.py \
    --dataset_name $DATASET_NAME \
    --exp_name $EXP_NAME \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --save_retrieved_ds_to $RETRIEVAL_DS_SAVEPATH \
    --do_sanity_check 
    # --debug_cases 1 10 20 30 40 50 60 70 80 90 100 1000 1100 1200 1500 2000 2500 2700 2700 2750 2800 2900 3000 3200 3500

