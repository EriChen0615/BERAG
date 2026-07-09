#!/bin/bash

DATASET_NAME="EVQA"
SPLIT="test"
RETRIEVER_NAME="PreFLMR-L"
MODE="Retrieval"
CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${RETRIEVER_NAME}.jsonnet"
IMG_BASEDIR="/mnt/g/Datasets/EVQA/images/archived/eval"
EXP_NAME="${DATASET_NAME}_${MODE}"

export WANDB_RUN_GROUP="3090"

python src/retriever_inference.py \
    --dataset_name $DATASET_NAME \
    --exp_name $EXP_NAME \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR 

