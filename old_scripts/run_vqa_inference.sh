#!/bin/bash

DATASET_NAME="EVQA"
SPLIT="test"
EXP_NAME="test_run"
CONFIG_FILE="config/config.jsonnet"
IMG_BASEDIR="../vqa_data/KBVQA_data/EVQA/images/"

python src/vqa_inference.py \
    --dataset_name $DATASET_NAME \
    --exp_name $EXP_NAME \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --debug


