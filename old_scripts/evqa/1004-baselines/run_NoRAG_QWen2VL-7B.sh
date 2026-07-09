#!/bin/bash
#SBATCH -J ARAVQA-Qwen2VL-7B
#SBATCH -A BYRNE-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere

source scripts/hpc_activate_env.sh
which python

DATASET_NAME="EVQA"
SPLIT="test"
MODEL_NAME="QWen2VL-7B"
MODE="NoRAG"
CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${MODEL_NAME}.jsonnet"
IMG_BASEDIR="../vqa_data/KBVQA_data/EVQA/images/"
EXP_NAME="${DATASET_NAME}_${MODE}_${MODEL_NAME}"

python src/vqa_inference.py \
    --dataset_name $DATASET_NAME \
    --exp_name $EXP_NAME \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --debug


