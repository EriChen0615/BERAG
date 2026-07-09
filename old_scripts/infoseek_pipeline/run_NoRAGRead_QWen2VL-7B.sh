#!/bin/bash
#SBATCH -J run_NoRAGRead_QWen2VL-7B
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
export WANDB_RUN_GROUP="HPC"

# source scripts/hpc_activate_env.sh
which python

DATASET_NAME="Infoseek"
SPLIT="test"
MODEL_NAME="QWen2VL-7B"
MODE="NoRAGRead"
CONFIG_FILE="config/${DATASET_NAME}_pipeline/${MODE}_${MODEL_NAME}.jsonnet"
IMG_BASEDIR="/rds/project/rds-iS0FZqj9lmg/wl356/infoseek/infoseek_images/images"
TAKE_N=256
# EXP_NAME="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_${MODEL_NAME}"
EXP_NAME="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_${MODEL_NAME}_pretrained"

python src/run_vqa_pipeline.py \
    --dataset_name $DATASET_NAME \
    --model_path "QWen/QWen2-VL-7B-Instruct" \
    --exp_name $EXP_NAME \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --take_n $TAKE_N \
    --do_eval
    # --override \
    # --debug


