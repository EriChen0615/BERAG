#!/bin/bash
#SBATCH -J run_NoRAGRead_QWen2VL-2B
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
## Uncomment this to prevent the job from being requeued (e.g. if
## interrupted by node failure or system downtime):
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

# Define arrays for MODEL_PATH and CUSTOM_NAME
MODEL_PATHS=(
    "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/norag_answer/checkpoint-2000"
    # "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/norag_answer/checkpoint-5000"
    # "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/norag_answer/checkpoint-10000"
    # "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/norag_answer/checkpoint-15000"
)
CUSTOM_NAMES=(
    "ckpt2000"
    # "ckpt5000"
    # "ckpt10000"
    # "ckpt15000"
)

# Loop through checkpoints
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[i]}"
    CUSTOM_NAME="${CUSTOM_NAMES[i]}"
    EXP_NAME="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_${MODEL_NAME}_${CUSTOM_NAME}"
    
    python src/run_vqa_pipeline.py \
        --dataset_name $DATASET_NAME \
        --exp_name $EXP_NAME \
        --split $SPLIT \
        --config_file $CONFIG_FILE \
        --img_basedir $IMG_BASEDIR \
        --take_n $TAKE_N \
        --model_path $MODEL_PATH \
        --do_eval
        # --override \
        # --debug
done
