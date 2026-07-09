#!/bin/bash
#SBATCH -J run_NoRAGRead_QWen2VL-2B
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

DATASET_NAME="OKVQA"
SPLIT="valid"
MODEL_NAME="QWen2VL-2B"
MODE="NoRAGRead"
CONFIG_FILE="config/${DATASET_NAME}_pipeline/${MODE}_${MODEL_NAME}.jsonnet"
MODEL_PATH="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/norag_answer/checkpoint-843"
IMG_BASEDIR="../vqa_data/KBVQA_data/ok-vqa/"
TAKE_N=0
EXP_NAME="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_${MODEL_NAME}"

python src/run_vqa_pipeline.py \
    --dataset_name $DATASET_NAME \
    --exp_name $EXP_NAME \
    --model_path $MODEL_PATH \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --take_n $TAKE_N \
    --do_eval \
    --continue_expdir "outputs/20241210-13-OKVQA_valid-0_NoRAGRead_QWen2VL-2B/"
    # --override \
    # --debug


