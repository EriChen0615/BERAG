#!/bin/bash
#SBATCH -J Infoseek_sample_rag1_on_train_QWen2-7B_2ndpass
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
#SBATCH --array=0-7 # Adjust this based on the number of experiments

export WANDB_RUN_GROUP="HPC"

# Load environment
# source scripts/hpc_activate_env.sh
which python

# Define experiments
DATASET_NAME="InfoseekNew"
SPLIT="train"
MODEL_NAME="QWen2VL-7B"
RETRIEVER_NAME="PreFLMR-L"
MODE="2nd-SampleCacheRetrieve[TopK]-Read"
CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}.jsonnet"
CUSTOM_NAME="RAG-DPO-Top1"
IMG_BASEDIR="/rds/project/rds-iS0FZqj9lmg/wl356/infoseek/infoseek_images/images"
TAKE_N=64000
# TAKE_N=0

MODEL_PATH="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1"
SEEDS=(42 122423 615926 2313 2341 53 0 1214 1129)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
  # Define experiment name
exp_name="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_${MODEL_NAME}_${CUSTOM_NAME}_${RETRIEVER_NAME}_SEED=${SEED}"

echo model_name = $MODEL_NAME
echo model_path = $MODEL_PATH
echo custom_name = $CUSTOM_NAME
echo exp_name = $exp_name

# Run experiment
CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/run_vqa_pipeline.py \
    --dataset_name "$DATASET_NAME" \
    --exp_name "$exp_name" \
    --split "$SPLIT" \
    --config_file "$CONFIG_FILE" \
    --model_path "$MODEL_PATH" \
    --img_basedir "$IMG_BASEDIR" \
    --take_n "$TAKE_N" \
    --retrieve_topk 1 \
    --seed $SEED \
    --do_eval \
    --ds_seed 2025
    # --continue_expdir "20241225-00-${exp_name}"
