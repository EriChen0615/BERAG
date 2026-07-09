#!/bin/bash
#SBATCH -J sample_rag5_on_train_QWen2-2B
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
DATASET_NAME="EVQA"
SPLIT="train"
MODEL_NAME="QWen2VL-2B-LoRA"
RETRIEVER_NAME="PreFLMR-L-train"
MODE="SampleCacheRetrieve[Top5]-Read"
CONFIG_FILE="config/${DATASET_NAME}_pipeline/${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}.jsonnet"
CUSTOM_NAME="RAG-Top1"
IMG_BASEDIR="../vqa_data/KBVQA_data/EVQA/images/"
# TAKE_N=160000
TAKE_N=0

MODEL_PATH="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/rag1_answer/checkpoint-5000"
SEEDS=(42 1129 122423 615926 2313 2341 53 0 1214)

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
    --seed $SEED \
    --do_eval
