#!/bin/bash
#SBATCH -J CacheRetrieve[TopK]-Read_QWen2VL-2B
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
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
MODEL_NAME="QWen2VL-2B"
RETRIEVER_NAME="PreFLMR-L"
MODE="CacheRetrieve[TopK]-Read"
CONFIG_FILE="config/${DATASET_NAME}_pipeline/${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}.jsonnet"
IMG_BASEDIR="/rds/project/rds-iS0FZqj9lmg/wl356/infoseek/infoseek_images/images"
TAKE_N=256
# EXP_NAME="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}"


# Define experiment name
declare -A exp1
exp1['custom_name']="pretrained"
exp1['model_path']="QWen/QWen2-VL-2B-Instruct"
exp1['retrieve_topk']=1
declare -A exp2
exp2['custom_name']="pretrained"
exp2['model_path']="QWen/QWen2-VL-2B-Instruct"
exp2['retrieve_topk']=5
declare -A exp3
exp3['custom_name']="pretrained"
exp3['model_path']="QWen/QWen2-VL-2B-Instruct"
exp3['retrieve_topk']=10
declare -A exp4
exp4['custom_name']="norag-ft_ckpt-2000"
exp4['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/infoseek/norag_answer/checkpoint-2000"
exp4['retrieve_topk']=1
declare -A exp5
exp5['custom_name']="norag-ft_ckpt-2000"
exp5['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/infoseek/norag_answer/checkpoint-2000"
exp5['retrieve_topk']=5
declare -A exp6
exp6['custom_name']="norag-ft_ckpt-2000"
exp6['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/infoseek/norag_answer/checkpoint-2000"
exp6['retrieve_topk']=10
declare -A exp7
exp7['custom_name']="norag-ft_ckpt-2000"
exp7['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/infoseek/norag_answer/checkpoint-2000"
exp7['retrieve_topk']=3
declare -A exp8
exp8['custom_name']="rag1-ft_ckpt-4995"
exp8['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/infoseek/rag1_answer/checkpoint-4995"
exp8['retrieve_topk']=1
declare -A exp9
exp9['custom_name']="rag1-ft_ckpt-2000"
exp9['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/infoseek/rag1_answer/checkpoint-2000"
exp9['retrieve_topk']=5
declare -A exp10
exp10['custom_name']="rag1-ft_ckpt-2000"
exp10['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/infoseek/rag1_answer/checkpoint-2000"
exp10['retrieve_topk']=10
declare -A exp11
exp11['custom_name']="rag1-ft_ckpt-2000"
exp11['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/infoseek/rag1_answer/checkpoint-2000"
exp11['retrieve_topk']=3

# List of experiments
# experiments=("exp4" "exp5" "exp6" "exp7")
experiments=("exp9" "exp10" "exp11")

# Iterate over experiments
for exp in "${experiments[@]}"; do
  eval "model_path=\${${exp}['model_path']}"
  eval "custom_name=\${${exp}['custom_name']}"
  eval "retrieve_topk=\${${exp}['retrieve_topk']}"
  exp_name="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_RetrieveTopK=${retrieve_topk}_${MODEL_NAME}_${custom_name}_${RETRIEVER_NAME}"
  echo "retrieve_topk=${retrieve_topk}"
  echo ${exp_name}
  echo ${model_path}

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/run_vqa_pipeline.py \
    --dataset_name $DATASET_NAME \
    --exp_name $exp_name \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --take_n $TAKE_N \
    --retrieve_topk $retrieve_topk \
    --do_eval \
    --model_path $model_path
    # --override \
done


