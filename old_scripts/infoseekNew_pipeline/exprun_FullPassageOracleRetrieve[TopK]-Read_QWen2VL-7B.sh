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


DATASET_NAME="InfoseekNew"
SPLIT="valid_m2kr"
MODEL_NAME="QWen2VL-7B-LoRA"
RETRIEVER_NAME="PreFLMR-L"
MODE="FullPassageOracleRetrieve[TopK]-Read"
CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}.jsonnet"
IMG_BASEDIR="/rds/project/rds-iS0FZqj9lmg/wl356/infoseek/infoseek_images/images"
TAKE_N=256
# EXP_NAME="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}"


# Define experiment name
declare -A exp1
exp1['custom_name']="pretrained"
exp1['model_path']="QWen/QWen2-VL-7B-Instruct"
exp1['retrieve_topk']=1
declare -A exp2
exp2['custom_name']="pretrained"
exp2['model_path']="QWen/QWen2-VL-7B-Instruct"
exp2['retrieve_topk']=5
declare -A exp3
exp3['custom_name']="pretrained"
exp3['model_path']="QWen/QWen2-VL-7B-Instruct"
exp3['retrieve_topk']=10

declare -A exp4
exp4['custom_name']="rag1-ft_ckpt-2000"
exp4['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag1_answer/checkpoint-2000"
exp4['retrieve_topk']=1
exp4['base_model_path']="QWen/QWen2-VL-7B-Instruct"

declare -A exp5
exp5['custom_name']="rag1-ft_ckpt-2000"
exp5['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag1_answer/checkpoint-2000"
exp5['retrieve_topk']=5
declare -A exp6
exp6['custom_name']="rag1-ft_ckpt-2000"
exp6['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag1_answer/checkpoint-2000"
exp6['retrieve_topk']=10
declare -A exp7
exp7['custom_name']="rag1-ft_ckpt-2000"
exp7['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag1_answer/checkpoint-2000"
exp7['retrieve_topk']=3

declare -A exp10
exp10['custom_name']="rag5-sft-ckpt718"
exp10['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-sft_max=4096/checkpoint-718"
exp10['retrieve_topk']=1
exp10['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp11
exp11['custom_name']="rag5-sft-ckpt718"
exp11['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-sft_max=4096/checkpoint-718"
exp11['retrieve_topk']=3
exp11['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp12
exp12['custom_name']="rag5-sft-ckpt718"
exp12['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-sft_max=4096/checkpoint-718"
exp12['retrieve_topk']=5
exp12['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp13
exp13['custom_name']="rag5-sft-ckpt718"
exp13['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-sft_max=4096/checkpoint-718"
exp13['retrieve_topk']=10
exp13['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"


# List of experiments
# experiments=("exp4" "exp5" "exp6" "exp7")
# experiments=("exp10" "exp11" "exp12" "exp13")
experiments=("exp1")

# Iterate over experiments
for exp in "${experiments[@]}"; do
  eval "model_path=\${${exp}['model_path']}"
  eval "custom_name=\${${exp}['custom_name']}"
  eval "retrieve_topk=\${${exp}['retrieve_topk']}"
  eval "base_model_path=\${${exp}['base_model_path']}"

  exp_name="${DATASET_NAME}/${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_RetrieveTopK=${retrieve_topk}_${MODEL_NAME}_${custom_name}_${RETRIEVER_NAME}"

  echo "model_path=${model_path}"
  echo "retrieve_topk=${retrieve_topk}"
  echo "base_model=${base_model_path}"
  echo ${exp_name}

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/run_vqa_pipeline.py \
    --dataset_name $DATASET_NAME \
    --exp_name $exp_name \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --take_n $TAKE_N \
    --retrieve_topk $retrieve_topk \
    --model_path $model_path \
    --base_model_path "$base_model_path" \
    --do_eval
    # --override \
done


