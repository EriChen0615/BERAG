#!/bin/bash
#SBATCH -J CacheRetrieveRerank[TopK]-Read_QWen2VL-7B
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


DATASET_NAME="OKVQA"
SPLIT="valid"
MODEL_NAME="QWen2VL-7B"
RETRIEVER_NAME="PreFLMR-L"
MODE="CacheRetrieveRerank[TopK]-Read"
CONFIG_FILE="config/${DATASET_NAME}_pipeline/${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}.jsonnet"
IMG_BASEDIR="../vqa_data/KBVQA_data/ok-vqa/"
TAKE_N=0


# Define experiment name
declare -A exp1
exp1['custom_name']="ckpt-843"
exp1['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/norag_answer/checkpoint-843"
exp1['retrieve_topk']=1
declare -A exp2
exp2['custom_name']="ckpt-843"
exp2['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/norag_answer/checkpoint-843"
exp2['retrieve_topk']=5
declare -A exp3
exp3['custom_name']="ckpt-843"
exp3['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/norag_answer/checkpoint-843"
exp3['retrieve_topk']=10
declare -A exp4
exp4['custom_name']="rag1-ft_ckpt-1686"
exp4['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/rag1_answer/checkpoint-1686"
exp4['retrieve_topk']=1
declare -A exp5
exp5['custom_name']="rag1-ft_ckpt-1686"
exp5['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/rag1_answer/checkpoint-1686"
exp5['retrieve_topk']=5
declare -A exp6
exp6['custom_name']="rag1-ft_ckpt-1686"
exp6['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/rag1_answer/checkpoint-1686"
exp6['retrieve_topk']=10
declare -A exp7
exp7['custom_name']="rag1-ft_ckpt-1686"
exp7['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/rag1_answer/checkpoint-1686"
exp7['retrieve_topk']=3
# declare -A exp8
# exp8['custom_name']="rag5-ft_ckpt-234"
# exp8['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/rag5_answer-sft/checkpoint-234"
# exp8['retrieve_topk']=5
# declare -A exp9
# exp9['custom_name']="rag5-ft_ckpt-234"
# exp9['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/rag5_answer-sft/checkpoint-234"
# exp9['retrieve_topk']=10
# declare -A exp10
# exp10['custom_name']="rag5-ft_ckpt-234"
# exp10['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/rag5_answer-sft/checkpoint-234"
# exp10['retrieve_topk']=1
# declare -A exp11
# exp11['custom_name']="rag5-ft_ckpt-234"
# exp11['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/rag5_answer-sft/checkpoint-234"
# exp11['retrieve_topk']=3


# List of experiments
# experiments=("exp3" "exp2" "exp1")
experiments=("exp4" "exp5" "exp6" "exp7")
# experiments=("exp4")
# experiments=("exp8" "exp9" "exp10" "exp11")

# Iterate over experiments
for exp in "${experiments[@]}"; do
  eval "model_path=\${${exp}['model_path']}"
  eval "custom_name=\${${exp}['custom_name']}"
  eval "retrieve_topk=\${${exp}['retrieve_topk']}"
  exp_name="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_RetrieveTopK=${retrieve_topk}_${MODEL_NAME}_${custom_name}_${RETRIEVER_NAME}"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/run_vqa_pipeline.py \
    --dataset_name $DATASET_NAME \
    --exp_name ${exp_name} \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --take_n $TAKE_N \
    --retrieve_topk $retrieve_topk \
    --model_path $model_path \
    --do_eval 
    # --override \
done


