#!/bin/bash
#SBATCH -J OKVQA_CacheRetrieve[TopK]-Read_QWen2VL-2B
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
MODEL_NAME="QWen2VL-2B"
RETRIEVER_NAME="PreFLMR-L"
MODE="CacheRetrieve[TopK]-Read"
CONFIG_FILE="config/${DATASET_NAME}_pipeline/${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}.jsonnet"
IMG_BASEDIR="../vqa_data/KBVQA_data/ok-vqa/"
TAKE_N=0


# Define experiment name
declare -A exp1
exp1['custom_name']="ckpt-843"
exp1['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/norag_answer/checkpoint-843"
exp1['retrieve_topk']=1
declare -A exp2
exp2['custom_name']="ckpt-843"
exp2['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/norag_answer/checkpoint-843"
exp2['retrieve_topk']=5
declare -A exp3
exp3['custom_name']="ckpt-843"
exp3['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/norag_answer/checkpoint-843"
exp3['retrieve_topk']=10
declare -A exp4
exp4['custom_name']="rag1-ft_ckpt-1686"
exp4['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag1_answer/checkpoint-1686"
exp4['retrieve_topk']=1
declare -A exp5
exp5['custom_name']="rag1-ft_ckpt-1686"
exp5['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag1_answer/checkpoint-1686"
exp5['retrieve_topk']=5
declare -A exp6
exp6['custom_name']="rag1-ft_ckpt-1686"
exp6['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag1_answer/checkpoint-1686"
exp6['retrieve_topk']=10
declare -A exp7
exp7['custom_name']="rag1-ft_ckpt-1686"
exp7['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag1_answer/checkpoint-1686"
exp7['retrieve_topk']=3
declare -A exp8
exp8['custom_name']="rag5-ft_ckpt-234"
exp8['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag5_answer-sft/checkpoint-234"
exp8['retrieve_topk']=5
exp8['base_model_path']="data/jinghong_chen/Qwen2-VL-2B-Instruct_OKVQA-RAG1_LoRA-SFT"
declare -A exp9
exp9['custom_name']="rag5-ft_ckpt-234"
exp9['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag5_answer-sft/checkpoint-234"
exp9['retrieve_topk']=10
exp9['base_model_path']="data/jinghong_chen/Qwen2-VL-2B-Instruct_OKVQA-RAG1_LoRA-SFT"
declare -A exp10
exp10['custom_name']="rag5-ft_ckpt-234"
exp10['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag5_answer-sft/checkpoint-234"
exp10['retrieve_topk']=1
exp10['base_model_path']="data/jinghong_chen/Qwen2-VL-2B-Instruct_OKVQA-RAG1_LoRA-SFT"
declare -A exp11
exp11['custom_name']="rag5-ft_ckpt-234"
exp11['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag5_answer-sft/checkpoint-234"
exp11['retrieve_topk']=3
exp11['base_model_path']="data/jinghong_chen/Qwen2-VL-2B-Instruct_OKVQA-RAG1_LoRA-SFT"
declare -A exp12
exp12['custom_name']="rag5-dpo_beta=0.1_ckpt-234"
exp12['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag5_answer-dpo_beta=0.1/checkpoint-234"
exp12['retrieve_topk']=5
exp12['base_model_path']="data/jinghong_chen/Qwen2-VL-2B-Instruct_OKVQA-RAG5_LoRA-SFT"
declare -A exp13
exp13['custom_name']="rag5-dpo_beta=0.1_ckpt-234"
exp13['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag5_answer-dpo_beta=0.1/checkpoint-234"
exp13['retrieve_topk']=1
exp13['base_model_path']="data/jinghong_chen/Qwen2-VL-2B-Instruct_OKVQA-RAG5_LoRA-SFT"
declare -A exp14
exp14['custom_name']="rag5-dpo_beta=0.1_ckpt-234"
exp14['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag5_answer-dpo_beta=0.1/checkpoint-234"
exp14['retrieve_topk']=3
exp14['base_model_path']="data/jinghong_chen/Qwen2-VL-2B-Instruct_OKVQA-RAG5_LoRA-SFT"
declare -A exp15
exp14['custom_name']="rag5-dpo_beta=0.1_ckpt-234"
exp14['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/rag5_answer-dpo_beta=0.1/checkpoint-234"
exp14['retrieve_topk']=10
exp14['base_model_path']="data/jinghong_chen/Qwen2-VL-2B-Instruct_OKVQA-RAG5_LoRA-SFT"



# List of experiments
# experiments=("exp3" "exp2" "exp1")
# experiments=("exp4" "exp5" "exp6")
# experiments=("exp8" "exp9" "exp10" "exp11")
# experiments=("exp12" "exp13" "exp14" "exp15")
experiments=("exp12")

# Iterate over experiments
for exp in "${experiments[@]}"; do
  eval "model_path=\${${exp}['model_path']}"
  eval "custom_name=\${${exp}['custom_name']}"
  eval "retrieve_topk=\${${exp}['retrieve_topk']}"
  eval "base_model_path=\${${exp}['base_model_path']}"
  exp_name="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_RetrieveTopK=${retrieve_topk}_${MODEL_NAME}_${custom_name}_${RETRIEVER_NAME}"

  echo $model_path
  echo $custom_name
  echo $exp_name
  echo $base_model_path

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/run_vqa_pipeline.py \
    --dataset_name $DATASET_NAME \
    --exp_name ${exp_name} \
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


