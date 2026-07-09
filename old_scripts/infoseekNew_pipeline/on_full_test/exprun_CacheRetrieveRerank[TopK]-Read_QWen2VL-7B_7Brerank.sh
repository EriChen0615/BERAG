#!/bin/bash
#SBATCH -J InfoseekNew_CacheRetrieveRerank[TopK]-Read_QWen2VL-7B
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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
RETRIEVER_NAME="PreFLMR-L_7BRerank"
MODE="CacheRetrieveRerank[TopK]-Read"
CONFIG_FILE="config/${DATASET_NAME}/full_test/${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}.jsonnet"
IMG_BASEDIR="/rds/project/rds-iS0FZqj9lmg/wl356/infoseek/infoseek_images/images"
TAKE_N=0

# Define experiment name
declare -A exp1
exp1['custom_name']="rag1_answer_ckpt2000"
exp1['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer/checkpoint-2000"
exp1['retrieve_topk']=1
exp1['base_model_path']="QWen/QWen2-VL-7B-Instruct"

declare -A exp2
exp2['custom_name']="rag1_answer_ckpt2000"
exp2['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer/checkpoint-2000"
exp2['retrieve_topk']=3
exp2['base_model_path']="QWen/QWen2-VL-7B-Instruct"

declare -A exp3
exp3['custom_name']="rag1_answer_ckpt2000"
exp3['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer/checkpoint-2000"
exp3['retrieve_topk']=5
exp3['base_model_path']="QWen/QWen2-VL-7B-Instruct"

declare -A exp4
exp4['custom_name']="rag1_answer_ckpt2000"
exp4['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer/checkpoint-2000"
exp4['retrieve_topk']=10
exp4['base_model_path']="QWen/QWen2-VL-7B-Instruct"

declare -A exp5
exp5['custom_name']="rag5_answer_sft_ckpt500"
exp5['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag5_answer-sft_max=16384=1e-5/checkpoint-500"
exp5['retrieve_topk']=1
exp5['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp6
exp6['custom_name']="rag5_answer_sft_ckpt500"
exp6['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag5_answer-sft_max=16384=1e-5/checkpoint-500"
exp6['retrieve_topk']=3
exp6['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp7
exp7['custom_name']="rag5_answer_sft_ckpt500"
exp7['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag5_answer-sft_max=16384=1e-5/checkpoint-500"
exp7['retrieve_topk']=5
exp7['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp8
exp8['custom_name']="rag5_answer_sft_ckpt500"
exp8['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag5_answer-sft_max=16384=1e-5/checkpoint-500"
exp8['retrieve_topk']=10
exp8['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp9
exp9['custom_name']="rag5_answer_dpo_beta=2.0_ckpt932"
exp9['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag5_answer-dpo_max=8196_beta=2.0/checkpoint-932"
exp9['retrieve_topk']=1
exp9['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG5_LoRA-SFT"

declare -A exp10
exp10['custom_name']="rag5_answer_dpo_beta=2.0_ckpt932"
exp10['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag5_answer-dpo_max=8196_beta=2.0/checkpoint-932"
exp10['retrieve_topk']=3
exp10['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG5_LoRA-SFT"

declare -A exp11
exp11['custom_name']="rag5_answer_dpo_beta=2.0_ckpt932"
exp11['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag5_answer-dpo_max=8196_beta=2.0/checkpoint-932"
exp11['retrieve_topk']=5
exp11['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG5_LoRA-SFT"

# declare -A exp12
# exp12['custom_name']="rag5_answer_dpo_beta=2.0_ckpt932"
# exp12['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag5_answer-dpo_max=8196_beta=2.0/checkpoint-932"
# exp12['retrieve_topk']=10
# exp12['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG5_LoRA-SFT"

declare -A exp12
exp12['custom_name']="rag1_answer_dpo_beta=2.0_ckpt1295"
exp12['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_beta=2.0/checkpoint-1295"
exp12['retrieve_topk']=1
exp12['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp13
exp13['custom_name']="rag1_answer_dpo_beta=2.0_ckpt1295"
exp13['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_beta=2.0/checkpoint-1295"
exp13['retrieve_topk']=3
exp13['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp14
exp14['custom_name']="rag1_answer_dpo_beta=2.0_ckpt1295"
exp14['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_beta=2.0/checkpoint-1295"
exp14['retrieve_topk']=5
exp14['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp15
exp15['custom_name']="rag1_answer_dpo_beta=2.0_ckpt1295"
exp15['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_beta=2.0/checkpoint-1295"
exp15['retrieve_topk']=10
exp15['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp16
exp16['custom_name']="rag3_answer_dpo_beta=2.0_ckpt1242"
exp16['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag3_answer-dpo_max=8196_beta=2.0/checkpoint-1242"
exp16['retrieve_topk']=1
exp16['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp17
exp17['custom_name']="rag3_answer_dpo_beta=2.0_ckpt1242"
exp17['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag3_answer-dpo_max=8196_beta=2.0/checkpoint-1242"
exp17['retrieve_topk']=3
exp17['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp18
exp18['custom_name']="rag3_answer_dpo_beta=2.0_ckpt1242"
exp18['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag3_answer-dpo_max=8196_beta=2.0/checkpoint-1242"
exp18['retrieve_topk']=5
exp18['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp19
exp19['custom_name']="rag3_answer_dpo_beta=2.0_ckpt1242"
exp19['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag3_answer-dpo_max=8196_beta=2.0/checkpoint-1242"
exp19['retrieve_topk']=10
exp19['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

declare -A exp23
exp23['custom_name']="rag1_answer_dpo_it2_beta=1.5_ckpt1550"
exp23['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_it2_beta=1.5/checkpoint-1550"
exp23['retrieve_topk']=1
exp23['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1"

declare -A exp24
exp24['custom_name']="rag1_answer_dpo_it2_beta=1.5_ckpt1550"
exp24['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_it2_beta=1.5/checkpoint-1550"
exp24['retrieve_topk']=3
exp24['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1"

declare -A exp25
exp25['custom_name']="rag1_answer_dpo_it2_beta=1.5_ckpt1550"
exp25['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_it2_beta=1.5/checkpoint-1550"
exp25['retrieve_topk']=5
exp25['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1"

declare -A exp26
exp26['custom_name']="rag1_answer_dpo_it2_beta=1.5_ckpt1550"
exp26['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_it2_beta=1.5/checkpoint-1550"
exp26['retrieve_topk']=10
exp26['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1"

declare -A exp27
exp27['custom_name']="rag1_answer_dpo_it2_beta=2.0_ckpt1550"
exp27['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_it2_beta=2.0/checkpoint-1550"
exp27['retrieve_topk']=1
exp27['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1"

declare -A exp28
exp28['custom_name']="rag1_answer_dpo_it2_beta=2.0_ckpt1550"
exp28['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_it2_beta=2.0/checkpoint-1550"
exp28['retrieve_topk']=3
exp28['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1"

declare -A exp29
exp29['custom_name']="rag1_answer_dpo_it2_beta=2.0_ckpt1550"
exp29['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_it2_beta=2.0/checkpoint-1550"
exp29['retrieve_topk']=5
exp29['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1"

declare -A exp30
exp30['custom_name']="rag1_answer_dpo_it2_beta=2.0_ckpt1550"
exp30['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_it2_beta=2.0/checkpoint-1550"
exp30['retrieve_topk']=10
exp30['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-DPO_it1"

# List of experiments
# experiments=("exp1" "exp2" "exp3" "exp4" "exp5")
# experiments=("exp10" "exp11")
# experiments=("exp5" "exp7" "exp8")
# experiments=("exp1" "exp2" "exp3" "exp4" "exp5" "exp6" "exp7" "exp8" "exp9" "exp10" "exp11" "exp12")
# experiments=("exp12" "exp13" "exp14" "exp15")
# experiments=("exp16" "exp17" "exp18" "exp19")
# experiments=("exp1" "exp2" "exp3" "exp4" "exp12" "exp13" "exp14" "exp15" "exp16" "exp17" "exp18" "exp19")
# experiments=("exp15" "exp16" "exp17" "exp18" "exp19")
experiments=("exp23" "exp24" "exp25" "exp26" "exp27" "exp28" "exp29" "exp30")



# Iterate over experiments
for exp in "${experiments[@]}"; do
  eval "model_path=\${${exp}['model_path']}"
  eval "custom_name=\${${exp}['custom_name']}"
  eval "retrieve_topk=\${${exp}['retrieve_topk']}"
  eval "base_model_path=\${${exp}['base_model_path']}"

  exp_name="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_RetrieveTopK=${retrieve_topk}_${MODEL_NAME}_${custom_name}_${RETRIEVER_NAME}"

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
    # --continue_expdir "outputs/20250305-07-InfoseekNew_valid_m2kr-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B-LoRA_rag1_answer_dpo_beta=2.0_ckpt1295_PreFLMR-L_7BRerank"
    # --override \
done
