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
MODEL_NAME="QWen2VL-7B"
RETRIEVER_NAME="PreFLMR-L"
MODE="CacheRetrieveRerank[TopK]-Read"
CONFIG_FILE="config/${DATASET_NAME}_pipeline/${MODE}_${MODEL_NAME}_${RETRIEVER_NAME}.jsonnet"
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
# declare -A exp4
# exp4['custom_name']="norag-ft_ckpt-2000"
# exp4['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/norag_answer/checkpoint-2000"
# exp4['retrieve_topk']=1
# declare -A exp5
# exp5['custom_name']="norag-ft_ckpt-2000"
# exp5['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/norag_answer/checkpoint-2000"
# exp5['retrieve_topk']=5
# declare -A exp6
# exp6['custom_name']="norag-ft_ckpt-2000"
# exp6['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/norag_answer/checkpoint-2000"
# exp6['retrieve_topk']=10
# declare -A exp7
# exp7['custom_name']="norag-ft_ckpt-2000"
# exp7['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/norag_answer/checkpoint-2000"
# exp7['retrieve_topk']=3
declare -A exp4
exp4['custom_name']="rag1-ft_ckpt-2000"
exp4['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag1_answer/checkpoint-2000"
exp4['retrieve_topk']=1
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

declare -A exp8
exp8['custom_name']="rag5-dpo_ckpt-500"
exp8['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.1/checkpoint-500"
exp8['retrieve_topk']=3
exp8['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp9
exp9['custom_name']="rag5-sft-ckpt500"
exp9['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-sft_max=4096/checkpoint-500"
exp9['retrieve_topk']=3
exp9['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

# declare -A exp9
# exp9['custom_name']="rag5-dpo_ckpt-2000"
# exp9['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.1/checkpoint-2000"
# exp9['retrieve_topk']=3
# exp9['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp10
exp10['custom_name']="rag5-sft-ckpt718"
exp10['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-sft_max=4096/checkpoint-718"
# exp10['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.1/checkpoint-2000"
exp10['retrieve_topk']=5
exp10['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp11
exp11['custom_name']="rag5-sft-ckpt718"
exp11['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-sft_max=4096/checkpoint-718"
exp11['retrieve_topk']=10
exp11['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

# declare -A exp9
# exp9['custom_name']="rag5-dpo_ckpt-2000"
# exp9['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.1/checkpoint-2000"
# exp9['retrieve_topk']=3
# exp9['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp12
exp12['custom_name']="rag5-dpo-beta=0.3-ckpt718"
exp12['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-718"
exp12['retrieve_topk']=1
exp12['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp13
exp13['custom_name']="rag5-dpo-beta=0.3-ckpt718"
exp13['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-718"
exp13['retrieve_topk']=3
exp13['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp14
exp14['custom_name']="rag5-dpo-beta=0.3-ckpt718"
exp14['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-718"
exp14['retrieve_topk']=5
exp14['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp15
exp15['custom_name']="rag5-dpo-beta=0.3-ckpt718"
exp15['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-718"
exp15['retrieve_topk']=10
exp15['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp16
exp16['custom_name']="rag5-dpo-beta=0.5-ckpt718"
exp16['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.5/checkpoint-718"
exp16['retrieve_topk']=1
exp16['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp17
exp17['custom_name']="rag5-dpo-beta=0.5-ckpt718"
exp17['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.5/checkpoint-718"
exp17['retrieve_topk']=3
exp17['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp18
exp18['custom_name']="rag5-dpo-beta=0.5-ckpt718"
exp18['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.5/checkpoint-718"
exp18['retrieve_topk']=5
exp18['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp19
exp19['custom_name']="rag5-dpo-beta=0.5-ckpt718"
exp19['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.5/checkpoint-718"
exp19['retrieve_topk']=10
exp19['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

# List of experiments
# experiments=("exp16" "exp17" "exp18" "exp19")

declare -A exp20
exp20['custom_name']="rag5-dpo-beta=0.7-ckpt718"
exp20['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.7/checkpoint-718"
exp20['retrieve_topk']=1
exp20['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp21
exp21['custom_name']="rag5-dpo-beta=0.7-ckpt718"
exp21['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.7/checkpoint-718"
exp21['retrieve_topk']=3
exp21['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp22
exp22['custom_name']="rag5-dpo-beta=0.7-ckpt718"
exp22['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.7/checkpoint-718"
exp22['retrieve_topk']=5
exp22['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp23
exp23['custom_name']="rag5-dpo-beta=0.7-ckpt718"
exp23['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.7/checkpoint-718"
exp23['retrieve_topk']=10
exp23['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

# List of experiments
# experiments=("exp20" "exp21" "exp22" "exp23")

declare -A exp24
exp24['custom_name']="rag5-dpo-beta=1.0-ckpt718"
exp24['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.0/checkpoint-718"
exp24['retrieve_topk']=1
exp24['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp25
exp25['custom_name']="rag5-dpo-beta=1.0-ckpt718"
exp25['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.0/checkpoint-718"
exp25['retrieve_topk']=3
exp25['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp26
exp26['custom_name']="rag5-dpo-beta=1.0-ckpt718"
exp26['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.0/checkpoint-718"
exp26['retrieve_topk']=5
exp26['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp27
exp27['custom_name']="rag5-dpo-beta=1.0-ckpt718"
exp27['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.0/checkpoint-718"
exp27['retrieve_topk']=10
exp27['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG5_LoRA-SFT"

declare -A exp28
exp28['custom_name']="rag1-sft-saved"
exp28['model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"
exp28['retrieve_topk']=3
exp28['base_model_path']=""

# Beta = 0.5
declare -A exp29
exp29['custom_name']="rag5-dpo-nosft-beta=0.5-ckpt718"
exp29['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.5/checkpoint-718"
exp29['retrieve_topk']=1
exp29['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp30
exp30['custom_name']="rag5-dpo-nosft-beta=0.5-ckpt718"
exp30['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.5/checkpoint-718"
exp30['retrieve_topk']=3
exp30['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp31
exp31['custom_name']="rag5-dpo-nosft-beta=0.5-ckpt718"
exp31['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.5/checkpoint-718"
exp31['retrieve_topk']=5
exp31['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp32
exp32['custom_name']="rag5-dpo-nosft-beta=0.5-ckpt718"
exp32['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.5/checkpoint-718"
exp32['retrieve_topk']=10
exp32['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

# Beta = 0.7
declare -A exp33
exp33['custom_name']="rag5-dpo-nosft-beta=0.7-ckpt718"
exp33['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.7/checkpoint-718"
exp33['retrieve_topk']=1
exp33['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp34
exp34['custom_name']="rag5-dpo-nosft-beta=0.7-ckpt718"
exp34['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.7/checkpoint-718"
exp34['retrieve_topk']=3
exp34['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp35
exp35['custom_name']="rag5-dpo-nosft-beta=0.7-ckpt718"
exp35['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.7/checkpoint-718"
exp35['retrieve_topk']=5
exp35['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp36
exp36['custom_name']="rag5-dpo-nosft-beta=0.7-ckpt718"
exp36['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.7/checkpoint-718"
exp36['retrieve_topk']=10
exp36['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

# Beta = 1.0
declare -A exp37
exp37['custom_name']="rag5-dpo-nosft-beta=1.0-ckpt718"
exp37['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.0/checkpoint-718"
exp37['retrieve_topk']=1
exp37['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp38
exp38['custom_name']="rag5-dpo-nosft-beta=1.0-ckpt718"
exp38['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.0/checkpoint-718"
exp38['retrieve_topk']=3
exp38['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp39
exp39['custom_name']="rag5-dpo-nosft-beta=1.0-ckpt718"
exp39['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.0/checkpoint-718"
exp39['retrieve_topk']=5
exp39['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp40
exp40['custom_name']="rag5-dpo-nosft-beta=1.0-ckpt718"
exp40['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.0/checkpoint-718"
exp40['retrieve_topk']=10
exp40['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

# Beta = 1.2
declare -A exp41
exp41['custom_name']="rag5-dpo-nosft-beta=1.2-ckpt718"
exp41['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.2/checkpoint-718"
exp41['retrieve_topk']=1
exp41['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp42
exp42['custom_name']="rag5-dpo-nosft-beta=1.2-ckpt718"
exp42['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.2/checkpoint-718"
exp42['retrieve_topk']=3
exp42['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp43
exp43['custom_name']="rag5-dpo-nosft-beta=1.2-ckpt718"
exp43['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.2/checkpoint-718"
exp43['retrieve_topk']=5
exp43['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp44
exp44['custom_name']="rag5-dpo-nosft-beta=1.2-ckpt718"
exp44['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.2/checkpoint-718"
exp44['retrieve_topk']=10
exp44['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

# Beta = 0.1, checkpoint 500
declare -A exp45
exp45['custom_name']="rag5-dpo-nosft-beta=0.1-ckpt500"
exp45['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.1/checkpoint-500"
exp45['retrieve_topk']=1
exp45['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp46
exp46['custom_name']="rag5-dpo-nosft-beta=0.1-ckpt500"
exp46['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.1/checkpoint-500"
exp46['retrieve_topk']=3
exp46['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp47
exp47['custom_name']="rag5-dpo-nosft-beta=0.1-ckpt500"
exp47['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.1/checkpoint-500"
exp47['retrieve_topk']=5
exp47['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp48
exp48['custom_name']="rag5-dpo-nosft-beta=0.1-ckpt500"
exp48['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=0.1/checkpoint-500"
exp48['retrieve_topk']=10
exp48['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp49
exp49['custom_name']="rag5-dpo-nosft-beta=1.5-ckpt718"
exp49['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.5/checkpoint-718"
exp49['retrieve_topk']=1
exp49['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp50
exp50['custom_name']="rag5-dpo-nosft-beta=1.5-ckpt718"
exp50['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.5/checkpoint-718"
exp50['retrieve_topk']=3
exp50['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp51
exp51['custom_name']="rag5-dpo-nosft-beta=1.5-ckpt718"
exp51['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.5/checkpoint-718"
exp51['retrieve_topk']=5
exp51['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"

declare -A exp52
exp52['custom_name']="rag5-dpo-nosft-beta=1.5-ckpt718"
exp52['model_path']="third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek/rag5_answer-dpo_max=4096_beta=1.5/checkpoint-718"
exp52['retrieve_topk']=10
exp52['base_model_path']="data/jinghong_chen/Qwen2-VL-7B-Instruct_Infoseek-RAG1_LoRA-SFT"


# Update experiments list to run all new experiments
# experiments=("exp49" "exp50" "exp51" "exp52")
experiments=("exp29" "exp31" "exp32")
# experiments=("exp29" "exp30" "exp31" "exp32" "exp33" "exp34" "exp35" "exp36" "exp37" "exp38" "exp39" "exp40" "exp41" "exp42" "exp43" "exp44")
# experiments=("exp45" "exp46" "exp47" "exp48")
# experiments=(
#     # All Top1 experiments
#     # "exp45"  # beta=0.1, topk=1
#     # "exp29"  # beta=0.5, topk=1
#     # "exp33"  # beta=0.7, topk=1
#     # "exp37"  # beta=1.0, topk=1
#     # "exp41"  # beta=1.2, topk=1
    
#     # All Top3 experiments
#     # "exp46"  # beta=0.1, topk=3
#     # "exp30"  # beta=0.5, topk=3
#     # "exp34"  # beta=0.7, topk=3
#     "exp38"  # beta=1.0, topk=3
#     "exp42"  # beta=1.2, topk=3
    
#     # All Top5 experiments
#     # "exp47"  # beta=0.1, topk=5
#     # "exp31"  # beta=0.5, topk=5
#     # "exp35"  # beta=0.7, topk=5
#     "exp39"  # beta=1.0, topk=5
#     "exp43"  # beta=1.2, topk=5
    
#     # All Top10 experiments
#     # "exp48"  # beta=0.1, topk=10
#     # "exp32"  # beta=0.5, topk=10
#     # "exp36"  # beta=0.7, topk=10
#     "exp40"  # beta=1.0, topk=10
#     "exp44"  # beta=1.2, topk=10
# )

# List of experiments
# experiments=("exp24" "exp25" "exp26" "exp27")
# experiments=("exp28")
# experiments=("exp1" "exp2" "exp3")


# Iterate over experiments
for exp in "${experiments[@]}"; do
  eval "model_path=\${${exp}['model_path']}"
  eval "custom_name=\${${exp}['custom_name']}"
  eval "retrieve_topk=\${${exp}['retrieve_topk']}"
  eval "base_model_path=\${${exp}['base_model_path']}"
  exp_name="${DATASET_NAME}_${SPLIT}-${TAKE_N}_${MODE}_RetrieveTopK=${retrieve_topk}_${MODEL_NAME}_${custom_name}_${RETRIEVER_NAME}"

  echo "model_path=${model_path}"
  echo "base_model_path=${base_model_path}"
  echo "retrieve_topk=${retrieve_topk}"
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
    # --debug
    # --override \
done


