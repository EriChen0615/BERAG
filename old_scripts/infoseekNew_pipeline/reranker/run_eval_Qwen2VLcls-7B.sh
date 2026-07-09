#!/bin/bash
#SBATCH -J run_InfoseekNew_valid_eval_reranker_QWen2VL-7BCLS
#SBATCH -A GVDD-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere


module purge
module load slurm
module load rhel8/default-amp
module load cuda/11.4
module load gcc/9

export HF_HOME='../jm2245/HF_HOME'
#export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64
DATASET_NAME="InfoseekNew"
SPLIT="valid_m2kr"
TAKE_N=0
RETRIEVER_NAME="PreFLMR-L"
RERANKER_NAME="QWen2VLCLS7B-Doc1Verify-LoRA"
RERANK_TOPK=50
POST_RETRIEVAL_DATASET="outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval"
POST_RERANK_DATASET="outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B"

MODE="RetrievalRerank"
# CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${RETRIEVER_NAME}.jsonnet"
CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${RETRIEVER_NAME}_${RERANKER_NAME}.jsonnet"
IMG_BASEDIR="/rds/project/rds-iS0FZqj9lmg/wl356/infoseek/infoseek_images/images"
EXP_NAME="${DATASET_NAME}_${MODE}_${RETRIEVER_NAME}_${RERANKER_NAME}"

# export WANDB_RUN_GROUP="3090"
MODEL_PATH="/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/checkpoints/qwen2_vl-7b/qlora/infoseeknew/2025-01-24_doc1_verify"

python src/reranker_inference.py \
    --dataset_name $DATASET_NAME \
    --exp_name $EXP_NAME \
    --split $SPLIT \
    --take_n $TAKE_N \
    --rerank_topk $RERANK_TOPK \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --save_retrieved_ds_to $POST_RERANK_DATASET \
    --post_retrieval_dataset $POST_RETRIEVAL_DATASET \
    --model_path $MODEL_PATH \
    --do_retrieve \
    # --debug_cases 1 10 20 30 40 50 60 70 80 90 100 1000 1100 1200 1500 2000 2500 2700 2700 2750 2800 2900 3000 3200 3500

