#!/bin/bash
#!/bin/bash
#SBATCH -J EVQA_TestFull_Rerank
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere

DATASET_NAME="EVQA"
SPLIT="test"
TAKE_N=0
RETRIEVER_NAME="PreFLMR-L"
RERANKER_NAME="QWen2VL-Doc1Verify-LoRA"
RERANK_TOPK=50
# POST_RETRIEVAL_DATASET="cache/EVQA_test256/${RETRIEVER_NAME}_post_retrieval"
POST_RETRIEVAL_DATASET="outputs/jinghong_chen/EVQA-testfull-with-retrieval"
MODE="RetrievalRerank"
# CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${RETRIEVER_NAME}.jsonnet"
CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${RETRIEVER_NAME}_${RERANKER_NAME}.jsonnet"
IMG_BASEDIR="../vqa_data/KBVQA_data/EVQA/images/"
EXP_NAME="${DATASET_NAME}_${MODE}"

# export WANDB_RUN_GROUP="3090"

python src/reranker_inference.py \
    --dataset_name $DATASET_NAME \
    --exp_name $EXP_NAME \
    --split $SPLIT \
    --take_n $TAKE_N \
    --rerank_topk $RERANK_TOPK \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --save_retrieved_ds_to $POST_RETRIEVAL_DATASET \
    --do_retrieve
    # --post_retrieval_dataset $POST_RETRIEVAL_DATASET 
    # --debug 

    # --do_retrieve \
    # --debug_cases 1 10 20 30 40 50 60 70 80 90 100 1000 1100 1200 1500 2000 2500 2700 2700 2750 2800 2900 3000 3200 3500

