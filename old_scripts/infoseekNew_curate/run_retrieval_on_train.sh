#!/bin/bash
#SBATCH -J run_retrieval_on_train_InfoseekNew
#SBATCH -A GVDD-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere

DATASET_NAME="InfoseekNew"
SPLIT="train"
RETRIEVER_NAME="PreFLMR-L"
MODE="Retrieval"
CONFIG_FILE="config/${DATASET_NAME}/${MODE}_${RETRIEVER_NAME}.jsonnet"
IMG_BASEDIR="/rds/project/rds-iS0FZqj9lmg/wl356/infoseek/infoseek_images/images"
EXP_NAME="${DATASET_NAME}_${MODE}"
RETRIEVAL_DS_SAVEPATH="outputs/0jingbiao_mei/InfoseekNew-train64000-with-retrieval"
TAKE_N=64000

# export WANDB_RUN_GROUP="3090"
export HF_HOME='../jm2245/HF_HOME'
#export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64
module purge
module load slurm
module load rhel8/default-amp
module load cuda/11.4
# First time run colbert engine requires to load cuda/11.4
#module load cuda/12.1
module load gcc/9
python src/retriever_inference.py \
    --dataset_name $DATASET_NAME \
    --exp_name $EXP_NAME \
    --split $SPLIT \
    --config_file $CONFIG_FILE \
    --img_basedir $IMG_BASEDIR \
    --save_retrieved_ds_to $RETRIEVAL_DS_SAVEPATH \
    --take_n $TAKE_N
    # --do_sanity_check 
    # --debug_cases 1 10 20 30 40 50 60 70 80 90 100 1000 1100 1200 1500 2000 2500 2700 2700 2750 2800 2900 3000 3200 3500

