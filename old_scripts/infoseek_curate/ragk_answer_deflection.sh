#!/bin/bash
#SBATCH -A BYRNE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p icelake
#! Notes for icelake partition:
#! - Each node has 76 CPUs (cores)
#! - Each CPU is allocated 3380 MiB (~3.3 GB) of memory by default
#! - For 16 CPUs: default memory = 16 * 3380 MiB ≈ 54 GB
#! - We request 80G to have extra headroom for image processing

# Script to generate BEFT training data with controlled deflection ratio
# Samples data points with TopK=2, precisely controlling deflection=0/1 ratio

DROP_MAX_TOKENS=4096
SAMPLE_SIZE=64000
SAMPLE_OFFSET=0
TOPK_DOCS=2
DEFLECTION_RATIO=0.5  # Ratio of deflection=1 samples (50% deflection=1, 50% deflection=0)
SEED=123  # Different seed from other scripts
OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/Infoseek/rag${TOPK_DOCS}-answer-controlled-deflection-ratio=${DEFLECTION_RATIO}-size=${SAMPLE_SIZE}-max=${DROP_MAX_TOKENS}"
HF_DATASET_PATH="outputs/0jingbiao_mei/InfoseekNew-train64000-with-retrieval"

python src/curate/ragk_answer_ppl_controlled_deflection.py \
    --hf_dataset_path $HF_DATASET_PATH \
    --passage_set_name "InfoseekNew_FullPassage" \
    --topk_docs $TOPK_DOCS \
    --sample_size $SAMPLE_SIZE \
    --sample_offset $SAMPLE_OFFSET \
    --img_basedir "" \
    --output_dir $OUTPUT_DIR \
    --drop_max_tokens $DROP_MAX_TOKENS \
    --num_workers 16 \
    --seed $SEED \
    --batch_size 4096 \
    --mode "sft" \
    --deflection_ratio $DEFLECTION_RATIO \
    --passage_format "dict"

