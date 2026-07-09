#!/bin/bash
DATE=$(date +%m%d)

# Define experiment configurations as associative arrays
declare -A exp1=(
    [lora_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [base_inference_path]="outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-SFT-Top5-Rerank-lr=1e-5-0910/marked_inference_results.csv"
    [attn_rank_mode]="sum"
    [retrieval_field]="reranked_passage"
    [description]="Standard SFT Model"
)

declare -A exp2=(
    [lora_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-lateinter-source_span=question-cali_span=question_token-size=1000-max=4096/checkpoint-1942"
    [base_inference_path]="outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-QSpan-Agg=LateInter-Top5-Rerank-lr=1e-5-0916/marked_inference_results.csv"
    [attn_rank_mode]="late-interaction"
    [retrieval_field]="reranked_passage"
    [description]="AttnSFT Late Interaction Model"
)

declare -A exp3=(
    [lora_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-sum-source_span=question-cali_span=question_token-size=1000-max=4096/checkpoint-1942"
    [base_inference_path]="outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-QSpan-Agg=Sum-Top5-Rerank-lr=1e-5-0916/marked_inference_results.csv"
    [attn_rank_mode]="sum"
    [retrieval_field]="reranked_passage"
    [description]="AttnSFT Sum Aggregation Model"
)

# Base model experiment (no LoRA)
declare -A exp_base=(
    [lora_path]=""
    [base_inference_path]="outputs/0925/EVQA/Qwen2-VL-2B-Instruct/marked_inference_results.csv"
    [attn_rank_mode]="sum"
    [retrieval_field]="reranked_passage"
    [description]="Base Model (No LoRA)"
)

# Common parameters
DATASET_PATH="outputs/jinghong_chen/EVQA-testfull-with-retrieval_post_reranked"
MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"
TAKE_N=512
TOPK_DOCS=5
PROCESS_BATCH_SIZE=128
FORWARD_BATCH_SIZE=1
IMG_BASEDIR="./"

# Map experiment names to their config associative arrays
declare -A experiments=(
    # ["BASE"]="exp_base"
    ["SFT-Rerank"]="exp1"
    ["AttnSFT-LateInter-Rerank"]="exp2"
    ["AttnSFT-Sum-Rerank"]="exp3"
)

for exp_name in "${!experiments[@]}"; do
    exp_ref="${experiments[$exp_name]}"
    # Indirect reference to associative array
    declare -n exp_cfg="$exp_ref"

    lora_path="${exp_cfg[lora_path]}"
    base_inference_path="${exp_cfg[base_inference_path]}"
    attn_rank_mode="${exp_cfg[attn_rank_mode]}"
    retrieval_field="${exp_cfg[retrieval_field]}"
    description="${exp_cfg[description]}"

    # Generate output directory based on experiment name and date
    OUTPUT_DIR="outputs/${DATE}/EVQA-AttnRerank/${exp_name}"

    echo "=========================================="
    echo "Running attention reranking for: $exp_name"
    echo "Description: $description"
    echo "LoRA path: $lora_path"
    echo "Base inference path: $base_inference_path"
    echo "Attention rank mode: $attn_rank_mode"
    echo "Retrieval field: $retrieval_field"
    echo "Output directory: $OUTPUT_DIR"
    echo "=========================================="

    # Build python command with conditional LoRA path
    python_cmd="python src/run_vlm_attn_reranker.py \
        --dataset_path $DATASET_PATH \
        --model_path $MODEL_PATH \
        --take_n $TAKE_N \
        --base_inference_filepath $base_inference_path \
        --attn_rank_mode $attn_rank_mode \
        --retrieval_field $retrieval_field \
        --topk_docs $TOPK_DOCS \
        --process_batch_size $PROCESS_BATCH_SIZE \
        --forward_batch_size $FORWARD_BATCH_SIZE \
        --output_dir $OUTPUT_DIR \
        --img_basedir $IMG_BASEDIR \
        --ensure_passage_hit"

    # Add LoRA path only if it's not empty
    if [ -n "$lora_path" ]; then
        python_cmd="$python_cmd --lora_path $lora_path"
    fi

    # Execute the command
    eval $python_cmd

    echo "Finished attention reranking for: $exp_name"
    echo ""
    
    # Clean up the nameref to avoid conflicts
    unset -n exp_cfg
done

echo "All experiments completed!"