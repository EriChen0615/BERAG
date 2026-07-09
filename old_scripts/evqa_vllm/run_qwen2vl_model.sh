#!/bin/bash
DATE=$(date +%m%d)

declare -A models=(
    ["Qwen2VL-2B-EVQA_RAG1_LoRA-SFT"]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG1_LoRA-SFT"
    # ["QWen2VL-2B-EVQA_RAG5_LoRA-SFT"]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG5_LoRA-SFT"
    # ["Qwen2VL-7B-EVQA_RAG5_LoRA-SFT"]="data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT"
)

for model in "${!models[@]}"; do
    model_path="${models[$model]}"
    echo "Running inference for $model"
    echo "Model path: $model_path"


    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/vllm_vqa_inference.py \
        --retrieval_ds_path "outputs/jinghong_chen/EVQA-testfull-with-retrieval" \
        --dataset_name "EVQA" \
        --take_n 0 \
        --img_basedir "." \
        --prompt_template "config/prompts/1003_conventional_rag_noplaceholder.txt" \
        --prefill_ans_token \
        --retrieval_field "retrieved_passage" \
        --retrieval_topk 5 \
        --model_path "$model_path" \
        --seed 0 \
        --batch_size 256 \
        --exp_name "outputs/0925/EVQA/${model}" \
        --do_eval 
        # --use_cache 
        # --exp_name "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-${model}" \
    echo "Finished inference for $model"
    done

