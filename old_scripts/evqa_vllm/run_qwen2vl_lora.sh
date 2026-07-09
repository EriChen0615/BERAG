#!/bin/bash
DATE=$(date +%m%d)

# Define experiment configurations as associative arrays
declare -A exp1=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)
declare -A exp2=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp3=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
)
declare -A exp4=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
)

declare -A exp5=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft_max-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp6=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft_max-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
)

declare -A exp7=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft_sum_removesmall-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp8=(   
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft_lateinter_removesmall-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp9=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft_sum_removesmall-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
)

declare -A exp10=(   
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft_lateinter_removesmall-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
)

declare -A exp11=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-size=64000-max=4096/checkpoint-1000"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp12=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-size=64000-max=4096/checkpoint-1500"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp13=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1000"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp14=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1500"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp15=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-sum-source_span=question-cali_span=question_token-size=1000-max=4096/checkpoint-1942"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp16=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-sum-source_span=question-cali_span=question_token-size=1000-max=4096/checkpoint-1942"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
)

declare -A exp17=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-lateinter-source_span=question-cali_span=question_token-size=1000-max=4096/checkpoint-1942"
    [retrieval_topk]=5
    [retrieval_field]="retrieved_passage"
)

declare -A exp18=(
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-lateinter-source_span=question-cali_span=question_token-size=1000-max=4096/checkpoint-1942"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
)



# Map experiment names to their config associative arrays
declare -A experiments=(
    # ["EVQA-SFT-1e-5"]="exp1"
    # ["EVQA-AttnSFT-1e-5"]="exp2"
    # ["EVQA-SFT-Top5-Rerank-lr=1e-5"]="exp3"
    # ["EVQA-AttnSFT-Top5-Rerank-lr=1e-5"]="exp4"
    # ["EVQA-AttnSFT-Agg=Max-Top5-Retrieve-lr=1e-5"]="exp5"
    # ["EVQA-AttnSFT-Agg=Max-Top5-Rerank-lr=1e-5"]="exp6"
    # ["EVQA-AttnSFT-Agg=Sum-SMask-Top5-Retrieve-lr=1e-5"]="exp7"
    # ["EVQA-AttnSFT-Agg=LateInter-SMask-Top5-Retrieve-lr=1e-5"]="exp8"
    # ["EVQA-AttnSFT-Agg=Sum-SMask-Top5-Rerank-lr=1e-5"]="exp9"
    # ["EVQA-AttnSFT-Agg=LateInter-SMask-Top5-Rerank-lr=1e-5"]="exp10"
    # ["EVQA-AttnSFT-1e-5-step=1000"]="exp11"
    # ["EVQA-AttnSFT-1e-5-step=1500"]="exp12"
    # ["EVQA-SFT-1e-5-step=1000"]="exp13"
    # ["EVQA-SFT-1e-5-step=1500"]="exp14"
    ["EVQA-AttnSFT-QSpan-Agg=Sum-Top5-Retrieve-lr=1e-5"]="exp15"
    ["EVQA-AttnSFT-QSpan-Agg=Sum-Top5-Rerank-lr=1e-5"]="exp16"
    ["EVQA-AttnSFT-QSpan-Agg=LateInter-Top5-Retrieve-lr=1e-5"]="exp17"
    ["EVQA-AttnSFT-QSpan-Agg=LateInter-Top5-Rerank-lr=1e-5"]="exp18"
)

for exp_name in "${!experiments[@]}"; do
    exp_ref="${experiments[$exp_name]}"
    # Indirect reference to associative array
    declare -n exp_cfg="$exp_ref"

    adapter_path="${exp_cfg[adapter_path]}"
    retrieval_topk="${exp_cfg[retrieval_topk]}"
    retrieval_field="${exp_cfg[retrieval_field]}"

    echo "Running inference for $exp_name"
    echo "Adapter path: $adapter_path"
    echo "Retrieval topk: $retrieval_topk"
    echo "Retrieval field: $retrieval_field"

    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/vllm_vqa_inference.py \
        --retrieval_ds_path "outputs/jinghong_chen/EVQA-testfull-with-retrieval_post_reranked" \
        --dataset_name "EVQA" \
        --take_n 0 \
        --img_basedir "." \
        --retrieval_field "$retrieval_field" \
        --retrieval_topk "$retrieval_topk" \
        --base_model_path "Qwen/Qwen2-VL-2B-Instruct" \
        --processor_path "Qwen/Qwen2-VL-2B-Instruct" \
        --adapter_name_or_path "$adapter_path" \
        --seed 0 \
        --batch_size 3750 \
        --exp_name "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-${exp_name}-${DATE}" \
        --do_eval 
        # --use_cache

    echo "Finished inference for $exp_name"
done

