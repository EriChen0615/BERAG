#!/bin/bash
DATE=$(date +%m%d)

# Define experiment configurations as associative arrays
declare -A exp1=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp2=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp3=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1001_summarize_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
    [prefill_ans_token]="true"
)

declare -A exp4=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=1
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [include_gt_passage_only]="true"
)

declare -A exp5=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=1
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    # [include_z0_in_ensemble]="true"
    # [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp6=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=3
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    # [include_z0_in_ensemble]="true"
    # [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp7=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    # [include_z0_in_ensemble]="true"
    # [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp8=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=10
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    # [include_z0_in_ensemble]="true"
    # [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp9=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG1_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    # [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=1
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp10=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG1_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    # [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=3
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp11=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG1_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    # [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp12=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG1_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    # [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=1
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
    [include_gt_passage_only]="true"
)

declare -A exp13=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG5_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-2579"
    [retrieval_topk]=1
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp14=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG5_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-2579"
    [retrieval_topk]=3
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp15=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG5_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-2579"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp16=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG5_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-2579"
    [retrieval_topk]=1
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
    [include_gt_passage_only]="true"
)

declare -A exp17=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG5_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-2579"
    [retrieval_topk]=2
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
    # [include_gt_passage_only]="true"
)

declare -A exp18=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG1_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    # [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/sft/rag5_answer-sft-size=64000-max=4096/checkpoint-1943"
    [retrieval_topk]=2
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
)

declare -A exp19=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=2
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    # [include_z0_in_ensemble]="true"
    # [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp20=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=2
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [ensure_gt_passage_in_ensemble]="true"
    # [include_z0_inj_ensemble]="true"
)

declare -A exp21=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=3
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [ensure_gt_passage_in_ensemble]="true"
    # [include_z0_inj_ensemble]="true"
)

declare -A exp22=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    # [include_z0_inj_ensemble]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp23=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG5_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-2579"
    [retrieval_topk]=2
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp24=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG5_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-2579"
    [retrieval_topk]=3
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp25=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG5_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.3/checkpoint-2579"
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [prompt_template]="config/prompts/1003_conventional_rag.txt"
    [do_eval]="true"
    [use_cache]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp26=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG1_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=1
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [include_z0_in_ensemble]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp27=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG1_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=3
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [include_z0_in_ensemble]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp28=(
    [model_path]="data/jinghong_chen/Qwen2-VL-2B-Instruct_EVQA-RAG1_LoRA-SFT"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]=""
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [prompt_template]=""
    [do_eval]="true"
    [use_cache]="true"
    [include_z0_in_ensemble]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp29=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-sum-source_span=question-cali_span=question_token-size=1000-max=4096/checkpoint-1942"
    [prompt_template]=""
    [retrieval_topk]=1
    [retrieval_field]="reranked_passage"
    [do_eval]="true"
    [include_z0_in_ensemble]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp30=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-sum-source_span=question-cali_span=question_token-size=1000-max=4096/checkpoint-1942"
    [prompt_template]=""
    [retrieval_topk]=3
    [retrieval_field]="reranked_passage"
    [do_eval]="true"
    [include_z0_in_ensemble]="true"
    [ensure_gt_passage_in_ensemble]="true"
)


declare -A exp31=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/attn/rag5_answer-attn_sft-sum-source_span=question-cali_span=question_token-size=1000-max=4096/checkpoint-1942"
    [prompt_template]=""
    [retrieval_topk]=5
    [retrieval_field]="reranked_passage"
    [do_eval]="true"
    [include_z0_in_ensemble]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

declare -A exp32=(
    [model_path]="Qwen/Qwen2-VL-2B-Instruct"
    [processor_path]="Qwen/Qwen2-VL-2B-Instruct"
    [adapter_path]="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[joint]-size=64000-max=2048"
    [prompt_template]=""
    [retrieval_topk]=1
    [retrieval_field]="reranked_passage"
    [include_z0_in_ensemble]="true"
    [ensure_gt_passage_in_ensemble]="true"
)

# Map experiment names to their config associative arrays
declare -A experiments=(
    # ["EVQA-BAPE-Base-Answer"]="exp1"
    # ["EVQA-BAPE-SFT"]="exp2"
    # ["EVQA-BAPE-Base-Summarize"]="exp3"
    # ["EVQA-GTdoc"]="exp4"
    # ["EVQA-BAPE-Base-Answer-GTdoc"]="exp1"
    # ["EVQA-BAPE-Base-Answer-GTdoc-withZ0"]="exp5"
    # ["EVQA-BAPE-Base-Answer-hasGTdoc-withZ0-K=1"]="exp5"
    # ["EVQA-BAPE-Base-Answer-hasGTdoc-withZ0-K=3"]="exp6"
    # ["EVQA-BAPE-Base-Answer-hasGTdoc-withZ0-K=5"]="exp7"
    # ["EVQA-BAPE-Base-Answer-hasGTdoc-withZ0-K=10"]="exp8"
    # ["EVQA-BAPE-Base-Answer-K=1"]="exp5"
    # ["EVQA-BAPE-Base-Answer-K=3"]="exp6"
    # ["EVQA-BAPE-Base-Answer-K=5"]="exp7"
    # ["EVQA-BAPE-Base-Answer-withZ0-K=10"]="exp8"
    # ["EVQA-BAPE-SFT-Answer-K=1"]="exp9"
    # ["EVQA-BAPE-SFT-Answer-K=3"]="exp10"
    # ["EVQA-BAPE-SFT-Answer-K=5"]="exp11"
    # ["EVQA-BAPE-SFT-Answer-K=1-GTdoc"]="exp12"
    # ["EVQA-BAPE-DPO[Top5]-Answer-K=1"]="exp13"
    # ["EVQA-BAPE-DPO[Top5]-Answer-K=3"]="exp14"
    # ["EVQA-BAPE-DPO[Top5]-Answer-K=5"]="exp15"
    # ["EVQA-BAPE-DPO[Top5]-Answer-K=1-GTdoc"]="exp16"
    # ["EVQA-BAPE-DPO[Top5]-Answer-K=2"]="exp17"
    # ["EVQA-BAPE-SFT-Answer-K=2"]="exp18"
    # ["EVQA-BAPE-BASE-Answer-K=2"]="exp19"
    # ["EVQA-BAPE-BASE-hasGTdoc-K=2"]="exp20"
    # ["EVQA-BAPE-BASE-hasGTdoc-K=3"]="exp21"
    # ["EVQA-BAPE-BASE-hasGTdoc-K=5"]="exp22"
    # ["EVQA-BAPE-DPO[Top5]-hasGTdoc-Answer-K=2"]="exp23"
    # ["EVQA-BAPE-DPO[Top5]-hasGTdoc-Answer-K=3"]="exp24"
    # ["EVQA-BAPE-DPO[Top5]-hasGTdoc-Answer-K=5"]="exp25"
    # ["EVQA-BAPE-SFT-Answer-hasGTdoc-withZ0-K=1"]="exp26"
    # ["EVQA-BAPE-SFT-Answer-hasGTdoc-withZ0-K=3"]="exp27"
    # ["EVQA-BAPE-SFT-Answer-hasGTdoc-withZ0-K=5"]="exp28"
    # ["EVQA-BAPE-AttnSFT-Answer-hasGTdoc-withZ0-K=1"]="exp29"
    # ["EVQA-BAPE-AttnSFT-Answer-hasGTdoc-withZ0-K=3"]="exp30"
    # ["EVQA-BAPE-AttnSFT-Answer-hasGTdoc-withZ0-K=5"]="exp31"
    ["EVQA-BAPE-PPL-Answer-hasGTdoc-withZ0-K=1"]="exp32"
)

for exp_name in "${!experiments[@]}"; do
    exp_ref="${experiments[$exp_name]}"
    # Indirect reference to associative array
    declare -n exp_cfg="$exp_ref"

    adapter_path="${exp_cfg[adapter_path]}"
    retrieval_topk="${exp_cfg[retrieval_topk]}"
    retrieval_field="${exp_cfg[retrieval_field]}"

    echo "--------------------------------"
    echo "Running inference for $exp_name"
    echo "Adapter path: $adapter_path"
    echo "Retrieval topk: $retrieval_topk"
    echo "Retrieval field: $retrieval_field"
    echo "Prompt template: ${exp_cfg[prompt_template]}"

    # Build arguments array
    args=(
        --retrieval_ds_path "outputs/jinghong_chen/EVQA-testfull-with-retrieval_post_reranked"
        --dataset_name "EVQA"
        --take_n 16
        --img_basedir "."
        --retrieval_field "$retrieval_field"
        --retrieval_topk "$retrieval_topk"
        --model_path "${exp_cfg[model_path]}"
        --processor_path "${exp_cfg[processor_path]}"
        --adapter_name_or_path "$adapter_path"
        --prompt_template "${exp_cfg[prompt_template]}"
        --seed 0
        --batch_size 1
        --exp_name "outputs/0925/BAPE/Qwen2-VL-2B-Instruct-${exp_name}"
    )

    # Conditionally add store_true flags
    if [[ "${exp_cfg[do_eval]}" == "true" ]]; then
        args+=(--do_eval)
    fi

    if [[ "${exp_cfg[use_cache]}" == "true" ]]; then
        args+=(--use_cache)
    fi

    if [[ "${exp_cfg[prefill_ans_token]}" == "true" ]]; then
        args+=(--prefill_ans_token)
    fi

    if [[ "${exp_cfg[include_z0_in_ensemble]}" == "true" ]]; then
        args+=(--include_z0_in_ensemble)
    fi

    if [[ "${exp_cfg[include_gt_passage_only]}" == "true" ]]; then
        args+=(--include_gt_passage_only)
    fi

    if [[ "${exp_cfg[ensure_gt_passage_in_ensemble]}" == "true" ]]; then
        args+=(--ensure_gt_passage_in_ensemble)
    fi

    echo "Args: ${args[@]}"

    # Run the command
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/bape_vqa_inference.py "${args[@]}"

    echo "Finished inference for $exp_name"
    echo "--------------------------------"
done

