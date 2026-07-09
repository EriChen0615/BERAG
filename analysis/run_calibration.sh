#!/bin/bash

# E-VQA

TOPK_DOCS=5
TAKE_N=128

# CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python analysis/run_calibration.py \
#     --model_path "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.1/checkpoint-994" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --base_model_path data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT \
#     --is_lora \
#     --dataset_name EVQA_with_evidence \
#     --img_basedir "." \
#     --split test \
#     --take_n $TAKE_N \
#     --topk_docs $TOPK_DOCS \
#     --answer_candidates_file "analysis/calibration/evqa_rag${TOPK_DOCS}_Qwen2-VL-7B_dpo-reranked-${TAKE_N}.json" \
#     --evidence_field reranked_passage \
#     --do_dpo_reward_selection \
#     --ref_model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT"

# CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python analysis/run_calibration.py \
#     --model_path "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/rag5_answer-dpo_max=4096_beta=0.5/checkpoint-994" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --base_model_path data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT \
#     --is_lora \
#     --dataset_name EVQA_with_evidence \
#     --img_basedir "." \
#     --split test \
#     --take_n $TAKE_N \
#     --topk_docs $TOPK_DOCS \
#     --answer_candidates_file "analysis/calibration/evqa_rag${TOPK_DOCS}_Qwen2-VL-7B_dpo(b=0.5)-reranked-${TAKE_N}.json" \
#     --evidence_field reranked_passage \
#     --do_dpo_reward_selection \
#     --ref_model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT"

# CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python analysis/run_calibration.py \
#     --model_path "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/evqa/rag5_answer-dpo_max=4096_beta=1.0/checkpoint-994" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --base_model_path data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT \
#     --is_lora \
#     --dataset_name EVQA_with_evidence \
#     --img_basedir "." \
#     --split test \
#     --take_n $TAKE_N \
#     --topk_docs $TOPK_DOCS \
#     --answer_candidates_file "analysis/calibration/evqa_rag${TOPK_DOCS}_Qwen2-VL-7B_dpo(b=1.0)-reranked-${TAKE_N}.json" \
#     --evidence_field reranked_passage \
#     --do_dpo_reward_selection \
#     --ref_model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT"

CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python analysis/run_calibration.py \
    --model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT" \
    --processor_path QWen/Qwen2-VL-7B-Instruct \
    --dataset_name EVQA_with_evidence \
    --img_basedir "." \
    --split test \
    --take_n $TAKE_N \
    --topk_docs $TOPK_DOCS \
    --answer_candidates_file "analysis/calibration/evqa_rag${TOPK_DOCS}_Qwen2-VL-7B_sft(pre-dpo)-reranked-${TAKE_N}.json" \
    --evidence_field reranked_passage \
    --do_dpo_reward_selection \
    --ref_model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG1_LoRA-SFT"

# CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python analysis/run_calibration.py \
#     --model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG1_LoRA-SFT" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --dataset_name EVQA_with_evidence \
#     --img_basedir "." \
#     --split test \
#     --take_n $TAKE_N \
#     --topk_docs $TOPK_DOCS \
#     --answer_candidates_file "analysis/calibration/evqa_rag${TOPK_DOCS}_Qwen2-VL-7B_sft-reranked-${TAKE_N}.json" \
#     --evidence_field reranked_passage

# CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python analysis/run_calibration.py \
#     --model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_EVQA-RAG5_LoRA-SFT" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --dataset_name EVQA_with_evidence \
#     --img_basedir "." \
#     --split test \
#     --take_n $TAKE_N \
#     --topk_docs $TOPK_DOCS \
#     --answer_candidates_file "analysis/calibration/evqa_rag${TOPK_DOCS}_Qwen2-VL-7B_sft-retrieved-${TAKE_N}.json" \
#     --evidence_field retrieved_passage

# InfoseekNew
# TOPK_DOCS=1
# TAKE_N=512
# python analysis/run_calibration.py \
#     --model_path "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_beta=2.0" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --base_model_path data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT \
#     --is_lora \
#     --dataset_name InfoseekNew_with_evidence \
#     --img_basedir "" \
#     --split test \
#     --take_n $TAKE_N \
#     --topk_docs $TOPK_DOCS \
#     --answer_candidates_file "analysis/calibration/infoseek_rag${TOPK_DOCS}_Qwen2-VL-7B_dpo-retrieve-${TAKE_N}.json" \
#     --do_dpo_reward_selection \
#     --ref_model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT"

# python analysis/run_calibration.py \
#     --model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --dataset_name InfoseekNew_with_evidence \
#     --img_basedir "" \
#     --split test \
#     --take_n $TAKE_N \
#     --topk_docs $TOPK_DOCS \
#     --answer_candidates_file "analysis/calibration/infoseek_rag1_Qwen2-VL-7B_sft-retrieve-${TAKE_N}.json" 

#  python analysis/run_calibration.py \
#     --model_path "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_beta=1.0" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --base_model_path data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT \
#     --is_lora \
#     --dataset_name InfoseekNew_with_evidence \
#     --img_basedir "" \
#     --split test \
#     --take_n 256 \
#     --answer_candidates_file "analysis/calibration/infoseek_rag1_Qwen2-VL-7B_dpo-reranked-256.json" \
#     --evidence_field reranked_passage

# python analysis/run_calibration.py \
#     --model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --dataset_name InfoseekNew_with_evidence \
#     --img_basedir "" \
#     --split test \
#     --take_n 256 \
#     --answer_candidates_file "analysis/calibration/infoseek_rag1_Qwen2-VL-7B_sft-reranked-256.json" \
#     --evidence_field reranked_passage


# python analysis/run_calibration.py \
#     --model_path "third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/infoseek_new/rag1_answer-dpo_max=8196_beta=1.0" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --base_model_path data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT \
#     --is_lora \
#     --dataset_name InfoseekNew_with_evidence \
#     --img_basedir "" \
#     --split test \
#     --take_n 32 \
#     --answer_candidates_file "analysis/calibration/infoseek_rag1_Qwen2-VL-7B_dpo-retrieve-32.json" 

# python analysis/run_calibration.py \
#     --model_path "data/jinghong_chen/Qwen2-VL-7B-Instruct_InfoseekNew-RAG1_LoRA-SFT" \
#     --processor_path QWen/Qwen2-VL-7B-Instruct \
#     --dataset_name InfoseekNew_with_evidence \
#     --img_basedir "" \
#     --split test \
#     --take_n 32 \
#     --answer_candidates_file "analysis/calibration/infoseek_rag1_Qwen2-VL-7B_sft-retrieve-32.json" 