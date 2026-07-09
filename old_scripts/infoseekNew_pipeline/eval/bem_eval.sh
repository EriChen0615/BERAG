#! /bin/bash
DIRS_TO_EVAL=( # "outputs/20250228-11-InfoseekNew_valid_m2kr-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag1_answer_dpo_beta=2.0_ckpt1295_PreFLMR-L"
    # "outputs/20250221-04-InfoseekNew_valid_m2kr-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag1_answer_ckpt2000_PreFLMR-L"
    # "outputs/20250221-05-InfoseekNew_valid_m2kr-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=3_QWen2VL-7B-LoRA_rag1_answer_ckpt2000_PreFLMR-L"
    "outputs/20250221-06-InfoseekNew_valid_m2kr-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B-LoRA_rag1_answer_ckpt2000_PreFLMR-L"
    "outputs/20250221-08-InfoseekNew_valid_m2kr-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B-LoRA_rag1_answer_ckpt2000_PreFLMR-L"
    "outputs/20250306-11-InfoseekNew_valid_m2kr-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag3_answer_dpo_beta=2.0_ckpt1242_PreFLMR-L_7BRerank"
    "outputs/20250306-12-InfoseekNew_valid_m2kr-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=3_QWen2VL-7B-LoRA_rag3_answer_dpo_beta=2.0_ckpt1242_PreFLMR-L_7BRerank"
    "outputs/20250306-14-InfoseekNew_valid_m2kr-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B-LoRA_rag3_answer_dpo_beta=2.0_ckpt1242_PreFLMR-L_7BRerank"
    "outputs/20250306-16-InfoseekNew_valid_m2kr-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B-LoRA_rag3_answer_dpo_beta=2.0_ckpt1242_PreFLMR-L_7BRerank"
)

for dir in ${DIRS_TO_EVAL[@]}; do
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/evaluation/infoseek_bem_eval.py --eval_dir $dir --split valid
done