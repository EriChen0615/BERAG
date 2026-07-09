#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

DATE=$(date +%m%d)
RETRIEVAL_FIELD="retrieved_passage" # retrieved_passage
for TopK in 15; do
    echo "Running inference for TopK = $TopK"
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt python src/vllm_vqa_inference.py \
        --retrieval_ds_path "outputs/jinghong_chen/EVQA-testfull-with-retrieval_post_reranked" \
        --dataset_name "EVQA" \
        --take_n 0 \
        --img_basedir "." \
        --retrieval_field "$RETRIEVAL_FIELD" \
        --retrieval_topk $TopK \
        --base_model_path "Qwen/Qwen2-VL-7B-Instruct" \
        --processor_path "Qwen/Qwen2-VL-7B-Instruct" \
        --seed 0 \
        --batch_size 256 \
        --exp_name "outputs/0326/EVQA/Qwen2-VL-7B-Instruct-${RETRIEVAL_FIELD}-Top${TopK}" \
        --do_eval 
done