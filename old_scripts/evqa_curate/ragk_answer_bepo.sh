#/bin/bash

DROP_MAX_TOKENS=2048
# SAMPLE_SIZE=64000
# SAMPLE_SIZE=0
SAMPLE_SIZE=0
# SAMPLE_SIZE=10
# SEED=42
SEED=0
TOPK_DOCS=2
# BASE_MODEL_NAME="BEFT[K*=2]-l0h4"
# BASE_MODEL_NAME="BEFT_K=2-cont-K=4-l1h4-n4beams"
BASE_MODEL_NAME="BEFT_K=2-1epoch-l1h4-n4beams"
BEPO_MODE="find_wrong_from_most_likely"
# MODEL_SAMPLE_FILEPATH="outputs/1025/BAPE_sample/EVQA-BAPE-BEFT[K*=2]-l0h4-K=2-h4-prior=prior_head-retrieved_passage-TakeN=30000/marked_inference_results.csv"
# MODEL_SAMPLE_FILEPATH="outputs/1025/BAPE_sample/EVQA-BAPE-BEFT[K*=2]-cont-[K*=4]-l1h4-n4beams-K=2-h4-prior=prior_head-retrieved_passage-TakeN=128/aggregated_scores.csv"
# MODEL_SAMPLE_FILEPATH="outputs/1125/BAPE_sample/EVQA-BAPE-BEFT[K*=2]-cont-[K*=4]-l1h4-n4beams-K=5-h4-hasGTdoc-prior=prior_head-retrieved_passage-TakeN=20/aggregated_scores.csv"
# MODEL_SAMPLE_FILEPATH="outputs/1125/BAPE_sample/EVQA-BAPE-BEFT[K*=2]-cont-[K*=4]-l1h4-n4beams-K=5-h4-hasGTdoc-prior=prior_head-retrieved_passage-TakeN=15000/aggregated_scores.csv"
MODEL_SAMPLE_FILEPATH="outputs/1125/BAPE_sample/EVQA-BAPE-BEFT[K*=2]-1epoch-prior=mlp_lr1e-6-n4beams-K=2-h4-prior=prior_head-retrieved_passage-TakeN=64000/aggregated_scores.csv"
OUTPUT_DIR="third_party/LLaMAFactory/data/jinghong_chen/evqa/${BASE_MODEL_NAME}-bepo_K=2-mode=${BEPO_MODE}-size=${SAMPLE_SIZE}-max=${DROP_MAX_TOKENS}"

python src/curate/ragk_answer_ppl.py \
    --hf_dataset_path "outputs/jinghong_chen/EVQA-with-retrieval" \
    --model_sample_filepath $MODEL_SAMPLE_FILEPATH \
    --passage_set_name "EVQA" \
    --mode "bepo" \
    --topk_docs $TOPK_DOCS \
    --sample_size $SAMPLE_SIZE \
    --img_basedir "../.." \
    --output_dir $OUTPUT_DIR \
    --drop_max_tokens $DROP_MAX_TOKENS \
    --num_workers 8 \
    --seed $SEED \
    --batch_size 4096 \
    --bepo_mode $BEPO_MODE