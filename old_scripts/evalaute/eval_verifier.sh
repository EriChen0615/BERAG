#!/bin/bash

VERIFY_HISTORIES=(
    # "outputs/20241024-00-EVQA_CacheRetrieve[Top5]-VerifyDoc[EntityMatch]-Read_QWen2VL-7B/VerifyDoc_EntityMatch_histories.json"
    "outputs/20241023-13-EVQA_CacheRetrieve[Top5]-VerifyDoc[EntityMatch]-Read_QWen2VL-2B/VerifyDoc_EntityMatch_histories.json"
)

for VERIFY_HISTORY in "${VERIFY_HISTORIES[@]}"
do
    python src/evaluation/eval_verifier.py \
        --history_file $VERIFY_HISTORY
        # --prediction_file $PRED_FILE 
done