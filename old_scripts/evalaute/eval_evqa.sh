#!/bin/bash

EXPERIMENT_TO_EVAL=(
    # "outputs/202410-baselines/20241005-05-EVQA_ConvRAG_QWen2VL-7B"
    # "outputs/202410-baselines/20241005-01-EVQA_ConvRAG_QWen2VL-72B"
    # "outputs/202410-baselines/20241005-05-EVQA_NoRAG_QWen2VL-7B"
    # "outputs/202410-baselines/20241004-14-EVQA_NoRAG_QWen2VL-72B"
    # "outputs/202410-baselines/20241005-23-EVQA_ARAG_QWen2VL-7B" 
    # outputs/202410-baselines/20241006-12-EVQA_OracleRAG_QWen2VL-7B
    # outputs/202410-baselines/20241006-11-EVQA_ConvRAG_QWen2VL-7B-top100
    # outputs/202410-baselines/20241006-12-EVQA_OracleRAG_QWen2VL-72B
    # outputs/202410-baselines/20241005-23-EVQA_ARAG_QWen2VL-72B
    # outputs/202410-baselines/20241006-11-EVQA_ConvRAG_QWen2VL-72B-top100
    # outputs/202410-baselines/20241013-17-EVQA_ConvRAG-top5_QWen2VL-7B
    # outputs/202410-baselines/20241013-17-EVQA_ConvRAG-top10_QWen2VL-7B
    # outputs/202410-baselines/20241013-18-EVQA_ConvRAG-top5_QWen2VL-72B
    # outputs/202410-baselines/20241013-18-EVQA_ConvRAG-top10_QWen2VL-72B
    # outputs/202410-baselines/20241013-18-EVQA_OracleRAG_QWen2VL-2B
    # "outputs/202410-pipeline-baselines/20241016-14-EVQA_Rewrite[QR]-Retrieve[Top1]-Read_QWen2VL-7B"
    # "outputs/202410-pipeline-baselines/20241016-14-EVQA_Rewrite[QRwCoT]-Retrieve[Top1]-Read_QWen2VL-7B"
    # "outputs/202410-pipeline-baselines/20241016-17-EVQA_Retrieve[Top5]-Read_QWen2VL-7B"
    # "outputs/202410-pipeline-baselines-qlen512/20241016-14-EVQA_Rewrite[QR]-Retrieve[Top1]-Read_QWen2VL-7B"
    # "outputs/202410-pipeline-baselines-qlen512/20241016-15-EVQA_Rewrite[QRwCoT]-Retrieve[Top1]-Read_QWen2VL-72B"
    # "outputs/202410-pipeline-baselines-qlen512/20241016-16-EVQA_Rewrite[QR]-Retrieve[Top1]-Read_QWen2VL-72B"
    # "outputs/202410-pipeline-baselines-qlen512/20241016-17-EVQA_Retrieve[Top5]-ParallelReadAndRerank_QWen2VL-7B"
    # "outputs/20241017-13-EVQA_CacheRetrieve[Top1]-Read_QWen2VL-7B"
    # "outputs/202410-pipeline-baselines-qlen32/20241017-15-EVQA_CacheRetrieve[Top1]-Read_QWen2VL-7B"
    # "outputs/20241017-17-EVQA_CacheRetrieve[Top1]-Read_QWen2VL-7B"
    # "outputs/20241017-18-EVQA_CacheRetrieve[Top1]-Read_QWen2VL-2B"
    # "outputs/20241017-18-EVQA_CacheRetrieve[Top5]-Read_QWen2VL-2B"
    # "outputs/20241017-22-EVQA_CacheRetrieve[Top1]-Read_QWen2VL-2B"
    # "outputs/20241017-22-EVQA_CacheRetrieve[Top5]-Read_QWen2VL-2B"
    # "outputs/202410-pipeline-baselines-qlen32/20241018-14-EVQA_CacheRetrieve[Top1]-Read_QWen2VL-2B"
    # "outputs/202410-pipeline-baselines-qlen32/20241018-15-EVQA_CacheRetrieve[Top5]-Read_QWen2VL-2B"
    # "outputs/202410-pipeline-baselines-qlen32/20241018-15-EVQA_CacheRetrieve[Top5]-ParallelReadAndRerank_QWen2VL-2B"
    # "outputs/202410-pipeline-baselines-qlen32/20241018-17-EVQA_CacheRetrieve[Top5]-ParallelReadAndRerank_QWen2VL-7B"
    # "outputs/20241019-06-EVQA_CacheRetrieve[Top5]-ParallelReadAndRerank_QWen2VL-7B"
    # "outputs/202410-pipeline-baselines-qlen32/20241021-09-EVQA_CacheRetrieve[Top10]-Read_QWen2VL-2B"
    # "outputs/20241023-13-EVQA_CacheRetrieve[Top5]-VerifyDoc[EntityMatch]-Read_QWen2VL-2B"
    # "outputs/20241024-00-EVQA_CacheRetrieve[Top5]-VerifyDoc[EntityMatch]-Read_QWen2VL-7B"
    # "outputs/20241021-09-EVQA_CacheRetrieve[Top10]-Read_QWen2VL-2B"
    "outputs/20241024-09-EVQA_CacheRetrieve[Top5]-ParallelReadAndRerank_QWen2VL-2B"
)

for eval_exp in "${EXPERIMENT_TO_EVAL[@]}"
do
    PRED_FILE="${eval_exp}/histories.json"
    python src/evaluation/evqa_eval_1004.py \
        --history_file $PRED_FILE \
        --extract_answer_with_re
        # --prediction_file $PRED_FILE 
done