#NOTE JC: See https://xjpf20fazxn9.jp.larksuite.com/wiki/NjRaw7dciiK5PXka8ihjDJJ4pXd for corresponding table.

NAME_TO_EXPDIRS_TO_EVAL_MAP = {
    # "2B_pretrained": "",
    # "7B_pretrained", ""
    # "2B_RAG-SFT": "",
    # "2B_NoRAG-SFT": "",
    # "7B_NoRAG-SFT": "",
    # "2B_RAG-SFT+SFT": "",
    # "2B_RAG-SFT+DPO": "",
    # "7B_PreFLMR_RAG-SFT": "",
    # "7B_GoldRetrieval_RAG-SFT": "outputs/20250121-15-InfoseekNew_valid_m2kr-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag1-ft_ckpt-2000_PreFLMR-L",
    # "7B_PseudoGoldRetrieval_RAG-SFT": "outputs/20250121-16-Infoseek_test-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag1-ft_ckpt-2000_PreFLMR-L",
    "NoRetrieval_2B-pretrained": "outputs/20241209-17-Infoseek_test-256_NoRAGRead_QWen2VL-2B",
    # "NoRetrieval_7B-pretrained": "",
    # "NoRetrieval_2B-direct-SFT": "",
    "NoRetrieval_7B-direct-SFT": "outputs/20241212-14-Infoseek_test-256_NoRAGRead_QWen2VL-7B_ckpt2000",

    # "Retrieval-K=1_7B-pretrained": "",
    "Retrieval-K=1_7B-RAG1-SFT": "outputs/20241213-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    "Retrieval-K=5_7B-RAG1-SFT": "outputs/20241213-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    "Retrieval-K=1_7B-RAG5-SFT": "outputs/20250121-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag5-sft-ckpt718_PreFLMR-L",
    "Retrieval-K=5_7B-RAG5-SFT": "outputs/20250121-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=5_QWen2VL-7B-LoRA_rag5-sft-ckpt718_PreFLMR-L",

    # "RetrievalReranking-K=1_7B-pretrained": "",
    "RetrievalReranking-K=1_7B-RAG1-SFT": "outputs/20241213-15-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    "RetrievalReranking-K=5_7B-RAG1-SFT": "outputs/20241213-15-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    "RetrievalReranking-K=1_7B-RAG5-SFT": "outputs/20241229-03-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag5-sft_PreFLMR-L",
    "RetrievalReranking-K=5_7B-RAG5-SFT": "outputs/20241229-03-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag5-sft_PreFLMR-L",
}

import sys
import os
import json
from tqdm import tqdm
import pandas as pd
sys.path.append('./src/evaluation')
from vqaEval import VQAEval
from vqa_tools import VQA

def read_and_mark_infoseek_results_as_df(expdir):
    # setup 
    data = []
    for history in tqdm(histories, total=len(histories), desc=f"Reading from {expdir}"):
        question_id = history[0][0]['question_id']
        question = history[0][0]['question']
        answer = history[-1][0].split('[ANSWER] ')[-1]
        img_path = history[0][0]['img_path']
        vqaEval.evaluate(quesIds=[int(question_id)], verbose=False)
        # score = vqaEval.accuracy['overall']
        # gt_answer = vqaEval.vqa.qa[int(question_id)]['answers'][0]['answer']
        data.append((question_id, img_path, question, answer))
        print(data[-1])
        breakpoint()
    
    df = pd.DataFrame(data, columns=['question_id', 'img_path', 'question', 'answer'])
    sys.path.append('./src/evaluation')

    from evqa_eval_1004 import process_row as eval_process_row
    from evqa_eval_1004 import process_row_mp as eval_process_row_mp
    from evqa_eval_1004 import extract_queries_and_retrieved_docs

    import tensorflow as tf
    if tf.test.is_gpu_available():
        for row in tqdm(df.itertuples(), total=len(df)):
            eval_result = eval_process_row(row)
            all_eval_results.append(eval_result)
        dict_to_report = {f"avg_{k}": sum([res[k] for res in all_eval_results])/len(all_eval_results) for k in all_eval_results[0]}
        for k in all_eval_results[0]:
            df[k] = [res[k] for res in all_eval_results]
    else:
        import multiprocessing
        with multiprocessing.Pool(processes=8) as pool:
            all_eval_results = list(tqdm(pool.imap(eval_process_row_mp, df.iterrows(), chunksize=1), total=len(df)))
        dict_to_report = {f"avg_score": sum(all_eval_results)/len(all_eval_results)}
        df['score'] = all_eval_results
    
    return df