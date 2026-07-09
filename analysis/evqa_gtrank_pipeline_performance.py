import sys
sys.path.append('./src')
sys.path.append('./src/evaluation')
sys.path.append('./src/ops')
import pandas as pd
from tabulate import tabulate
import json
from tqdm import tqdm
import ast
# from retrieve_op import Retrieve

MODEL_NAME = "QWen2-7B-VL-LoRA-SFT"
RANK_FIELD = 'gtdoc_rank' # "pgold_rank" or "gtdoc_rank"
# RETRIEVAL_HISTORY="cache/EVQA/retrieve/Retrieve_histories.json"
RETRIEVAL_HISTORY="cache/EVQA_test256/retrieve-rerank/RetrieveRerank_histories.json"
PIPELINE_TO_MARKED_RESULTS_MAP = {
    # 'Retrieve[Top1]-Read': "outputs/20241111-13-EVQA_test-256_CacheRetrieve[Top1]-Read_QWen2VL-7B_LoRA-SFT[Top1-RAG]-ckpt5000/marked_answers.csv",
    # 'Retrieve[Top3]-Read': "outputs/20241111-13-EVQA_test-256_CacheRetrieve[Top3]-Read_QWen2VL-7B_LoRA-SFT[Top1-RAG]-ckpt5000/marked_answers.csv",
    # 'Retrieve[Top5]-Read': "outputs/20241111-13-EVQA_test-256_CacheRetrieve[Top5]-Read_QWen2VL-7B_LoRA-SFT[Top1-RAG]-ckpt5000/marked_answers.csv",
    # 'Retrieve[Top10]-Read': "outputs/20241111-14-EVQA_test-256_CacheRetrieve[Top10]-Read_QWen2VL-7B_LoRA-SFT[Top1-RAG]-ckpt5000/marked_answers.csv",
    'Retrieve[Top1]-Read': "outputs/20241204-10-EVQA_test-256_CacheRetrieveRerank[Top1]-Read_QWen2VL-7B-LoRA_ckpt5000_PreFLMR-L/marked_answers.csv",
    'Retrieve[Top3]-Read': "outputs/20241204-10-EVQA_test-256_CacheRetrieveRerank[Top3]-Read_QWen2VL-2B-LoRA_ckpt5000_PreFLMR-L/marked_answers.csv",
    'Retrieve[Top5]-Read': "outputs/20241204-11-EVQA_test-256_CacheRetrieveRerank[Top5]-Read_QWen2VL-7B-LoRA_ckpt5000_PreFLMR-L/marked_answers.csv",
    'Retrieve[Top10]-Read': "outputs/20241204-11-EVQA_test-256_CacheRetrieveRerank[Top10]-Read_QWen2VL-7B-LoRA_ckpt5000_PreFLMR-L/marked_answers.csv",
}

# RETRIEVE_OP = Retrieve(op_name='retrieve', retriever="", dummy=True) # for converting cache

def get_gt_rank(df, retrieval_dict):
    gt_ranks = []
    for row in tqdm(df.itertuples()):
        qid, question, pos_ids = row.question_id, row.question, row.pos_item_ids
        cache_key = f"{qid}-{question}".strip()
        retrieved_passages = retrieval_dict[cache_key]['retrieved_docs']
        retrieved_doc_ids = [item['passage_id'] for item in retrieved_passages]

        doc_rank = 100
        for rank, retrieved_doc_id in enumerate(retrieved_doc_ids, start=1):
            if rank >= 50:
                doc_rank = 50
                break
            elif retrieved_doc_id in pos_ids:
                doc_rank = rank
                break
        gt_ranks.append(doc_rank)
    return gt_ranks

def get_pseudogold_rank(df, retrieval_dict):
    pgold_ranks = []
    for row in tqdm(df.itertuples()):
        qid, question, pos_ids, answers = row.question_id, row.question, row.pos_item_ids, ast.literal_eval(row.answers)
        cache_key = f"{qid}-{question}".strip()
        retrieved_passages = retrieval_dict[cache_key]['retrieved_docs']
        retrieved_doc_contents = [item['text'] for item in retrieved_passages]

        doc_rank = 100
        for rank, doc_text in enumerate(retrieved_doc_contents, start=1):
            if rank >= 50:
                doc_rank = 50
                break
            elif any([ans.lower() in doc_text.lower() for ans in answers]):
                doc_rank = rank
                break
        pgold_ranks.append(doc_rank)
    return pgold_ranks

def compute_stats_for_each_df(df, rank_field='gtdoc_rank'):
    filtered_dfs = [
        df[df[rank_field].between(0, 1, inclusive="both")],
        df[df[rank_field].between(2, 3, inclusive="both")],
        df[df[rank_field].between(4, 5, inclusive="both")],
        df[df[rank_field].between(6, 10, inclusive="both")],
        df[df[rank_field].between(11, 25, inclusive="both")],
        df[df[rank_field].between(26, 49, inclusive="both")],
        df[df[rank_field].between(50, 100, inclusive="both")],
    ]
    counts = [len(fdf) for fdf in filtered_dfs]
    pts = [f"{c/len(df)*100:.1f}%" for c in counts]
    bems = [fdf['score'].mean() for fdf in filtered_dfs]
    return counts, pts, bems

def _convert_history_to_cache(history_file):
    cache_lookup = {}
    with open(history_file, 'r') as f:
        histories = json.load(f)
    for history in histories:
        internal_info = history[3]
        question_id, query = internal_info['question_id'], internal_info['text_query']
        cache_key = f"{question_id}-{query}".strip()
        cache_lookup[cache_key] = internal_info
    return cache_lookup

if __name__ == '__main__':
    retrieval_dict = _convert_history_to_cache(RETRIEVAL_HISTORY)
    df_dict = {
        k: pd.read_csv(v) for k, v in PIPELINE_TO_MARKED_RESULTS_MAP.items()
    }

    gtdoc_ranks = get_gt_rank(df_dict['Retrieve[Top1]-Read'], retrieval_dict)
    pgold_ranks = get_pseudogold_rank(df_dict['Retrieve[Top1]-Read'], retrieval_dict)
    result_data = {}
    for k in df_dict:
        df_dict[k]['gtdoc_rank'] = gtdoc_ranks
        df_dict[k]['pgold_rank'] = pgold_ranks
        counts, pts, bems = compute_stats_for_each_df(df_dict[k], rank_field=RANK_FIELD)
        result_data[f"{k}-count"] = [f"{cc} ({pp})" for cc, pp in zip(counts, pts)]
        result_data[f"{k}-bem"] = [f"{b*100:.1f}%" for b in bems]
        print(f"Average BEM {k} = {df_dict[k]['score'].mean()*100:.1f}%")

    result_df = pd.DataFrame(result_data).transpose()
    print(result_df)
    print(tabulate(result_data, headers=list(result_data.keys())))

    