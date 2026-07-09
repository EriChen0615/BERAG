import sys
sys.path.append('./src')
sys.path.append('./src/evaluation')
sys.path.append('./src/ops')
from vqa_datasets import load_vqa_dataset
from pprint import pprint
import json
import os
from retrieve_op import Retrieve

import ast
import evaluation_utils


# This line can be obtained from HF field: "question_type"
# either templated or automatic

from tqdm import tqdm
import multiprocessing
from functools import partial
# read csv
import pandas as pd
# print(df.columns)

# use multi-processing to speed up
import multiprocessing
import numpy as np
from collections import defaultdict

def compute_mrr(pos_ids, retrieved_doc_ids):
    """ Compute the Mean Reciprocal Rank (MRR) based on ground-truth pos_ids and retrieved doc ids """
    for rank, retrieved_doc_id in enumerate(retrieved_doc_ids, start=1):
        if retrieved_doc_id in pos_ids:
            return 1 / rank
    return 0

def compute_recall_at_k(pos_ids, retrieved_doc_ids, recall_dict, Ks=[1,3,5,10,20,50]):
    hit_list = []
    for retrieved_doc_id in retrieved_doc_ids:
        found = False
        for pos_id in pos_ids:
            if pos_id == retrieved_doc_id:
                found = True
        if found:
            hit_list.append(1)
        else:
            hit_list.append(0)
    for K in Ks:
        recall = float(np.max(np.array(hit_list[:K])))
        recall_dict[f"Recall@{K}"].append(recall)
    return recall_dict

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--history_file", type=str, default=None)

    args = parser.parse_args()

    vqa_dataset  = load_vqa_dataset("EVQA", split="test", img_basedir='data/')
    # vqa_dataset  = load_vqa_dataset("EVQA", split="test", take_n=256, img_basedir='data/')
    df = vqa_dataset.to_pandas()
    # Set the number of processes to use

    retrieve_op = Retrieve(op_name='retrieve', retriever="", dummy=True) # for converting cache

    dict_to_report = {}
    output_dir  = "/".join(args.history_file.split('/')[:-1])
    recall_dict = defaultdict(list)
    mrr_list = []
    cache_dict = {}
    with open(args.history_file, 'r') as f:
        retrieval_histories = json.load(f)
        retrieval_histories = retrieval_histories[len(retrieval_histories)-len(df):] #BUG to-fix
        for idx, turn in enumerate(retrieval_histories):
            if 'retrieved_docs' not in turn[3]:
                cache_file, cache_key = turn[3]['cache_file'], turn[3]['cache_key']
                cache_dict_to_use = cache_dict.get(cache_file, None) or retrieve_op._convert_history_to_cache(cache_file)
                cache_dict[cache_file] = cache_dict_to_use
                cached_turn = cache_dict_to_use[cache_key]
                turn[3]['retrieved_docs'] = cached_turn['retrieved_docs']
                    
            retrieved_docs_scores = turn[3]['retrieved_docs'] 
            retrieved_doc_ids = [x['passage_id'] for x in retrieved_docs_scores]
            mrr = compute_mrr(df.iloc[idx]['pos_item_ids'], retrieved_doc_ids)
            mrr_list.append(mrr)
            compute_recall_at_k(df.iloc[idx]['pos_item_ids'], retrieved_doc_ids, recall_dict)
            # for doc_id, score in retrieved_docs_scores:
                # if doc_id in df.iloc[idx]['pos_item_ids']:
                    # hit += 1
                    # break
    for k, v in recall_dict.items():
        df[k] = v
    df['mrr'] = mrr_list

    for k in list(recall_dict.keys())+['mrr']:
        dict_to_report[k] =  df[k].mean()
    pprint(dict_to_report)

    with open(f'{output_dir}/retrieval_report.json', 'w') as f:
        json.dump(dict_to_report, f)
    print("Evaluation results saved to", output_dir)





# all_scores = []
# # iterate over rows
# for index, row in tqdm(df.iterrows(), total=len(df)):
#     question = row[3]
#     answers = row[4]
#     prediction = row[6]
#     # print(question)
#     answers = answers.replace("\n", "").replace("' '", "', '")
#     answers = ast.literal_eval(answers)
#     answers = [answer.strip() for answer in answers]
#     # print(prediction, "->", answers)
#     score = evaluation_utils.evaluate_example(
#         question,
#         reference_list=answers,
#         candidate=prediction,
#         question_type=question_type)
#     all_scores.append(score)
#     # break
    
# print(f"Average score: {sum(all_scores) / len(all_scores)}")