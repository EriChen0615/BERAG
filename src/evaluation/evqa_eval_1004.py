
import sys
sys.path.append('./src')
sys.path.append('./src/evaluation')
from vqa_datasets import load_vqa_dataset
from pprint import pprint
import json
import os

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

# Function to process a single row of the DataFrame
def process_row(row):
    idx = row.Index
    question = row.question
    answers = row.answers.tolist()
    prediction = row.prediction
    question_type = row.question_type

    answers = [ans.replace("\n", "").replace("' '", "', '") for ans in answers]
    # answers = ast.literal_eval(answers)
    answers = [str(ans).strip() for ans in answers]
    try:
        score = evaluation_utils.evaluate_example(
            question,
            reference_list=answers,
            candidate=prediction,
            question_type=question_type)
    except:
        score = 0.0
    
    return_dict = {
        'score': score
    }

    if 'queries' in dir(row):
        qs = row.queries
        docs = row.retrieved_docs
        gt_documents = row.pos_item_contents
        hit = False
        
        for this_query, this_doc in zip(qs, docs):
            if any([this_doc.rstrip('\n') in gt_doc for gt_doc in gt_documents]):
                hit = True
            # if len(qs):
            #     print("qs:", qs)
            #     print("retrieved docs:", docs)
            #     print("gt_documents:", gt_documents)
            #     print("hit=", hit)
            #     breakpoint()
        mean_query_len = np.mean([len(q) for q in qs]) if len(qs) != 0 else 0 # in characters (i.e., len(q))
            
        return_dict.update({
            'retriever_calls': len(qs),
            'mean_query_len': mean_query_len,
            'hit': hit,
            'violated': row.violated
        })
    # print("\tquestion:", question)
    # print("\tprediction:", prediction)
    # print("\tanswers:", answers)
    # print("\tscore:", score)
    # breakpoint()
    return return_dict

def process_row_mp(row):
    idx = row[0]
    question = row[1].question
    answers_raw = row[1].answers.tolist() if isinstance(row[1].answers, pd.Series) else row[1].answers
    # Robust parsing for cached CSV cases where `answers` can be stringified.
    if isinstance(answers_raw, str):
        s = answers_raw.strip()
        if s:
            parsed = None
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = None
            if parsed is None:
                try:
                    parsed = json.loads(s)
                except Exception:
                    parsed = None
            if isinstance(parsed, str):
                # Handle double-encoded CSV values like "\"['a', 'b']\""
                t = parsed.strip()
                if t.startswith("[") or t.startswith("("):
                    try:
                        parsed2 = ast.literal_eval(t)
                    except Exception:
                        parsed2 = None
                    if isinstance(parsed2, (list, tuple)):
                        parsed = parsed2
            if isinstance(parsed, (list, tuple)):
                answers = list(parsed)
            elif parsed is None:
                # Treat as single answer string.
                answers = [answers_raw]
            else:
                answers = [parsed]
        else:
            answers = []
    else:
        # Expected: list-like of answers
        answers = answers_raw.tolist() if hasattr(answers_raw, "tolist") else answers_raw
        if isinstance(answers, tuple):
            answers = list(answers)
        if not isinstance(answers, list):
            answers = [answers]
    prediction = row[1].prediction
    question_type = row[1].question_type

    # Normalize answer strings
    answers = [str(ans).replace("\n", "").replace("' '", "', '") for ans in answers]
    # answers = ast.literal_eval(answers)
    answers = [str(ans).strip() for ans in answers]
    try:
        score = evaluation_utils.evaluate_example(
            question,
            reference_list=answers,
            candidate=prediction,
            question_type=question_type)
    except:
        score = 0.0
    # print("\tquestion:", question)
    # print("\tprediction:", prediction)
    # print("\tanswers:", answers)
    # print("\tscore:", score)
    return score

def extract_queries_and_retrieved_docs(history, extract_answer_with_re=False):
    all_queries = []
    all_docs = []
    for i, turn in enumerate(history):
        if i == 0:
            continue
        response = turn[0]
        if response.startswith("\n[EVIDENCE]"):
            doc = response.split('[EVIDENCE]')[1].lstrip()
            if i != 1:
                if '\n[RETRIEVE]' in history[i-1][0]:
                    query = history[i-1][0].split('\n[RETRIEVE]')[-1]
                else:
                    continue
            elif "[RETRIEVE]" in history[0][0]['text_context']: 
                query = history[0][0]['text_context'].split('[RETRIEVE]')[-1]
            else: # regular RAG
                query = history[0][0]['question']
            all_queries.append(query)
            all_docs.append(doc)
    all_calls = len(all_queries)
    return all_queries, all_docs, all_calls

VALID_BTITLES = ['[THINK]', '[RETRIEVE]', '[ANSWER]', '[EVIDENCE]']
def check_if_violate(history):
    violated = False
    for i, turn in enumerate(history):
        if i == 0:
            continue
        response = turn[0].lstrip('\n')
        if not any([response.startswith(btitle) for btitle in VALID_BTITLES]):
            violated = True
        elif '[RETRIEVE]' in response:
            after_ret = response.split('[RETRIEVE]')[1]
            if any([btitle in after_ret for btitle in VALID_BTITLES]):
                violated = True
    return violated

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction_file", type=str)
    parser.add_argument("--history_file", type=str, default=None)
    parser.add_argument("--mp", type=bool, default=False)
    parser.add_argument("--extract_answer_with_re", action="store_true")

    args = parser.parse_args()

    # Set the number of processes to use
    vqa_dataset  = load_vqa_dataset("EVQA", split="test", img_basedir='data/')
    df = vqa_dataset.to_pandas()

    if args.history_file:
        output_dir  = "/".join(args.history_file.split('/')[:-1])
        with open(args.history_file, 'r') as f:
            histories = json.load(f)
            assert len(histories) == len(df), "histories and df len not equal!"
            if not args.extract_answer_with_re:
                answers = [hist[-1][1] for hist in histories]
            else:
                answers = [hist[-1][0].split('[ANSWER] ')[1] for hist in histories]
            queries_and_docs_and_calls = [extract_queries_and_retrieved_docs(hist) for hist in histories]
            df['queries'] = [qd[0] for qd in queries_and_docs_and_calls]
            df['retrieved_docs'] = [qd[1] for qd in queries_and_docs_and_calls]
            df['calls'] = [qd[2] for qd in queries_and_docs_and_calls]
            df['violated'] = [check_if_violate(hist) for hist in histories]
    else:
        output_dir  = "/".join(args.prediction_file.split('/')[:-1])
        with open(args.prediction_file, 'r') as f:
            answers = json.load(f) 

    df['prediction'] = answers



    all_eval_results = []
    if not args.mp:
        for row in tqdm(df.itertuples(), total=len(df)):
            eval_result = process_row(row)
            all_eval_results.append(eval_result)
    else:
        # num_processes = multiprocessing.cpu_count() // 2 #multiprocessing.cpu_count() // 2 was 64
        num_processes = 8 #multiprocessing.cpu_count() // 2 was 64
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # no GPU for multiprocessing
    # # Create a Pool of workers
        with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm to display progress
            all_eval_results = list(tqdm(pool.imap(process_row_mp, df.iterrows(), chunksize=1), total=len(df)))

    # Calculate and print the average score
    dict_to_report = {f"avg_{k}": sum([res[k] for res in all_eval_results])/len(all_eval_results) for k in all_eval_results[0]}

    

    # average_score = sum(all_scores) / len(all_scores)
    # print(f"Average score: {average_score}")
    for k in all_eval_results[0]:
        df[k] = [res[k] for res in all_eval_results]

    call_df = df[df['calls'] != 0]
    if len(call_df) != 0:
        dict_to_report.update({
            "hit|call": sum(call_df["hit"])/len(call_df) if len(call_df) else 0
        })
    pprint(dict_to_report)

    df.to_csv(f"{output_dir}/marked_answers.csv")
    with open(f'{output_dir}/scores.json', 'w') as f:
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