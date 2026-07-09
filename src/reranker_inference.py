import sys
sys.path.append('./src')

import _jsonnet
from easydict import EasyDict
from pprint import pprint
import json
from tqdm import tqdm
import wandb 
import retrievers
import rerankers
import argparse
from datetime import datetime
import os
from vqa_datasets import load_vqa_dataset, load_passages
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd
from pprint import pprint
import ast
from collections import defaultdict
import numpy as np


def evaluate_retrieval(ds, retrieval_field, pid_to_content_map, Ks=[1,3,5,10,25,50,100]):
    def compute_mrr(pos_ids, retrieved_doc_ids):
        """ Compute the Mean Reciprocal Rank (MRR) based on ground-truth pos_ids and retrieved doc ids """
        for rank, retrieved_doc_id in enumerate(retrieved_doc_ids, start=1):
            if retrieved_doc_id in pos_ids:
                return 1 / rank
        return 0
    
    eval_results = defaultdict(list)
    for idx, item in enumerate(ds):
        pos_ids = item['pos_item_ids']
        ret_ids = [d['passage_id'] for d in item[retrieval_field]]
        ret_texts = [pid_to_content_map[d['passage_id']] for d in item[retrieval_field]]

        mrr = compute_mrr(pos_ids, ret_ids)
        eval_results['mrr'].append(mrr)

        hit_list = []
        pseudohit_list = []
        for ret_text, ret_id in zip(ret_texts, ret_ids):
            found = False
            pseudo_found = False
            for pos_id in pos_ids:
                if pos_id == ret_id:
                    found = True
            if found:
                hit_list.append(1)
            else:
                hit_list.append(0)

            if any([ans.lower() in ret_text.lower() for ans in item['answers']]):
                pseudo_found = True
            if pseudo_found:
                pseudohit_list.append(1)
            else:
                pseudohit_list.append(0)
        for K in Ks:
            recall = float(np.max(np.array(hit_list[:K])))
            eval_results[f"Recall@{K}"].append(recall)
            precall = float(np.max(np.array(pseudohit_list[:K])))
            eval_results[f"Pseudo Recall@{K}"].append(precall)
    
    dict_to_report = {
        'MRR': np.mean(eval_results['mrr'])
    }
    for K in Ks:
        dict_to_report.update({
            f"Recall@{K}": np.mean(eval_results[f"Recall@{K}"]),
            f"Pseudo Recall@{K}": np.mean(eval_results[f"Pseudo Recall@{K}"]),
        })
    return dict_to_report, eval_results
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--take_n", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--img_basedir", type=str, default='data')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv_filename", type=str, default=None)
    parser.add_argument("--query_field", type=str, default='question')
    parser.add_argument("--no_image_search", action="store_true")
    parser.add_argument("--use_doc_encoder_for_query", action="store_true")
    parser.add_argument("--do_sanity_check", action="store_true")
    parser.add_argument("--do_retrieve", action="store_true")
    parser.add_argument("--post_retrieval_dataset", type=str, help='path to ds with retrieval results')
    parser.add_argument("--save_retrieved_ds_to", type=str, default=None)
    parser.add_argument("--rerank_topk", type=int, default=50)
    

    parser.add_argument("--config_file", type=str, default='config/config.jsonnet')
    parser.add_argument("--output_dir", type=str, default='outputs')
    parser.add_argument("--no_instruction", action='store_true')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_cases", nargs="+")

    parser.add_argument("--model_path", type=str, default=None, 
                        help='path to the model to overwrite the jsonnet config')
    parser.add_argument("--load_4bit", action="store_true", default=False)
    args = parser.parse_args()

    # Initialize wandb
    # wandb.init(project="a-ravqa", name=args.exp_name, entity="byrne-lab")
 
    current_time = datetime.now().strftime("%Y%m%d-%H")
    output_dir = f"{args.output_dir}/{current_time}-{args.exp_name}"
    print("output_dir=",output_dir)
    os.makedirs(output_dir, exist_ok=True)
    test_schema = json.loads(_jsonnet.evaluate_file(args.config_file))
    pprint(test_schema)

    if args.do_retrieve:
        # When we want to do query rewritting evaluation, 
        # we need to load the dataset with overwrtten queries
        # Or we can just load the new set of queries 
        # and overwrite the questions in the dataset
        if args.csv_filename is None:
            vqa_dataset = load_vqa_dataset(args.dataset_name, split=args.split, img_basedir=args.img_basedir, take_n=args.take_n)
        else:
            df = pd.read_csv(args.csv_filename)
            for col in df.columns: # Convert back to list
                if type(df[col][0]) is str and df[col][0].startswith("["):
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            vqa_dataset = Dataset.from_pandas(df)
        if args.debug:
            if args.debug_cases:
                vqa_dataset = vqa_dataset.select([int(i) for i in args.debug_cases])
            else:
                vqa_dataset = vqa_dataset.select([i for i in range(100)])
                args.save_retrieved_ds_to += '_debug'
                args.post_retrieval_dataset += '_debug'

        retriever = getattr(retrievers, test_schema["retriever_class"])(
            **test_schema["retriever_config"],
            add_instruction=(not args.no_instruction),
        )

        if args.do_sanity_check:
            all_passages_in_index = set(retriever.passage_ids)
            for row in tqdm(vqa_dataset, desc='sanity checking GT doc in index'):
                assert all([gtdoc in all_passages_in_index for gtdoc in row['pos_item_ids']]), "GT document not in index!"
        
        # Retrieval
        ds, retrieval_report = retriever.query_and_evaluate_ds(
            vqa_dataset,
            query_field=args.query_field,
            no_image_search=args.no_image_search,
            use_doc_encoder_for_query=args.use_doc_encoder_for_query,
            save_retrieved_ds_to=args.save_retrieved_ds_to,
            return_ds=True,
            **test_schema["query_and_evaluate_ds_kwargs"]
        )
        pprint(retrieval_report)
        with open(f"{output_dir}/retrieval_report.json", 'w') as f:
            json.dump(retrieval_report, f)
        print("Saved to", f"{output_dir}/retrieval_report.json")
    else:
        ds = load_from_disk(args.post_retrieval_dataset)
        print("Loaded post retrieval dataset at", args.post_retrieval_dataset)
    
    # ds is a dataset with retrieval results
    if args.model_path is not None:
        test_schema["reranker_config"]["model_path"] = args.model_path
    if args.load_4bit:
        test_schema["reranker_config"]["load_4bit"] = args.load_4bit
    reranker = getattr(rerankers, test_schema["reranker_class"])(
        **test_schema["reranker_config"]
    )

    # Load the passage set
    passage_ds, pid_to_content_map = load_passages(args.dataset_name, split=args.split)

    def rerank(item):
        retrieved_docs = item['retrieved_passage'][:args.rerank_topk]
        for doc in retrieved_docs:
            doc['text'] = pid_to_content_map[doc['passage_id']]
        reranked_docs = reranker.rank(question=item['question'], query_img=item['img_path'], retrieved_docs=retrieved_docs)
        item['reranked_passage'] = [{k: v for k, v in d.items() if k!='text'} for d in reranked_docs]
        return item
    
    # ds = ds.map(rerank, load_from_cache_file=False)
    ds = ds.map(rerank)
    
    print(ds)
    without_rerank_res, _  = evaluate_retrieval(ds, 'retrieved_passage', pid_to_content_map)
    with_rerank_res, _ = evaluate_retrieval(ds, 'reranked_passage', pid_to_content_map)
    print("Without reranking:", without_rerank_res)
    print("With reranking:", with_rerank_res)

    ds.save_to_disk(args.save_retrieved_ds_to+'_post_reranked')
    print(f"Saved ds to {args.save_retrieved_ds_to+'_post_reranked'}")

    with open(f"{output_dir}/reranked_retrieval_report.json", 'w') as f:
        json.dump(with_rerank_res, f)
    print("Saved to", f"{output_dir}/retrieval_report.json")

