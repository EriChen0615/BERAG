import sys
sys.path.append('./src')

import _jsonnet
from easydict import EasyDict
from pprint import pprint
import json
from tqdm import tqdm
import wandb 
import retrievers
import argparse
from datetime import datetime
import os
from vqa_datasets import load_vqa_dataset
from datasets import Dataset
import pandas as pd
from pprint import pprint
import ast

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--img_basedir", type=str, default='data')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ds_seed", type=int, default=0)
    parser.add_argument("--csv_filename", type=str, default=None)
    parser.add_argument("--query_field", type=str, default='question')
    parser.add_argument("--no_image_search", action="store_true")
    parser.add_argument("--use_doc_encoder_for_query", action="store_true")
    parser.add_argument("--do_sanity_check", action="store_true")
    parser.add_argument("--save_retrieved_ds_to", type=str, default=None)
    parser.add_argument("--reranker_name", type=str, default=None)
    parser.add_argument("--reranker_model_path", type=str, default=None)
    parser.add_argument("--rerank_topk", type=int, default=50)
    parser.add_argument("--take_n", type=int, default=0)

    parser.add_argument("--config_file", type=str, default='config/config.jsonnet')
    parser.add_argument("--output_dir", type=str, default='outputs')
    parser.add_argument("--no_instruction", action='store_true')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_cases", nargs="+")

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="a-ravqa", name=args.exp_name, entity="byrne-lab")
 
    current_time = datetime.now().strftime("%Y%m%d-%H")
    output_dir = f"{args.output_dir}/{current_time}-{args.exp_name}"
    print("output_dir=",output_dir)
    os.makedirs(output_dir, exist_ok=True)
    test_schema = json.loads(_jsonnet.evaluate_file(args.config_file))
    
    # When we want to do query rewritting evaluation, 
    # we need to load the dataset with overwrtten queries
    # Or we can just load the new set of queries 
    # and overwrite the questions in the dataset
    if args.csv_filename is None:
        vqa_dataset = load_vqa_dataset(args.dataset_name, split=args.split, img_basedir=args.img_basedir, take_n=args.take_n, seed=args.ds_seed)
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
            vqa_dataset = vqa_dataset.select([i for i in range(60)])

    retriever = getattr(retrievers, test_schema["retriever_class"])(
        **test_schema["retriever_config"],
        add_instruction=(not args.no_instruction),
    )

    #NOTE NOT USED. See reranker_inference.py
    if args.reranker_name is not None:
        # using reranker
        if args.reranker_name == 'EVQA_QWen2-VL-2B-LoRA_Reranker':
            from rerankers import QWen2Reranker
            reranker = QWen2Reranker(
                model_path=args.reranker_model_path,
                is_lora=true,
                base_modwl_path='QWen/QWen2-VL-2B-Instruct',
                processor_path='QWen/QWen2-VL-2B-Instruct',
                prompt_template_file='config/prompts/1111_doc1_verify.txt',
            )
            rerank_topk = args.rerank_topk
    else:
        reranker = None

    if args.do_sanity_check:
        all_passages_in_index = set(retriever.passage_ids)
        for row in tqdm(vqa_dataset, desc='sanity checking GT doc in index'):
            assert all([gtdoc in all_passages_in_index for gtdoc in row['pos_item_ids']]), "GT document not in index!"

    retrieval_report = retriever.query_and_evaluate_ds(
        vqa_dataset,
        query_field=args.query_field,
        no_image_search=args.no_image_search,
        use_doc_encoder_for_query=args.use_doc_encoder_for_query,
        save_retrieved_ds_to=args.save_retrieved_ds_to,
        **test_schema["query_and_evaluate_ds_kwargs"]
    )

    pprint(retrieval_report)
    with open(f"{output_dir}/retrieval_report.json", 'w') as f:
        json.dump(retrieval_report, f)
    print("Saved to", f"{output_dir}/retrieval_report.json")
    
    wandb.finish()

