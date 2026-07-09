import sys
sys.path.append('./src')

import vlms
from vqa_datasets import load_vqa_dataset
from collections import defaultdict
import random
from tqdm import tqdm
import argparse
import wandb
from datetime import datetime
import os
import _jsonnet
import json
from pprint import pprint

def make_query_dataset_from_vqa_dataset(ds, shuffle=True, seed=0, take_n=256):
    random.seed(seed)
    indices = [i for i in range(len(ds))]
    if shuffle:
        random.seed(seed)
        chosen_indices = random.sample(indices, take_n)
    else:
        chosen_indices = indices[:take_n]
    ds = ds.select(chosen_indices)
    return ds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--img_basedir", type=str, default='data')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--config_file", type=str, default='config/config.jsonnet')
    parser.add_argument("--prompt_template_file", type=str, default='prompt_template_file')
    parser.add_argument("--output_dir", type=str, default='outputs')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_cases", nargs="+")

    args = parser.parse_args()

    # Initialize wandb
    if not args.debug:
        wandb.init(project="a-ravqa", name=args.exp_name, entity="byrne-lab")
 
    current_time = datetime.now().strftime("%Y%m%d-%H")
    output_dir = f"{args.output_dir}/{current_time}-{args.exp_name}"
    print("output_dir=",output_dir)
    os.makedirs(output_dir, exist_ok=True)
    test_schema = json.loads(_jsonnet.evaluate_file(args.config_file))
    if args.model_path is not None:
        test_schema['vlm_config']['model_path'] = args.model_path
    pprint(test_schema)

    with open(args.prompt_template_file, 'r') as f:
        prompt_template = f.read()

    vlm = getattr(vlms, test_schema['vlm_class'])(**test_schema['vlm_config'])

    vqa_dataset = load_vqa_dataset(args.dataset_name, split=args.split, img_basedir=args.img_basedir)
    query_dataset = make_query_dataset_from_vqa_dataset(vqa_dataset, seed=args.seed)

    if args.debug:
        query_dataset = query_dataset.select([i for i in range(3)])

    inference_result = defaultdict(list)
    for row in tqdm(query_dataset, desc='VLM inferencing'):
        input_text = prompt_template.replace("<<QUESTION>>", row['question'])
        response = vlm.generate_response(input_text, row['img_path'])
        inference_result['model_response'].append(response)
        if args.debug:
            print(response)
    
    result_df = query_dataset.to_pandas()
    for k, v in inference_result.items():
        result_df[k] = v

    result_df.to_csv(f"{output_dir}/model_responses.csv")
    print(f"Saved to {output_dir}!")

    if not args.debug:
        wandb.finish()