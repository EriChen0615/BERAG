import sys
sys.path.append('./src')

from tqdm import tqdm
from vqa_datasets import load_vqa_dataset
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
import random
import copy
import os
import json

"""
- Training data is curated as follows:
  1. For each question $$q$$, we retrieve the Top-50 documents $$z_1, z_2, ..., z_{50}$$ with PreFLMR-L.
  2. We form 2 training instances for each question, with the following inputs:
    1. Input: $$q+z_{gt}$$; Output: "yes"
    2. Input: $$q+z_i$$; Output: "no". $$z_i$$ is randomly selected from the Top-5 documents and is not identical to $$z_{gt}$$. 
  3. We cut the document to at most 512 space-delimited words.
  4. In training, the max number of tokens is set to 1024. 
  5. This yields 167K*2 = 334K training examples. Of these 167k training examples, 1% is held-out for validation. 
"""

def convert_to_sharegpt_format(dataset, output_path):
    """
    Convert dataset items to ShareGPT format and save as a JSON file.
    Args:
        dataset: List of processed dataset items (from train, validation, or test).
        output_path: Path to save the ShareGPT formatted JSON.
    """
    sharegpt_data = []
    for item in dataset:
        conversation = {
            "messages": [
                {"content": f"<image> {item['verify_input']}", "role": "user"},
                {"content": item['verify_output'], "role": "assistant"}
            ],
            "images": [item['img_path']]
        }
        sharegpt_data.append(conversation)

    # Save to JSON file in ShareGPT format
    with open(output_path, 'w') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=4)

def truncate_doc(doc_content, wlen=512):
    return ' '.join(doc_content.split(' ')[:wlen])

def process_item(item, prompt_template, pid_to_content_map):
    """
    Format each item into the required input-output format.
    Args:
        image: Image object or filepath.
        question: String, the VQA question.
        gt_title: Ground-truth title for the Wikipedia document.
        gt_content: Ground-truth content for the Wikipedia document.

    Returns:
        Dict with 'input' and 'output' keys.
    """
    gt_doc_id = item['pos_item_ids'][0]
    gt_title = gt_doc_id.split('_')[1] if len(gt_doc_id.split('_')) > 1 else ""
    gt_content = item['pos_item_contents'][0].replace('\n', ' ')

    arr = [i for i in range(50)]
    random.shuffle(arr)

    for idx in arr:
        retrieved_doc_id = item['retrieved_passage'][idx]['passage_id']
        if retrieved_doc_id != gt_doc_id:
            retrieved_doc_title = retrieved_doc_id.split('_')[1] if len(retrieved_doc_id.split('_')) > 1 else ""
            retrieved_doc_content = pid_to_content_map[retrieved_doc_id].replace('\n', ' ')
            break

    prompt = prompt_template.replace('<<QUESTION>>', item['question'])
    if gt_title != "":
        pos_doc = f"Title={gt_title}; Content={truncate_doc(gt_content)}"
    else:
        pos_doc = f"Content={truncate_doc(gt_content)}"

    if retrieved_doc_title != "":
        neg_doc = f"Title={retrieved_doc_title}; Content={truncate_doc(retrieved_doc_content)}"
    else:
        neg_doc = f"Content={truncate_doc(retrieved_doc_content)}"

    input_text_pos = prompt.replace('<<EVIDENCE>>', pos_doc)
    input_text_neg = prompt.replace('<<EVIDENCE>>', neg_doc)

    pos_item = {k: v for k, v in item.items()}
    pos_item.update({
        'verify_input': input_text_pos,
        'verify_output': "yes",
    })

    neg_item = {k: v for k, v in item.items()}
    neg_item.update({
        'verify_input': input_text_neg,
        'verify_output': "no"
    })
    
    return [pos_item, neg_item]

def load_and_process_split(input_dataset, img_basedir, doc_split, sample_size, prompt_template):
    """
    Loads and processes a dataset split, sampling if necessary.
    Args:
        split_name: The name of the dataset split, e.g., 'train', 'validation', 'test'.
        img_basedir: The base directory for image data.
        sample_size: The number of items to sample.

    Returns:
        A list of processed dataset items.
    """
    dataset = load_from_disk(input_dataset)
    def add_prefix(row):
        row['img_path'] = f"{img_basedir}/{row['img_path']}"
        return row

    if sample_size > 0:
        dataset = dataset.select(random.sample(range(len(dataset)), k=sample_size))
    dataset = dataset.map(add_prefix)

    if "EVQA" in input_dataset:
        passage_set = load_dataset('BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR', 'EVQA_passages', split=f"{doc_split}_passages")
    elif "OKVQA" in input_dataset:
        passage_set = load_dataset('BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR', 'OKVQA_passages', split=f"{doc_split}_passages")
    elif "InfoseekNew" in input_dataset:
        passage_set = load_dataset('Jingbiao/aravqa', 'Infoseek_passages', split=f"{doc_split}_passages")
    elif "Infoseek" in input_dataset:
        passage_set = load_dataset('BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR', 'Infoseek_passages', split=f"{doc_split}_passages")
    else:
        raise NotImplementedError("passage set undefined")
    # passage_set = load_dataset('BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR', 'EVQA_passages', split=f"{doc_split}_passages")
    pid_to_content_map = {}
    for row in passage_set:
        pid_to_content_map[row['passage_id']] = row['passage_content']

    
    
    pos_items = []
    for item in tqdm(dataset, desc='processing'):
        pos_items.extend(process_item(item, prompt_template, pid_to_content_map))

    # Downsample the dataset to the specified sample size
    return pos_items


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, required=True)
    parser.add_argument("--prompt_template_file", type=str, required=True)
    parser.add_argument("--img_basedir", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_size_train", type=int, default=-1, help="Number of samples for training subset. -1 for all data")
    parser.add_argument("--sample_size_eval", type=int, default=128, help="Number of samples for validation and test subsets.")
    parser.add_argument("--output_dir", type=str, default='outputs/', help="Directory to save the HF dataset.")
    parser.add_argument("--save_as_hf_dataset", type=int, default=0)

    args = parser.parse_args()
    random.seed(args.seed)

    with open(args.prompt_template_file, 'r') as f:
       prompt_template = f.read()

    # evqa_dataset = load_from_disk(args.input_dataset)
    # evqa_dataset = load_vqa_dataset('EVQA', split='train', img_basedir=args.img_basedir)

    # Randomly subsample 50k from the 167k positive items
    # Load, process, and sample the dataset splits
    train_data = load_and_process_split(args.input_dataset, args.img_basedir, 'train', args.sample_size_train, prompt_template)
    # val_data = load_and_process_split('valid', args.img_basedir, args.sample_size_eval, rewrite_prompt_template)
    # test_data = load_and_process_split('test', args.img_basedir, args.sample_size_eval, rewrite_prompt_template)

    # Convert each split to a Hugging Face Dataset
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_as_hf_dataset:
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            # "validation": Dataset.from_list(val_data),
            # "test": Dataset.from_list(test_data)
        })

        # Save the dataset to the specified directory
        dataset_dict.save_to_disk(args.output_dir)

        print(f"Datasets saved to {args.output_dir}")

    # Save ShareGPT-format JSONs for train, validation, and test splits
    convert_to_sharegpt_format(train_data, os.path.join(args.output_dir, "train_sharegpt.json"))
    # convert_to_sharegpt_format(val_data, os.path.join(args.output_dir, "val_sharegpt.json"))
    # convert_to_sharegpt_format(test_data, os.path.join(args.output_dir, "test_sharegpt.json"))

    print("ShareGPT format datasets saved to JSON files in", args.output_dir)