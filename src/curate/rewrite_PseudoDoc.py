import sys
sys.path.append('./src')

from tqdm import tqdm
from vqa_datasets import load_vqa_dataset
from datasets import Dataset, DatasetDict
import random
import copy
import os
import json

"""
- The training data for SFT is taken from the "pos_item_ids" and the "pos_item_contents" fields in the "train" split of E-VQA's "data" subset. 
  - Input: the image, the question, and instructions to generate a Wikipedia document about the entity shown in the image that is useful for answering the question. 
  - Output: the ground-truth wikipedia document formatted as "Title: <GT title>\nContent: <GT content>". The content is truncated to the first 50 space-delimited words. 
  - This yields 167k training examples, which we subsample randomly to 50k. 
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
                {"content": f"<image> {item['query_rewrite_input']}", "role": "user"},
                {"content": item['query_rewrite_target'], "role": "assistant"}
            ],
            "images": [item['img_path']]
        }
        sharegpt_data.append(conversation)

    # Save to JSON file in ShareGPT format
    with open(output_path, 'w') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=4)

def process_item(item, rewrite_prompt_template):
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
    gt_title = item['pos_item_ids'][0].split('_')[1] 
    gt_content = item['pos_item_contents'][0]
    rewrite_prompt = rewrite_prompt_template.replace('<<QUESTION>>', item['question'])

    input_text = rewrite_prompt
    output_text = f"\n- TITLE: {gt_title}\n- CONTENT: {' '.join(gt_content.split()[:50])}"

    new_item = {k: v for k, v in item.items()}
    new_item.update({
        'query_rewrite_input': input_text,
        'query_rewrite_target': output_text,
    })
    
    return new_item

def load_and_process_split(split_name, img_basedir, sample_size, rewrite_prompt_template):
    """
    Loads and processes a dataset split, sampling if necessary.
    Args:
        split_name: The name of the dataset split, e.g., 'train', 'validation', 'test'.
        img_basedir: The base directory for image data.
        sample_size: The number of items to sample.

    Returns:
        A list of processed dataset items.
    """
    dataset = load_vqa_dataset('EVQA', split=split_name, img_basedir=img_basedir)
    dataset = dataset.select(random.sample(range(len(dataset)), k=sample_size))
    
    pos_items = [
        process_item(
           item,
           rewrite_prompt_template
        )
        for item in tqdm(dataset, desc=f'proceesing split={split_name}')
    ]
    # Downsample the dataset to the specified sample size
    return pos_items


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_basedir", type=str, default='data')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_size_train", type=int, default=50000, help="Number of samples for training subset.")
    parser.add_argument("--sample_size_eval", type=int, default=128, help="Number of samples for validation and test subsets.")
    parser.add_argument("--output_dir", type=str, default='outputs/', help="Directory to save the HF dataset.")
    parser.add_argument("--rewrite_prompt_template_file", type=str, required=True)

    args = parser.parse_args()
    random.seed(args.seed)

    with open(args.rewrite_prompt_template_file, 'r') as f:
        rewrite_prompt_template = f.read()

    evqa_dataset = load_vqa_dataset('EVQA', split='train', img_basedir=args.img_basedir)

    # Randomly subsample 50k from the 167k positive items
    # Load, process, and sample the dataset splits
    train_data = load_and_process_split('train', args.img_basedir, args.sample_size_train, rewrite_prompt_template)
    val_data = load_and_process_split('valid', args.img_basedir, args.sample_size_eval, rewrite_prompt_template)
    test_data = load_and_process_split('test', args.img_basedir, args.sample_size_eval, rewrite_prompt_template)

    # Convert each split to a Hugging Face Dataset
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })

    # Save the dataset to the specified directory
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)

    print(f"Datasets saved to {args.output_dir}")

    # Save ShareGPT-format JSONs for train, validation, and test splits
    convert_to_sharegpt_format(train_data, os.path.join(args.output_dir, "train_sharegpt.json"))
    convert_to_sharegpt_format(val_data, os.path.join(args.output_dir, "val_sharegpt.json"))
    convert_to_sharegpt_format(test_data, os.path.join(args.output_dir, "test_sharegpt.json"))

    print("ShareGPT format datasets saved to JSON files in", args.output_dir)