import sys
sys.path.append('./src')

from tqdm import tqdm
from vqa_datasets import load_vqa_dataset
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
import random
import copy
import os
import json
import pandas as pd

"""
- The training data for SFT is taken from the "pos_item_ids" and the "pos_item_contents" fields in the "train" split of E-VQA's "data" subset. 
  - Input: the image, the question, and instructions to generate a Wikipedia document about the entity shown in the image that is useful for answering the question. 
  - Output: the ground-truth wikipedia document formatted as "Title: <GT title>\nContent: <GT content>". The content is truncated to the first 50 space-delimited words. 
  - This yields 167k training examples, which we subsample randomly to 50k. 
"""

def convert_to_sharegpt_format(dataset, output_path, mode='sft'):
    """
    Convert dataset items to ShareGPT format and save as a JSON file.
    Args:
        dataset: List of processed dataset items (from train, validation, or test).
        output_path: Path to save the ShareGPT formatted JSON.
    """
    sharegpt_data = []
    print(f"Convert to sharegpt format for {mode} training...")
    for item in dataset:
        if mode == 'sft':
            conversation = {
                "messages": [
                    {"content": f"<image> {item['answer_input']}", "role": "user"},
                    {"content": item['answer_target'], "role": "assistant"}
                ],
                "images": [item['img_path']]
            }
            sharegpt_data.append(conversation)
        elif mode == 'dpo':
            instance = {
                "conversations": [
                  {
                    "from": "human",
                    "value": f"<image> {item['answer_input']}",
                  }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": item['chosen_answer_target'],
                },
                "rejected": {
                    "from": "gpt",
                    "value": item['rejected_answer_target'],
                },
                "images": [item['img_path']],
            }
            sharegpt_data.append(instance)
        else:
            raise NotImplementedError(f"convert_to_sharegpt_format {mode}")

    # Save to JSON file in ShareGPT format
    with open(output_path, 'w') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=4)

def process_data(df, img_basedir=None, sample_size=0, mode='sft'):
    def add_prefix(row):
        row['img_path'] = f"{img_basedir}/{row['img_path']}"
        return row
    
    if sample_size > 0:
        df = df.iloc[random.sample(range(len(dataset)), k=sample_size)]
        print(f"taking {sample_size} random samples")
    
    def process_item_from_df_sft(item):
        example = {
            'img_path': item.img_path,
            'answer_input': item.prompt,
            'answer_target': item.chosen,
        }
        return [example]
    
    def process_item_from_df_dpo(item):
        example = {
            'img_path': item.img_path,
            'answer_input': item.prompt,
            'chosen_answer_target': item.chosen,
            'rejected_answer_target': item.rejected,
        }
        return [example]
    
    def process_item_from_df_attn_sft(item):
        example = {
            'img_path': item.img_path,
            'answer_input': item.prompt,
            'answer_target': item.chosen,
            'gt_evidence_labels': item.gt_evidence_labels,
        }
        breakpoint()
        return [example]

    
    examples = []
    for item in tqdm(df.itertuples(), desc='processing'):
        if mode == 'sft':
            examples.extend(process_item_from_df_sft(item))
        elif mode == 'dpo':
            examples.extend(process_item_from_df_dpo(item))
        elif mode == 'attn_sft':
            examples.extend(process_item_from_df_attn_sft(item))
        else:
            raise NotImplementedError()

    return examples


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csvfile", type=str, required=True)
    parser.add_argument("--img_basedir", type=str, default=None)
    parser.add_argument("--mode", type=str, required=True, help='["sft", "dpo", "attn_sft"]')
    parser.add_argument("--output_dir", type=str, required=True, help='["sft", "dpo"]')
    parser.add_argument("--sample_size_train", type=int, default=0)
    parser.add_argument("--report_token_length", action='store_true')
    parser.add_argument("--drop_max_tokens", type=int, default=0)

    args = parser.parse_args()
    # random.seed(args.seed)

    # evqa_dataset = load_vqa_dataset('EVQA', split='train', img_basedir=args.img_basedir)

    # Randomly subsample 50k from the 167k positive items
    # Load, process, and sample the dataset splits
    df = pd.read_csv(args.input_csvfile)
    train_data = process_data(df, img_basedir=args.img_basedir, sample_size=args.sample_size_train, mode=args.mode)

    if args.report_token_length or args.drop_max_tokens > 0:
        from transformers import AutoTokenizer
        import numpy as np
        tokenizer = AutoTokenizer.from_pretrained("QWen/QWen2-VL-2B-Instruct")
        if args.mode in ['sft', 'attn_sft']:
            texts_to_tokenizer = [item['answer_input']+item['answer_target'] for item in train_data]
        elif args.mode == 'dpo':
            texts_to_tokenizer = [item['answer_input']+item['chosen_answer_target'] for item in train_data]
        else:
            raise NotImplementedError("mode")
        tokenized_texts = tokenizer(texts_to_tokenizer, padding=False, truncation=False, return_length=True)
        # Get the token lengths for each instance
        token_lengths = tokenized_texts['length']

        # Calculate statistics
        avg_token_length = np.mean(token_lengths)
        min_token_length = np.min(token_lengths)
        max_token_length = np.max(token_lengths)

        # Drop examples with token length > args.drop_max_tokens, if specified
        if args.drop_max_tokens > 0:
            filtered_data = [train_data[i] for i, length in enumerate(token_lengths) if length <= args.drop_max_tokens]
            print(f"Filtered out {len(train_data) - len(filtered_data)} ({(len(train_data)-len(filtered_data))/len(train_data)*100:.1f}%) instances exceeding {args.drop_max_tokens} tokens.")
            train_data = filtered_data

        # Count instances with token length > 3000
        # num_exceeding_3000 = sum(length > 3000 for length in token_lengths)

        # Print results
        print(f"Average token length: {avg_token_length}")
        print(f"Minimum token length: {min_token_length}")
        print(f"Maximum token length: {max_token_length}")
        # print(f"Number of training instances with token length > 3000: {num_exceeding_3000}")

    # train_data = load_and_process_split(args.input_dataset, args.img_basedir, 'train', args.sample_size_train, prompt_template)
    # val_data = load_and_process_split('valid', args.img_basedir, args.sample_size_eval, rewrite_prompt_template)
    # test_data = load_and_process_split('test', args.img_basedir, args.sample_size_eval, rewrite_prompt_template)

    # Convert each split to a Hugging Face Dataset
    os.makedirs(args.output_dir, exist_ok=True)
    # if args.save_as_hf_dataset:
    #     dataset_dict = DatasetDict({
    #         "train": Dataset.from_list(train_data),
    #         # "validation": Dataset.from_list(val_data),
    #         # "test": Dataset.from_list(test_data)
    #     })

    #     # Save the dataset to the specified directory
    #     dataset_dict.save_to_disk(args.output_dir)

    #     print(f"Datasets saved to {args.output_dir}")

    # Save ShareGPT-format JSONs for train, validation, and test splits
    convert_to_sharegpt_format(train_data, os.path.join(args.output_dir, "train_sharegpt.json"), mode=args.mode)
    # convert_to_sharegpt_format(val_data, os.path.join(args.output_dir, "val_sharegpt.json"))
    # convert_to_sharegpt_format(test_data, os.path.join(args.output_dir, "test_sharegpt.json"))

    print("ShareGPT format datasets saved to JSON files in", args.output_dir)