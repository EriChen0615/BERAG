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
                {"content": f"<image> {item['answer_input']}", "role": "user"},
                {"content": item['answer_target'], "role": "assistant"}
            ],
            "images": [item['img_path']]
        }
        sharegpt_data.append(conversation)

    # Save to JSON file in ShareGPT format
    with open(output_path, 'w') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=4)

def truncate_doc(doc_content, wlen=512):
    return ' '.join(doc_content.split(' ')[:wlen])

def process_item(item, prompt_template, pid_to_content_map, wlen=512, use_passage_set_doc=False):
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
    k = random.randint(0, len(item['pos_item_ids'])-1)
    gt_doc_id = item['pos_item_ids'][k]
    gt_title = gt_doc_id.split('_')[1] if len(gt_doc_id.split('_')) > 1 else ""
    if not use_passage_set_doc:
        gt_content = item['pos_item_contents'][k].replace('\n', ' ')
    else:
        gt_content = pid_to_content_map[gt_doc_id].replace('\n', ' ')

    arr = [0, 1, 2, 3, 4]
    random.shuffle(arr)

    for idx in arr:
        retrieved_doc_id = item['retrieved_passage'][idx]['passage_id']
        if retrieved_doc_id != gt_doc_id:
            retrieved_doc_title = retrieved_doc_id.split('_')[1] if len(retrieved_doc_id.split('_')) > 1 else ""
            retrieved_doc_content = pid_to_content_map[retrieved_doc_id].replace('\n', ' ')
            break

    prompt = prompt_template.replace('<<QUESTION>>', item['question'])

    if gt_title != "":
        input_text_gtdoc = prompt + f'\n[EVIDENCE] Title={gt_title}; Content={truncate_doc(gt_content, wlen)}\n[ANSWER] '
    else:
        input_text_gtdoc = prompt + f'\n[EVIDENCE] Content={truncate_doc(gt_content, wlen)}\n[ANSWER] '
    if retrieved_doc_title != "":
        input_text_retdoc = prompt  + f'\n[EVIDENCE] Title={retrieved_doc_title}; Content={truncate_doc(retrieved_doc_content, wlen)}\n[ANSWER] '
    else:
        input_text_retdoc = prompt  + f'\n[EVIDENCE] Content={truncate_doc(retrieved_doc_content, wlen)}\n[ANSWER] '

    output_text = item['gold_answer']

    gtdoc_item = {k: v for k, v in item.items()}
    gtdoc_item.update({
        'answer_input': input_text_gtdoc,
        'answer_target': output_text,
    })

    retdoc_item = {k: v for k, v in item.items()}
    retdoc_item.update({
        'answer_input': input_text_retdoc,
        'answer_target': output_text
    })
    
    return [gtdoc_item, retdoc_item]

def load_and_process_split(input_dataset, img_basedir, doc_split, sample_size, prompt_template, wlen=512, use_passage_set_doc=False):
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
        print("Using passage set for EVQA")
    elif "OKVQA" in input_dataset:
        passage_set = load_dataset('BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR', 'OKVQA_passages', split=f"{doc_split}_passages")
        print("Using passage set for OKVQA")
    elif "InfoseekNew" in input_dataset:
        passage_set = load_dataset('Jingbiao/aravqa', 'InfoseekFull_passages', split=f"train_passages")
        print("Using passage set for InfoseekNew (Full-length Passages)")
    elif "Infoseek" in input_dataset:
        passage_set = load_dataset('BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR', 'Infoseek_passages', split=f"{doc_split}_passages")
        print("Using passage set for Infoseek")
    else:
        raise NotImplementedError("passage set undefined")
    pid_to_content_map = {}
    for row in passage_set:
        pid_to_content_map[row['passage_id']] = row['passage_content']

    
    
    pos_items = []
    for item in tqdm(dataset, desc='processing'):
        pos_items.extend(process_item(item, prompt_template, pid_to_content_map, wlen, use_passage_set_doc))

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
    parser.add_argument("--wlen", type=int, default=512, help="Maximum number of words to keep in truncated documents")
    parser.add_argument("--use_passage_set_doc", action="store_true", default=False, help="Whether to use passage set document content")

    args = parser.parse_args()
    random.seed(args.seed)

    print("Running with arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    with open(args.prompt_template_file, 'r') as f:
       prompt_template = f.read()

    # evqa_dataset = load_from_disk(args.input_dataset)
    # evqa_dataset = load_vqa_dataset('EVQA', split='train', img_basedir=args.img_basedir)

    # Randomly subsample 50k from the 167k positive items
    # Load, process, and sample the dataset splits
    train_data = load_and_process_split(args.input_dataset, args.img_basedir, 'train', args.sample_size_train, prompt_template, args.wlen, args.use_passage_set_doc)
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