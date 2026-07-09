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

def convert_to_sharegpt_format(dataset_gen, output_path):
    """
    Convert dataset items to ShareGPT format and save as a JSON file incrementally.
    Args:
        dataset_gen: Generator of processed dataset items (from train, validation, or test).
        output_path: Path to save the ShareGPT formatted JSON.
    """
    with open(output_path, 'w') as f:
        f.write("[\n")  # Begin the JSON array
        first_item = True
        for item in dataset_gen:
            conversation = {
                "messages": [
                    {"content": f"<image> {item['answer_input']}", "role": "user"},
                    {"content": item['answer_target'], "role": "assistant"}
                ],
                "images": [item['img_path']]
            }
            if not first_item:
                f.write(",\n")  # Add a comma before subsequent items
            json.dump(conversation, f, ensure_ascii=False, indent=4)
            first_item = False
        f.write("\n]\n")  # Close the JSON array

def process_item(item, prompt_template):
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
    # gt_title = item['pos_item_ids'][0].split('_')[1] 
    # gt_content = item['pos_item_contents'][0]
    prompt = prompt_template.replace('<<QUESTION>>', item['question'])

    input_text = prompt
    output_text = item['gold_answer']

    new_item = {k: v for k, v in item.items()}
    new_item.update({
        'answer_input': input_text,
        'answer_target': output_text,
    })
    
    return new_item

def load_and_process_split(dataset, split_name, img_basedir, sample_size, rewrite_prompt_template):
    """
    Loads and processes a dataset split, sampling if necessary, optimized for large datasets.
    Args:
        dataset: The dataset object to process.
        split_name: The name of the dataset split, e.g., 'train', 'validation', 'test'.
        img_basedir: The base directory for image data.
        sample_size: The number of items to sample.
        rewrite_prompt_template: Template used for processing items.

    Returns:
        A generator of processed dataset items.
    """
    def process_dataset_item(item):
        """Helper function to process a single dataset item."""
        return process_item(item, rewrite_prompt_template)

    # Streamlined sampling if sample_size > 0
    indices = None
    if sample_size > 0:
        indices = random.sample(range(len(dataset)), k=sample_size)

    # Create a generator to process items lazily
    def processed_generator():
        for idx, item in enumerate(dataset):
            if indices and idx not in indices:
                continue
            yield process_dataset_item(item)

    # Wrapping generator with tqdm for progress tracking
    return tqdm(processed_generator(), desc=f'Processing split={split_name}', total=sample_size if sample_size > 0 else len(dataset))



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_basedir", type=str, default='data')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_size_train", type=int, default=-1, help="Number of samples for training subset. -1 for all data")
    parser.add_argument("--sample_size_eval", type=int, default=128, help="Number of samples for validation and test subsets.")
    parser.add_argument("--output_dir", type=str, default='outputs/', help="Directory to save the HF dataset.")
    parser.add_argument("--rewrite_prompt_template_file", type=str, required=True)
    parser.add_argument("--save_as_hf_dataset", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default='EVQA')

    args = parser.parse_args()
    random.seed(args.seed)

    with open(args.rewrite_prompt_template_file, 'r') as f:
        rewrite_prompt_template = f.read()

    vqa_dataset = load_vqa_dataset(args.dataset_name, split='train', img_basedir=args.img_basedir)

    # Randomly subsample 50k from the 167k positive items
    # Load, process, and sample the dataset splits
    # Load, process, and sample the dataset splits
    train_data_gen = load_and_process_split(vqa_dataset, 'train', args.img_basedir, args.sample_size_train, rewrite_prompt_template)
    if args.dataset_name == 'Infoseek':
        val_data_gen = load_and_process_split(vqa_dataset, 'test', args.img_basedir, args.sample_size_eval, rewrite_prompt_template)
    else:
        val_data_gen = load_and_process_split(vqa_dataset, 'valid', args.img_basedir, args.sample_size_eval, rewrite_prompt_template)
    test_data_gen = load_and_process_split(vqa_dataset, 'test', args.img_basedir, args.sample_size_eval, rewrite_prompt_template)
    # Convert each split to a Hugging Face Dataset
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_as_hf_dataset:
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })

        # Save the dataset to the specified directory
        dataset_dict.save_to_disk(args.output_dir)

        print(f"Datasets saved to {args.output_dir}")

    # Save ShareGPT-format JSONs for train, validation, and test splits
    # Save train data
    train_data_path = os.path.join(args.output_dir, "train_sharegpt.json")
    convert_to_sharegpt_format(list(train_data_gen), train_data_path)

    # Save validation data
    val_data_path = os.path.join(args.output_dir, "val_sharegpt.json")
    convert_to_sharegpt_format(list(val_data_gen), val_data_path)

    # Save test data
    test_data_path = os.path.join(args.output_dir, "test_sharegpt.json")
    convert_to_sharegpt_format(list(test_data_gen), test_data_path)

    print("ShareGPT format datasets saved to JSON files in", args.output_dir)