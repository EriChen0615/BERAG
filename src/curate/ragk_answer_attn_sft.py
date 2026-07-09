import sys
sys.path.append('./src')
from datasets import load_from_disk
from vqa_datasets import load_passages
import argparse
import json
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import gc

VLM_PROMPT_FOR_VQA = (
    "Answer the question after [QUESTION] about the image."
    "A retriever has retrieved a relevant document for you and provided it after [EVIDENCE]."
    "Give your answer after [ANSWER] without explanations\n"
)
EVIDENCE_START_TOKEN = "<evidence_start>"
EVIDENCE_END_TOKEN = "<evidence_end>"
ATTN_SOURCE_START_TOKEN= "<attn_source_start>"
ATTN_SOURCE_END_TOKEN= "<attn_source_end>"
ATTN_CALI_START_TOKEN= "<attn_cali_start>"
ATTN_CALI_END_TOKEN= "<attn_cali_end>"
IMG_TOKEN_COUNTS = 400 # (512/28)^2 approx 400

def convert_to_sharegpt_format(dataset, output_dir, mode='sft'):
    """
    Convert dataset items to ShareGPT format and save as a JSON file.
    Args:
        dataset: HuggingFace dataset with processed items.
        output_path: Path to save the ShareGPT formatted JSON.
    """
    sharegpt_data = []
    print(f"Convert to sharegpt format for {mode} training...")
    
    # Process dataset directly without converting to list
    for item in tqdm(dataset, desc="Converting to ShareGPT format"):
        if mode == 'sft':
            conversation = {
                "messages": [
                    {
                        "content": f"<image> {item['prompt']}", 
                        "role": "user"
                    },
                    {
                        "content": f"[ANSWER] {item['gold_answer']}", 
                        "role": "assistant"
                    }
                ],
                "images": [item['img_path']],
                "gt_evidence_labels": item.get('gt_evidence_labels')
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
                "gt_evidence_labels": item.get('gt_evidence_labels')
            }
            sharegpt_data.append(instance)
        else:
            raise NotImplementedError(f"convert_to_sharegpt_format {mode}")

    # Save to JSON file in ShareGPT format
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train_sharegpt.json'), 'w') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=4)

def add_prefix_and_form_prompt(batch, args, pid_to_content_map):
    """Process a batch of data efficiently using HuggingFace batched processing"""
    batch_size = len(batch['img_path'])
    
    # Process image paths
    img_paths = [f"{args.img_basedir}/{p}" for p in batch['img_path']]
    
    # Pre-allocate prompts list for better performance
    prompts = [''] * batch_size
    gt_evidence_labels = [[0] * args.topk_docs for _ in range(batch_size)]
    
    # Process each item in the batch
    for idx in range(batch_size):
        # Build evidence parts efficiently
        evidence_parts = []
        for i in range(args.topk_docs):
            passage_dict = batch['retrieved_passage'][idx][i]
            if passage_dict['passage_id'] in batch['pos_item_ids'][idx]:
                gt_evidence_labels[idx][i] = 1
            if 'text' in passage_dict:
                text = passage_dict['text']
            else:
                text = pid_to_content_map[passage_dict['passage_id']]
            
            evidence_part = (
                "[EVIDENCE]"
                f"{EVIDENCE_START_TOKEN}"
                f"Title: {passage_dict['passage_id']}\t"
                f"Content: {text}"
                f"{EVIDENCE_END_TOKEN}\n"
            )
            evidence_parts.append(evidence_part)
        
        question_part = None
        if args.attn_calibration_span == 'question_token':
            question_part = f"\n{ATTN_CALI_START_TOKEN}[QUESTION]{ATTN_CALI_END_TOKEN}"
        else:
            question_part = f"\n[QUESTION] "
        if args.attn_source_span == 'question':
            question_part += f"{ATTN_SOURCE_START_TOKEN}{batch['question'][idx]}{ATTN_SOURCE_END_TOKEN}"
        else:
            question_part += f"{batch['question'][idx]}"

        prompt = VLM_PROMPT_FOR_VQA \
            + ''.join(evidence_parts) \
            + question_part

        prompts[idx] = prompt
    
    # Update batch with processed data
    batch['img_path'] = img_paths
    batch['prompt'] = prompts
    batch['gt_evidence_labels'] = gt_evidence_labels
    return batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_path", type=str, required=True, help='Must have `retrieved_passage` field')
    parser.add_argument("--passage_set_name", type=str, help='Used to load passages from hf dataset. [EVQA, InfoseekNew_FullPassage, etc.]')
    parser.add_argument("--mode", type=str, default='sft', choices=['sft', 'dpo'])
    parser.add_argument("--img_basedir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--report_token_length", action='store_true')
    parser.add_argument("--drop_max_tokens", type=int, default=0)
    parser.add_argument("--topk_docs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8, help='Number of workers for parallel processing')
    parser.add_argument("--batch_size", type=int, default=100, help='Batch size for dataset processing')
    parser.add_argument("--attn_calibration_span", type=str, default=None, choices=['question_token'])
    parser.add_argument("--attn_source_span", type=str, default=None, choices=['question'])
    args = parser.parse_args()

    print("Attention span settings:")
    print(f"  Attn calibration span: {args.attn_calibration_span}")
    print(f"  Attn source span: {args.attn_source_span}")

    print("Loading dataset...")
    ds = load_from_disk(args.hf_dataset_path)
    if args.sample_size > 0:
        ds = ds.shuffle(seed=args.seed).select(range(args.sample_size))
    print(f"Dataset loaded with {len(ds)} items")

    print("Loading passages...")
    passages, pid_to_content_map = load_passages(args.passage_set_name, split='train')
    print(f"Loaded {len(pid_to_content_map)} passages")
    
    # Only keep the specified fields in the dataset
    # For SFT mode, keep: 'img_path', 'prompt', 'gold_answer'
    # For DPO mode, keep: 'img_path', 'answer_input', 'chosen_answer_target', 'rejected_answer_target'
    if args.mode == 'sft':
        keep_fields = ['img_path', 'prompt', 'gold_answer', 'retrieved_passage', 'pos_item_ids', 'question']
    elif args.mode == 'dpo':
        keep_fields = ['img_path', 'answer_input', 'chosen_answer_target', 'rejected_answer_target', 'retrieved_passage', 'pos_item_ids', 'question']
    else:
        raise NotImplementedError(f"Field filtering not implemented for mode: {args.mode}")

    ds = ds.remove_columns([col for col in ds.column_names if col not in keep_fields])
    
    # Use HuggingFace's native batched processing with multiprocessing
    print("Processing dataset with batched operations...")
    
    # Create a partial function with the required arguments
    process_func = partial(add_prefix_and_form_prompt, args=args, pid_to_content_map=pid_to_content_map)
    
    # Process the dataset using HuggingFace's map with multiprocessing
    ds = ds.map(
        process_func,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        desc="Processing batches",
        load_from_cache_file=False  # Disable caching for fresh processing
    )
    
    # Free memory from passages
    del passages, pid_to_content_map
    gc.collect()

    # Token counting and filtering
    if args.drop_max_tokens > 0:
        print("Computing token counts...")
        
        # Import tokenizer for counting tokens
        from transformers import AutoTokenizer
        
        # Load tokenizer (you may need to adjust the model name based on your setup)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        
        def count_tokens_for_example(example):
            """Count tokens for a single example"""
            if args.mode == 'sft':
                # For SFT mode, count tokens in the full conversation
                user_content = example['prompt']
                assistant_content = f"[ANSWER] {example['gold_answer']}"
                full_text = user_content + assistant_content
            elif args.mode == 'dpo':
                # For DPO mode, count tokens in the input and both responses
                input_text = example['answer_input']
                chosen_text = example['chosen_answer_target']
                rejected_text = example['rejected_answer_target']
                full_text = input_text + chosen_text + rejected_text
            else:
                raise NotImplementedError(f"Token counting not implemented for mode: {args.mode}")
            
            # Count tokens without padding
            tokens = tokenizer.encode(full_text, add_special_tokens=True)
            token_count = len(tokens) + IMG_TOKEN_COUNTS  # add space for image tokens
            
            return {"token_count": token_count}
        
        # Count tokens for all examples (single worker, no batching)
        ds_with_tokens = ds.map(
            count_tokens_for_example,
            desc="Counting tokens",
            num_proc=1  # Single worker for speed
        )
        
        # Report token statistics
        token_counts = ds_with_tokens['token_count']
        print(f"Token count (with {IMG_TOKEN_COUNTS} for image) statistics:")
        print(f"  Total examples: {len(token_counts)}")
        print(f"  Mean tokens: {sum(token_counts) / len(token_counts):.1f}")
        print(f"  Median tokens: {sorted(token_counts)[len(token_counts) // 2]}")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Std tokens: {(sum([(x - sum(token_counts)/len(token_counts))**2 for x in token_counts]) / len(token_counts))**0.5:.1f}")
        
        # Filter examples that exceed max tokens
        print(f"Filtering examples with > {args.drop_max_tokens} tokens...")
        original_count = len(ds_with_tokens)
        
        # Filter dataset with batching and multiple workers for speed
        # Use batched=True and set num_proc to a higher value (e.g., 8)
        def filter_by_token_count(batch):
            # batch['token_count'] is a list of token counts
            return [tc <= args.drop_max_tokens for tc in batch['token_count']]

        ds_filtered = ds_with_tokens.filter(
            filter_by_token_count,
            batched=True,
            num_proc=8,  # Adjust this number based on your CPU cores
            desc="Filtering by token count"
        )
        
        filtered_count = len(ds_filtered)
        dropped_count = original_count - filtered_count
        
        print(f"Token filtering results:")
        print(f"  Original examples: {original_count}")
        print(f"  Filtered examples: {filtered_count}")
        print(f"  Dropped examples: {dropped_count} ({dropped_count/original_count*100:.1f}%)")
        
        # Update dataset to use filtered version
        ds = ds_filtered

    print("Converting to ShareGPT format...")
    convert_to_sharegpt_format(ds, args.output_dir, mode=args.mode)
    print(f"Processing complete! Output saved to {args.output_dir}")

    

    
