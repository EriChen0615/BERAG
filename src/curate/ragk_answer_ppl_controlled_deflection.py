import sys
sys.path.append('./src')
from datasets import load_from_disk, concatenate_datasets
from vqa_datasets import load_passages
import argparse
import json
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import gc
import numpy as np
import pandas as pd
import ast
import random
from collections import defaultdict

VLM_PROMPT_FOR_VQA = (
    "Answer the question after [QUESTION] about the image."
    "A retriever has retrieved a relevant document for you and provided it after [EVIDENCE]."
    "Give your answer after [ANSWER] without explanations\n"
    "[EVIDENCE] <<<EVIDENCE>>>"
)
VLM_PROMPT_FOR_PRIOR = (
    "Please determine whether the document provided after '[EVIDENCE]' satisfies the following criteria:\n"
    "1. contains information about the entity shown in the image.\n"
    "2. provides useful information for answering the question shown after [Question].\n"
    "You should generate either 'yes' or 'no' after 'DECISION:'. You should generate 'yes' only when all criteria are met."
    "[Question] <<<QUESTION>>>\n"
    "[EVIDENCE] <<<EVIDENCE>>>\n"
    "DECISION:"
)

MAX_PASSAGE_WORD_COUNT = 2048
IMG_TOKEN_COUNTS = 400 # (512/28)^2 approx 400

def convert_to_sharegpt_format(dataset, output_dir, mode='sft', args=None):
    """
    Convert dataset items to ShareGPT format and save as a JSON file.
    Args:
        dataset: HuggingFace dataset with processed items.
        output_path: Path to save the ShareGPT formatted JSON.
        mode: 'sft' or 'bepo'
        args: arguments object
    """
    sharegpt_data = []
    print(f"Convert to sharegpt format for {mode} training...")
    
    # Process dataset directly without converting to list
    for item in tqdm(dataset, desc="Converting to ShareGPT format"):
        if mode == 'sft':
            # Validate and normalize passages format
            passages = item['passages']
            if passages:
                # Check if passages are in dict format
                if isinstance(passages[0], dict):
                    # Ensure all passages are dicts with 'text' key
                    normalized_passages = []
                    for idx, p in enumerate(passages):
                        if isinstance(p, dict):
                            if "text" not in p:
                                raise ValueError(f"Passage at index {idx} is a dict but missing 'text' key. Passage: {p}")
                            if not isinstance(p["text"], str):
                                raise TypeError(f"Passage at index {idx} has 'text' key but value is not a string. Got type: {type(p['text'])}, value: {p['text']}")
                            normalized_passages.append(p)
                        else:
                            # Convert non-dict to dict format
                            normalized_passages.append({"text": str(p)})
                    passages = normalized_passages
                else:
                    # Passages are strings, convert to dict format if needed
                    if args and args.passage_format == 'dict':
                        passages = [{"text": str(p)} for p in passages]
                    # Otherwise keep as strings
            
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
                "gt_passage_idx": item['gt_passage_idx'],
                "passages": passages,
                "passage_scores": item['passage_scores']
            }
            if args and args.add_deflection:
                conversation['deflection'] = item.get('deflection', 0)
            if args and args.add_separate_prompt_for_prior:
                conversation['prior_prompt'] = [
                    {
                        "content": f"<image> {item['prior_prompt'].replace('<<<QUESTION>>>', item['question'])}",
                        "role": "user"
                    },
                ]
        elif mode == 'bepo':
            conversation = {
                "messages": [
                    {
                        "from": "human",
                        "value": f"<image> {item['prompt']}", 
                    },
                ],
                "chosen": {
                    "from": "gpt",
                    "value": f"[ANSWER] {item['gold_answer']}",
                },
                "rejected": {
                    "from": "gpt",
                    "value": f"[ANSWER] {item['rejected_answers']}",
                },
                "images": [item['img_path']],
                "gt_passage_idx": item['gt_passage_idx'],
                "passages": item['passages'],
                "passage_scores": item['passage_scores']
            }
            if args and args.add_deflection:
                conversation['deflection'] = item.get('deflection', 0)
            if args and args.add_separate_prompt_for_prior:
                conversation['prior_prompt'] = [
                    {
                        "content": f"<image> {item['prior_prompt'].replace('<<<QUESTION>>>', item['question'])}",
                        "role": "user"
                    },
                ]
        else:
            raise NotImplementedError(f"convert_to_sharegpt_format {mode}")
        sharegpt_data.append(conversation)
    # Save to JSON file in ShareGPT format
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train_sharegpt.json'), 'w') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=4)

def _truncate_passage_content(passage_content, max_word_count=MAX_PASSAGE_WORD_COUNT):
    words = passage_content.split()
    if len(words) <= max_word_count:
        return passage_content
    return ' '.join(words[:max_word_count])

def add_prefix_and_form_prompt_controlled_deflection(batch, args, pid_to_content_map, deflection_target):
    """
    Process a batch of data with controlled deflection logic.
    
    Args:
        batch: Batch of data items
        args: Arguments object
        pid_to_content_map: Mapping from passage_id to content
        deflection_target: 0 or 1, determines processing logic
            - 0: Ensure GT passage is in topK
            - 1: Replace GT passage from topK with passage from [topK, top2K]
    """
    batch_size = len(batch['img_path'])
    
    # Process image paths
    img_paths = [f"{args.img_basedir}/{p}" for p in batch['img_path']]
    
    # Pre-allocate prompts list for better performance
    prompts = [''] * batch_size
    gt_passage_indices = [-1] * batch_size
    all_passage_contents = [''] * batch_size
    all_passage_scores = [-1] * batch_size
    all_gt_passage_idx = [-1] * batch_size
    
    # Process each item in the batch
    for idx in range(batch_size):
        gt_passage_id = batch['pos_item_ids'][idx][0]
        # Get top2K passages for potential replacement
        original_ret_passages = batch['retrieved_passage'][idx][:2*args.topk_docs]
        
        if deflection_target == 0:
            # For deflection=0: ensure GT passage is in topK (similar to --ensure_gt_passage_in_topk)
            # Get all retrieved passages (not limited to top2K, use all available)
            all_ret_passages = batch['retrieved_passage'][idx]
            topk_passages = all_ret_passages[:args.topk_docs] if len(all_ret_passages) >= args.topk_docs else all_ret_passages
            
            # Ensure GT is in topK (insert if not present, following original script logic)
            if gt_passage_id not in [p['passage_id'] for p in topk_passages]:
                # GT not in topK, add it (using minimum score from existing passages)
                gt_passage_dict = {
                    'passage_id': gt_passage_id,
                    'score': np.min([p['score'] for p in topk_passages]) if topk_passages else 0.0
                }
                passages = [gt_passage_dict] + topk_passages[:args.topk_docs-1]
            else:
                # GT already in topK, use topK as-is
                passages = topk_passages
            
            # Ensure we have exactly topK passages
            passages = passages[:args.topk_docs]
            
        else:  # deflection_target == 1
            # For deflection=1: replace GT from topK with passage from [topK, top2K]
            topk_passages = original_ret_passages[:args.topk_docs]
            top2k_passages = original_ret_passages[args.topk_docs:2*args.topk_docs]
            
            # Check if GT is in topK
            gt_in_topk = gt_passage_id in [p['passage_id'] for p in topk_passages]
            
            if gt_in_topk:
                # Find GT index in topK
                gt_idx_in_topk = [p['passage_id'] for p in topk_passages].index(gt_passage_id)
                # Find replacement from top2K (non-GT passage)
                replacement = None
                for p in top2k_passages:
                    if p['passage_id'] != gt_passage_id:
                        replacement = p
                        break
                
                if replacement:
                    # Replace GT with replacement
                    passages = topk_passages.copy()
                    passages[gt_idx_in_topk] = replacement
                else:
                    # Fallback: remove GT from topK and pad from top2K
                    passages = [p for p in topk_passages if p['passage_id'] != gt_passage_id]
                    # Pad to topK if needed
                    top2k_remaining = [p for p in top2k_passages if p['passage_id'] != gt_passage_id]
                    while len(passages) < args.topk_docs and len(top2k_remaining) > 0:
                        passages.append(top2k_remaining.pop(0))
                    # If still not enough, keep what we have (shouldn't happen in practice)
                    if len(passages) < args.topk_docs:
                        # Pad with duplicates if necessary (edge case)
                        while len(passages) < args.topk_docs:
                            passages.append(passages[0] if passages else topk_passages[0])
            else:
                # GT not in topK, use topK as-is
                passages = topk_passages
        
        # Ensure we have exactly topK passages
        passages = passages[:args.topk_docs]
        
        # Calculate gt_passage_idx after processing
        retrieved_passage_ids = [p['passage_id'] for p in passages]
        if gt_passage_id in retrieved_passage_ids:
            gt_passage_idx = retrieved_passage_ids.index(gt_passage_id)
        else:
            gt_passage_idx = -1
        
        # Add z0 passage if enabled
        if args.add_z0:
            # Calculate z0 score as mean of existing passage scores
            z0_score = np.mean([p['score'] for p in passages]) if passages else 0.0
            # Create z0 passage
            z0_passage = {
                'passage_id': 'z0',
                'score': z0_score
            }
            # Append z0 to passages list
            passages.append(z0_passage)
            
            # Update gt_passage_idx based on deflection_target
            if deflection_target == 1:
                # For deflection=1, z0 becomes the GT passage (last index)
                gt_passage_idx = len(passages) - 1
            # For deflection=0, keep original gt_passage_idx (z0 is not GT)
        
        # Format passages based on passage_format argument
        passage_contents = []
        for p in passages:
            if p['passage_id'] == 'z0':
                # Format z0 passage with text=" "
                if args.passage_format == 'dict':
                    passage_contents.append({"text": " "})
                else:
                    passage_contents.append(" ")
            else:
                # Format regular passages
                if args.passage_format == 'dict':
                    passage_contents.append({"text": f"Title: {p['passage_id']}\tContent: {_truncate_passage_content(pid_to_content_map[p['passage_id']])}"})
                else:
                    passage_contents.append(f"Title: {p['passage_id']}\tContent: {_truncate_passage_content(pid_to_content_map[p['passage_id']])}")
        passage_scores = [p['score'] for p in passages]
        
        question_part = f"\n[QUESTION] {batch['question'][idx]}"
        prompt = VLM_PROMPT_FOR_VQA + question_part
        
        prompts[idx] = prompt
        all_passage_contents[idx] = passage_contents
        all_passage_scores[idx] = passage_scores
        all_gt_passage_idx[idx] = gt_passage_idx
        
        # Set deflection label based on target
        deflection = deflection_target
        if 'deflection' not in batch:
            batch['deflection'] = []
        batch['deflection'].append(deflection)
    
    # Update batch with processed data
    batch['img_path'] = img_paths
    batch['prompt'] = prompts
    batch['gt_passage_idx'] = all_gt_passage_idx
    batch['passages'] = all_passage_contents
    batch['passage_scores'] = all_passage_scores
    
    if args.add_separate_prompt_for_prior:
        prior_prompts = [VLM_PROMPT_FOR_PRIOR for _ in range(batch_size)]
        batch['prior_prompt'] = prior_prompts
    
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_path", type=str, required=True, help='Must have `retrieved_passage` field')
    parser.add_argument("--passage_set_name", type=str, help='Used to load passages from hf dataset. [EVQA, InfoseekNew_FullPassage, etc.]')
    parser.add_argument("--mode", type=str, default='sft', choices=['sft', 'bepo'])
    parser.add_argument("--img_basedir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--report_token_length", action='store_true')
    parser.add_argument("--drop_max_tokens", type=int, default=0)
    parser.add_argument("--topk_docs", type=int, default=2, help='Number of top passages to use (default: 2)')
    parser.add_argument("--random_sample_1passage_from_topk", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8, help='Number of workers for parallel processing')
    parser.add_argument("--batch_size", type=int, default=100, help='Batch size for dataset processing')
    parser.add_argument("--deflection_ratio", type=float, default=0.5, help='Ratio of deflection=1 samples (default: 0.5)')
    parser.add_argument("--model_sample_filepath", type=str, default=None, help="Path to base model's samples")
    parser.add_argument("--add_separate_prompt_for_prior", action='store_true')
    parser.add_argument("--prior_prompt", type=str, default=None)
    parser.add_argument("--pairwise_data_mode", type=str, default='selfgen')
    parser.add_argument("--bepo_mode", type=str, default='find_wrong_from_all_responses')
    parser.add_argument("--add_deflection", action='store_true', default=True, help='Add deflection labels (always enabled for controlled deflection)')
    parser.add_argument("--passage_format", type=str, default='text', choices=['text', 'dict'], help='Format for passages: text (string) or dict (with "text" key)')
    parser.add_argument("--add_z0", action='store_true', help='Add z0 passage (empty text passage) to each example. For deflection=0, z0 is non-GT. For deflection=1, z0 is GT.')
    args = parser.parse_args()
    
    # Force add_deflection to True for controlled deflection script
    args.add_deflection = True
    
    print("Loading dataset...")
    ds = load_from_disk(args.hf_dataset_path)
    if args.sample_size > 0 or args.sample_offset > 0:
        print(f"Sampling {args.sample_size} items from {args.sample_offset} to {args.sample_offset + args.sample_size}")
        ds = ds.shuffle(seed=args.seed).select(range(args.sample_offset, min(args.sample_offset + args.sample_size, len(ds))))
    print(f"Dataset loaded with {len(ds)} items")
    
    print("Loading passages...")
    _, pid_to_content_map = load_passages(args.passage_set_name, split='train')
    print(f"Loaded {len(pid_to_content_map)} passages")
    
    # Only keep the specified fields in the dataset
    keep_fields = ['question_id', 'img_path', 'prompt', 'gold_answer', 'retrieved_passage', 'pos_item_ids', 'question']
    ds = ds.remove_columns([col for col in ds.column_names if col not in keep_fields])
    
    # Calculate sample counts based on deflection_ratio
    total_samples = args.sample_size if args.sample_size > 0 else len(ds)
    deflection_0_count = int(total_samples * (1 - args.deflection_ratio))
    deflection_1_count = total_samples - deflection_0_count
    
    print(f"Target sample distribution:")
    print(f"  Total samples: {total_samples}")
    print(f"  Deflection=0 samples: {deflection_0_count} ({100*(1-args.deflection_ratio):.1f}%)")
    print(f"  Deflection=1 samples: {deflection_1_count} ({100*args.deflection_ratio:.1f}%)")
    
    # Process deflection=0 group
    print("\nProcessing deflection=0 group...")
    # For deflection=0, process all data (no pre-filtering needed)
    # GT will be inserted if not in topK (similar to --ensure_gt_passage_in_topk)
    ds_deflection_0 = ds.map(
        partial(add_prefix_and_form_prompt_controlled_deflection,
                args=args, pid_to_content_map=pid_to_content_map, deflection_target=0),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        desc="Processing deflection=0 samples",
        load_from_cache_file=False
    )
    
    # Verify all have deflection=0 and GT in topK
    def verify_deflection_0(batch):
        results = []
        for idx in range(len(batch.get('deflection', []))):
            deflection = batch['deflection'][idx]
            gt_passage_idx = batch['gt_passage_idx'][idx]
            passages = batch.get('passages', [])
            num_passages = len(passages[idx]) if isinstance(passages, list) and idx < len(passages) else 0
            
            # Should have deflection=0 and gt_passage_idx != -1
            # If add_z0 is enabled, gt_passage_idx should not point to z0 (last index)
            if args.add_z0 and num_passages > 0:
                # z0 is at the last index, so gt_passage_idx should be < num_passages - 1
                is_valid = (deflection == 0) and (gt_passage_idx != -1) and (gt_passage_idx < num_passages - 1)
            else:
                # Without z0, just check that gt_passage_idx != -1
                is_valid = (deflection == 0) and (gt_passage_idx != -1)
            results.append(is_valid)
        return results
    
    ds_deflection_0 = ds_deflection_0.filter(
        verify_deflection_0,
        batched=True,
        desc="Verifying deflection=0 samples"
    )
    print(f"  After verification: {len(ds_deflection_0)} valid deflection=0 samples")
    
    # Sample deflection_0_count
    if len(ds_deflection_0) > deflection_0_count:
        ds_deflection_0 = ds_deflection_0.shuffle(seed=args.seed).select(range(deflection_0_count))
        print(f"  Sampled {deflection_0_count} deflection=0 samples")
    else:
        print(f"  Warning: Only {len(ds_deflection_0)} deflection=0 samples available (requested {deflection_0_count})")
    
    # Process deflection=1 group
    print("\nProcessing deflection=1 group...")
    ds_deflection_1 = ds.map(
        partial(add_prefix_and_form_prompt_controlled_deflection,
                args=args, pid_to_content_map=pid_to_content_map, deflection_target=1),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        desc="Processing deflection=1 samples",
        load_from_cache_file=False
    )
    
    # Filter to only deflection=1 samples (GT not in topK, or z0 is GT if add_z0)
    def verify_deflection_1(batch):
        results = []
        for idx in range(len(batch.get('deflection', []))):
            deflection = batch['deflection'][idx]
            gt_passage_idx = batch['gt_passage_idx'][idx]
            passages = batch.get('passages', [])
            num_passages = len(passages[idx]) if isinstance(passages, list) and idx < len(passages) else 0
            
            if args.add_z0:
                # If add_z0 is enabled, deflection=1 should have gt_passage_idx pointing to z0 (last index)
                # z0 is at index (num_passages - 1)
                expected_z0_idx = num_passages - 1 if num_passages > 0 else -1
                is_valid = (deflection == 1) and (gt_passage_idx == expected_z0_idx)
            else:
                # Without z0, deflection=1 should have gt_passage_idx == -1 (GT not in topK)
                is_valid = (deflection == 1) and (gt_passage_idx == -1)
            results.append(is_valid)
        return results
    
    ds_deflection_1 = ds_deflection_1.filter(
        verify_deflection_1,
        batched=True,
        desc="Verifying deflection=1 samples"
    )
    print(f"  After verification: {len(ds_deflection_1)} valid deflection=1 samples")
    
    # Sample deflection_1_count
    if len(ds_deflection_1) > deflection_1_count:
        ds_deflection_1 = ds_deflection_1.shuffle(seed=args.seed).select(range(deflection_1_count))
        print(f"  Sampled {deflection_1_count} deflection=1 samples")
    else:
        print(f"  Warning: Only {len(ds_deflection_1)} deflection=1 samples available (requested {deflection_1_count})")
    
    # Merge both groups
    print("\nMerging deflection=0 and deflection=1 groups...")
    ds = concatenate_datasets([ds_deflection_0, ds_deflection_1])
    ds = ds.shuffle(seed=args.seed)  # Final shuffle
    
    # Verify final distribution
    deflection_counts = {}
    for item in ds:
        deflection = item.get('deflection', 0)
        deflection_counts[deflection] = deflection_counts.get(deflection, 0) + 1
    
    print(f"Final dataset distribution:")
    print(f"  Total samples: {len(ds)}")
    for deflection_val in sorted(deflection_counts.keys()):
        count = deflection_counts[deflection_val]
        ratio = count / len(ds) * 100
        print(f"  Deflection={deflection_val}: {count} ({ratio:.1f}%)")
    
    # Free memory from passages
    del pid_to_content_map
    gc.collect()
    
    # Token counting and filtering
    if args.drop_max_tokens > 0:
        print("\nComputing token counts...")
        
        # Import tokenizer for counting tokens
        from transformers import AutoTokenizer
        
        # Load tokenizer (you may need to adjust the model name based on your setup)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        
        def count_tokens_for_example(example):
            """Count tokens for a single example"""
            if args.mode in ['sft', 'bepo']:
                # For SFT mode, count tokens in the full conversation
                # Handle both text and dict formats for passages
                passages = example['passages']
                passage_texts = []
                for p in passages:
                    if isinstance(p, dict):
                        if "text" not in p:
                            raise ValueError(f"Passage is a dict but missing 'text' key. Passage: {p}")
                        text = p["text"]
                        if not isinstance(text, str):
                            raise TypeError(f"Passage has 'text' key but value is not a string. Got type: {type(text)}, value: {text}")
                        passage_texts.append(text)
                    else:
                        # If it's not a dict, convert to string
                        passage_texts.append(str(p))
                
                if not passage_texts:
                    raise ValueError("No passages found in example")
                
                passage_lengths = [len(text) for text in passage_texts]
                max_len_passage_idx = np.argmax(passage_lengths)
                passage_text = passage_texts[max_len_passage_idx]
                
                # Ensure passage_text is a string before using replace
                if not isinstance(passage_text, str):
                    raise TypeError(f"passage_text is not a string. Got type: {type(passage_text)}, value: {passage_text}")
                
                user_content = example['prompt'].replace('<<<EVIDENCE>>>', passage_text)
                assistant_content = f"[ANSWER] {example['gold_answer']}"
                full_text = user_content + assistant_content
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
            num_proc=8  # Single worker for speed
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
        def filter_by_token_count(batch):
            # batch['token_count'] is a list of token counts
            return [tc <= args.drop_max_tokens for tc in batch['token_count']]
        
        ds_filtered = ds_with_tokens.filter(
            filter_by_token_count,
            batched=True,
            num_proc=8,  # Adjust this number based on your CPU cores
            desc="Filtering by token count",
            batch_size=4096,
        )
        
        filtered_count = len(ds_filtered)
        dropped_count = original_count - filtered_count
        
        print(f"Token filtering results:")
        print(f"  Original examples: {original_count}")
        print(f"  Filtered examples: {filtered_count}")
        print(f"  Dropped examples: {dropped_count} ({dropped_count/original_count*100:.1f}%)")
        
        # Update dataset to use filtered version
        ds = ds_filtered
    
    print("\nConverting to ShareGPT format...")
    convert_to_sharegpt_format(ds, args.output_dir, mode=args.mode, args=args)
    print(f"Processing complete! Output saved to {args.output_dir}")

