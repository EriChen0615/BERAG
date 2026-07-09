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
                    if args.passage_format == 'dict':
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
            if args.add_deflection:
                conversation['deflection'] = item.get('deflection', 0)
            if args.add_separate_prompt_for_prior:
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
            if args.add_deflection:
                conversation['deflection'] = item.get('deflection', 0)
            if args.add_separate_prompt_for_prior:
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

def add_prefix_and_form_prompt(batch, args, pid_to_content_map):
    """Process a batch of data efficiently using HuggingFace batched processing"""
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
        original_ret_passages = batch['retrieved_passage'][idx][:args.topk_docs]

        if args.random_sample_1passage_from_topk:
            if gt_passage_id in [p['passage_id'] for p in original_ret_passages]:
                gt_passage_idx = [p['passage_id'] for p in original_ret_passages].index(gt_passage_id)
                candidates = original_ret_passages[:gt_passage_idx] + original_ret_passages[gt_passage_idx+1:]
                passages = [random.choice(candidates)]
            else:
                passages = [random.choice(original_ret_passages)]
        else:
            passages = original_ret_passages

        if args.ensure_gt_passage_in_topk and gt_passage_id not in [p['passage_id'] for p in passages]: # if gt passage is not in the retrieved passage set, add it.
            gt_passage_dict = {
                'passage_id': gt_passage_id,
                'score': np.min([p['score'] for p in passages])
            }
            passages = [gt_passage_dict] + passages[:args.topk_docs-1]

        retrieved_passage_ids = [p['passage_id'] for p in passages]
        if gt_passage_id in retrieved_passage_ids:
            gt_passage_idx = retrieved_passage_ids.index(gt_passage_id)
        else:
            gt_passage_idx = -1

        # Format passages based on passage_format argument
        if args.passage_format == 'dict':
            passage_contents = [{"text": f"Title: {p['passage_id']}\tContent: {_truncate_passage_content(pid_to_content_map[p['passage_id']])}"} for p in passages]
        else:
            passage_contents = [f"Title: {p['passage_id']}\tContent: {_truncate_passage_content(pid_to_content_map[p['passage_id']])}" for p in passages]
        passage_scores = [p['score'] for p in passages]

        question_part = f"\n[QUESTION] {batch['question'][idx]}"
        prompt = VLM_PROMPT_FOR_VQA + question_part

        prompts[idx] = prompt
        all_passage_contents[idx] = passage_contents
        all_passage_scores[idx] = passage_scores
        all_gt_passage_idx[idx] = gt_passage_idx
        
        # Compute deflection label: 1 if no GT passage, 0 otherwise
        if args.add_deflection:
            deflection = 1 if gt_passage_idx == -1 else 0
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


def add_prefix_and_form_prompt_bepo(batch, args, pid_to_content_map, qid_to_pairs_info_map):
    """Process a batch of data efficiently using HuggingFace batched processing"""
    batch_size = len(batch['img_path'])
    
    # Process image paths
    img_paths = [f"{args.img_basedir}/{p}" for p in batch['img_path']]
    
    # Pre-allocate prompts list for better performance
    prompts = [''] * batch_size
    gt_passage_indices = [-1] * batch_size
    all_passage_contents = [''] * batch_size
    all_passage_scores = [-1] * batch_size
    all_gt_passage_idx = [-1] * batch_size
    rejected_answers = [''] * batch_size
    
    # Process each item in the batch
    # Note: dataset is pre-filtered to only include question_ids in qid_to_pair_info_map
    for idx in range(batch_size):
        qid = batch['question_id'][idx]
        pairs_info = qid_to_pairs_info_map[qid]
        if pairs_info['chosen_passage_idx'] != pairs_info['rejected_passage_idx']:
            chosen_passage_idx, rejected_passage_idx = pairs_info['chosen_passage_idx'], pairs_info['rejected_passage_idx']
        else:
            chosen_passage_idx = pairs_info['chosen_passage_idx']
            rejected_passage_idx = random.choice([i for i in range(len(pairs_info['passages'])) if i != chosen_passage_idx])
        if chosen_passage_idx is not None:
            passages = [pairs_info['passages'][chosen_passage_idx]] + [pairs_info['passages'][rejected_passage_idx]]
            gt_passage_idx = 0 # NOTE: for BEPO, we use the first passage as the ground truth
        else:
            passages = pairs_info['passages']
            gt_passage_idx = -1

        passage_contents = [f"Title: {p['passage_id']}\tContent: {_truncate_passage_content(pid_to_content_map[p['passage_id']])}" for p in passages]
        passage_scores = [p['score'] for p in passages]

        question_part = f"\n[QUESTION] {batch['question'][idx]}"
        prompt = VLM_PROMPT_FOR_VQA + question_part

        prompts[idx] = prompt
        all_passage_contents[idx] = passage_contents
        all_passage_scores[idx] = passage_scores
        all_gt_passage_idx[idx] = gt_passage_idx
        rejected_answers[idx] = pairs_info['rejected']

    # Update batch with processed data
    batch['img_path'] = img_paths
    batch['prompt'] = prompts
    batch['gt_passage_idx'] = all_gt_passage_idx
    batch['passages'] = all_passage_contents
    batch['passage_scores'] = all_passage_scores
    batch['rejected_answers'] = rejected_answers
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
    parser.add_argument("--topk_docs", type=int, default=5)
    parser.add_argument("--random_sample_1passage_from_topk", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8, help='Number of workers for parallel processing')
    parser.add_argument("--batch_size", type=int, default=100, help='Batch size for dataset processing')
    parser.add_argument("--ensure_gt_passage_in_topk", action='store_true')
    parser.add_argument("--model_sample_filepath", type=str, default=None, help="Path to base model's samples")
    parser.add_argument("--add_separate_prompt_for_prior", action='store_true')
    parser.add_argument("--prior_prompt", type=str, default=None)
    parser.add_argument("--pairwise_data_mode", type=str, default='selfgen')
    parser.add_argument("--bepo_mode", type=str, default='find_wrong_from_all_responses')
    parser.add_argument("--add_deflection", action='store_true', help='Add deflection labels: 1 if no GT passage, 0 otherwise')
    parser.add_argument("--passage_format", type=str, default='text', choices=['text', 'dict'], help='Format for passages: text (string) or dict (with "text" key)')
    args = parser.parse_args()

    print("Loading dataset...")
    ds = load_from_disk(args.hf_dataset_path)
    if args.sample_size > 0 or args.sample_offset > 0:
        print(f"Sampling {args.sample_size} items from {args.sample_offset} to {args.sample_offset + args.sample_size}")
        ds = ds.shuffle(seed=args.seed).select(range(args.sample_offset, min(args.sample_offset + args.sample_size, len(ds))))
    print(f"Dataset loaded with {len(ds)} items")

    print("Loading passages...")
    _, pid_to_content_map = load_passages(args.passage_set_name, split='train')
    print(f"Loaded {len(pid_to_content_map)} passages")

    if args.mode == 'bepo' and args.model_sample_filepath is not None:
        print(f"Loading base model samples from {args.model_sample_filepath}...")
        df = pd.read_csv(args.model_sample_filepath)

        def get_qid_to_pairs_map(df):
            qid_to_pairs_info_map = defaultdict(dict)
            for index, row in df.iterrows():
                qid, all_generated_answers, answer_llks, all_posterior_logits_over_steps, scores, gold_answer = row['question_id'], row['all_generated_answers'], row['all_log_all_tokens_llk'], row['all_posterior_logits_over_steps'], row['all_scores'], row['gold_answer']

                all_generated_answers = ast.literal_eval(all_generated_answers)
                answer_llks = ast.literal_eval(answer_llks)
                all_posterior_logits_over_steps = ast.literal_eval(all_posterior_logits_over_steps)
                scores = ast.literal_eval(scores)

                gt_passage_idx = row['gt_passage_in_zidx']
                if gt_passage_idx != -1:
                    chosen_resp, chosen_passage_idx = gold_answer, gt_passage_idx
                else:
                    chosen_resp = gold_answer
                    chosen_passage_idx = None

                rejected_resp, rejected_passage_idx = None, None
                if args.bepo_mode == 'find_wrong_from_most_likely':
                    most_likely_idx = np.argmax(answer_llks)
                    most_likely_resp = all_generated_answers[most_likely_idx]
                    most_likely_score = scores[most_likely_idx]
                    if most_likely_score < 0.5:
                        rejected_resp = most_likely_resp
                        rejected_passage_idx = np.argmax(all_posterior_logits_over_steps[most_likely_idx][-1])
                    else:
                        pass
                else:
                    for ii, (resp, llk, posterior_logits, score) in enumerate(zip(all_generated_answers, answer_llks, all_posterior_logits_over_steps, scores)):
                        if len(resp) == 0: 
                            continue

                        selected_passage_idx = np.argmax(posterior_logits[-1])
                        if score < 0.5: # correct answer
                            if rejected_resp is None:
                                rejected_resp = resp
                                rejected_passage_idx = selected_passage_idx
                                break
                        if score > 0.5 and gt_passage_idx is None:
                            gt_passage_idx = selected_passage_idx


                if chosen_resp is not None and rejected_resp is not None:
                    qid_to_pairs_info_map[qid] = {
                        'chosen': chosen_resp,
                        'chosen_passage_idx': chosen_passage_idx,
                        'rejected': rejected_resp,
                        'rejected_passage_idx': rejected_passage_idx,
                        'passages': ast.literal_eval(row['passages']),
                    }

                # print("gold answer: ", gold_answer)
                # print("generated answer: ", all_generated_answers)
                # print("chosen: ", chosen_resp, "chosen_passage_idx: ", chosen_passage_idx)
                # print("rejected: ", rejected_resp, "rejected_passage_idx: ", rejected_passage_idx)
                # print(qid_to_pairs_info_map[qid])
                
            return qid_to_pairs_info_map

        qid_to_pairs_info_map = get_qid_to_pairs_map(df)
        print(f"Loaded {len(qid_to_pairs_info_map)} pairs of chosen and rejected samples")
    
    # Only keep the specified fields in the dataset
    # For SFT mode, keep: 'img_path', 'prompt', 'gold_answer'
    # For DPO mode, keep: 'img_path', 'answer_input', 'chosen_answer_target', 'rejected_answer_target'
    keep_fields = ['question_id', 'img_path', 'prompt', 'gold_answer', 'retrieved_passage', 'pos_item_ids', 'question']

    ds = ds.remove_columns([col for col in ds.column_names if col not in keep_fields])
    
    # Use HuggingFace's native batched processing with multiprocessing
    print("Processing dataset with batched operations...")
    
    # Create a partial function with the required arguments
    if args.mode == 'sft':
        process_func = partial(add_prefix_and_form_prompt, args=args, pid_to_content_map=pid_to_content_map)
    elif args.mode == 'bepo':
        process_func = partial(add_prefix_and_form_prompt_bepo, args=args, pid_to_content_map=pid_to_content_map, qid_to_pairs_info_map=qid_to_pairs_info_map)
        
        # Filter dataset to only keep rows with question_id in qid_to_pair_info_map
        print(f"Filtering dataset for bepo mode (before: {len(ds)} rows)...")
        ds = ds.filter(
            lambda batch: [qid in qid_to_pairs_info_map for qid in batch['question_id']],
            batched=True,
            batch_size=1024,
            num_proc=args.num_workers,
            desc="Filtering by question_id in pair_info_map"
        )
        print(f"After filtering: {len(ds)} rows remaining")
    else:
        raise NotImplementedError(f"Processing function not implemented for mode: {args.mode}")
    
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
    del pid_to_content_map
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
        # Use batched=True and set num_proc to a higher value (e.g., 8)
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

    print("Converting to ShareGPT format...")
    convert_to_sharegpt_format(ds, args.output_dir, mode=args.mode)
    print(f"Processing complete! Output saved to {args.output_dir}")

    

    
