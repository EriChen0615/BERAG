import sys
sys.path.append('./src')
from hf_backend import HFQwen2VLBackend
from bape_inference_engine import BAPEInferenceEngine
from vqa_datasets import load_passages
import json
from pprint import pprint
from datasets import load_from_disk
from torch.utils.data import DataLoader
from PIL import Image
import argparse

import torch
import torch.distributed as dist

from tqdm import tqdm
import gc
import os
import time
import numpy as np
import pandas as pd

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_passage_parallel_runtime(args):
    args.pp_world_size = 1
    args.pp_rank = 0
    args.pp_local_rank = 0
    if not args.passage_parallel:
        return args
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available, but --passage_parallel was set.")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.pp_rank = rank
    args.pp_world_size = world_size
    args.pp_local_rank = local_rank

    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.pp_backend, init_method="env://")
    print(
        f"[PassageParallel] enabled={args.passage_parallel}, "
        f"backend={args.pp_backend}, rank/world={args.pp_rank}/{args.pp_world_size}, local_rank={args.pp_local_rank}"
    )
    return args


def is_main_process(args):
    return (not args.passage_parallel) or args.pp_rank == 0


def compute_prior_recall_at_k(all_results):
    """
    Compute recall@K based on prior_logits sorted passages.
    Similar to evaluate_retrieval in reranker_inference.py
    """
    import ast

    def _to_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return []
            try:
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    if not all_results:
        return {}

    normalized_prior_lists = [_to_list(r.get('prior_sorted_passage_ids', [])) for r in all_results]
    non_empty_prior_lists = [lst for lst in normalized_prior_lists if len(lst) > 0]
    if len(non_empty_prior_lists) == 0:
        return {}

    # Determine max K from normalized data
    max_k = max(len(lst) for lst in non_empty_prior_lists)

    recall_at_k = {}
    for k in range(1, max_k + 1):
        hits = []
        for result, prior_sorted_ids in zip(all_results, normalized_prior_lists):
            if not prior_sorted_ids:
                continue
            
            # Get ground truth passage IDs
            # In the dataset, pos_item_ids contains ground truth passage IDs
            # But in all_results, we need to get it from the original item
            # For now, we use gt_passage_in_zidx and passages to get the GT passage ID
            gt_passage_in_zidx_raw = result.get('gt_passage_in_zidx', -1)
            try:
                gt_passage_in_zidx = int(float(gt_passage_in_zidx_raw))
            except Exception:
                gt_passage_in_zidx = -1
            if gt_passage_in_zidx == -1:
                # No ground truth in retrieved passages
                hits.append(0)
                continue
            
            # Get GT passage ID from passages
            passages = _to_list(result.get('passages', []))
            if gt_passage_in_zidx < len(passages) and isinstance(passages[gt_passage_in_zidx], dict):
                gt_passage_id = passages[gt_passage_in_zidx].get('passage_id')
                if gt_passage_id is None:
                    hits.append(0)
                    continue
                
                # Check if GT is in top-k of prior sorted passages
                top_k_prior_ids = prior_sorted_ids[:k]
                if gt_passage_id in top_k_prior_ids:
                    hits.append(1)
                else:
                    hits.append(0)
            else:
                hits.append(0)
        
        if hits:
            recall_at_k[k] = np.mean(hits)
    
    return recall_at_k

def get_prompt_template(prompt_template: str):
    if prompt_template is None or prompt_template == '':
        return (
            "Answer the question after [QUESTION] about the image."
            "A retriever has retrieved a relevant document for you and provided it after [EVIDENCE]."
            "Give your answer after [ANSWER]\n"
        )
    else:
        with open(prompt_template, 'r') as f:
            return f.read()


def write_inference_report(df: pd.DataFrame, output_dir: str, total_inference_runtime_sec=None):
    metric_cols = [
        "prefill_ms",
        "prefill_forward_ms",
        "prior_head_ms",
        "decode_ms",
        "decode_tokens_per_ms",
        "input_tokens",
        "output_tokens",
        "decode_tokens",
    ]
    numeric_df = df.copy()
    for col in metric_cols:
        if col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    if "num_prefill_branches" in numeric_df.columns:
        numeric_df["num_prefill_branches"] = pd.to_numeric(numeric_df["num_prefill_branches"], errors="coerce").fillna(1.0)
    else:
        numeric_df["num_prefill_branches"] = 1.0
    numeric_df["input_tokens_total"] = numeric_df["input_tokens"] * numeric_df["num_prefill_branches"]
    timing_df = numeric_df.iloc[1:].copy() if len(numeric_df) > 1 else numeric_df.copy()
    if "decode_ms" in timing_df.columns and "decode_tokens" in timing_df.columns:
        per_token_decode_ms = (
            timing_df.loc[timing_df["decode_tokens"] > 0, "decode_ms"]
            / timing_df.loc[timing_df["decode_tokens"] > 0, "decode_tokens"]
        )
        avg_per_token_decode_ms = float(per_token_decode_ms.mean()) if len(per_token_decode_ms) > 0 else 0.0
    else:
        avg_per_token_decode_ms = 0.0

    report = {
        "num_examples": int(len(numeric_df)),
        "num_examples_for_timing": int(len(timing_df)),
        "num_valid_decode_examples": int(timing_df["decode_tokens_per_ms"].notna().sum()) if "decode_tokens_per_ms" in timing_df.columns else 0,
        "avg_prefill_ms": float(timing_df["prefill_ms"].mean()) if "prefill_ms" in timing_df.columns else 0.0,
        "avg_prefill_forward_ms": float(timing_df["prefill_forward_ms"].mean()) if "prefill_forward_ms" in timing_df.columns else 0.0,
        "avg_prior_head_ms": float(timing_df["prior_head_ms"].mean()) if "prior_head_ms" in timing_df.columns else 0.0,
        "avg_decode_tokens_per_ms": float(timing_df["decode_tokens_per_ms"].mean()) if "decode_tokens_per_ms" in timing_df.columns else 0.0,
        "avg_input_tokens_per_branch": float(numeric_df["input_tokens"].mean()) if "input_tokens" in numeric_df.columns else 0.0,
        "avg_input_tokens_total": float(numeric_df["input_tokens_total"].mean()) if "input_tokens_total" in numeric_df.columns else 0.0,
        "avg_output_tokens": float(numeric_df["output_tokens"].mean()) if "output_tokens" in numeric_df.columns else 0.0,
        "avg_decode_ms": float(timing_df["decode_ms"].mean()) if "decode_ms" in timing_df.columns else 0.0,
        "avg_per_token_decode_ms": avg_per_token_decode_ms,
        "avg_decode_tokens": float(timing_df["decode_tokens"].mean()) if "decode_tokens" in timing_df.columns else 0.0,
        "total_inference_runtime_sec": float(total_inference_runtime_sec) if total_inference_runtime_sec is not None else None,
    }

    with open(f"{output_dir}/inference_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved inference report to {output_dir}/inference_report.json")

def process_dataset_with_bape(engine, vqa_dataset, max_new_tokens, args):
    all_results = []
    for idx, item in enumerate(tqdm(vqa_dataset, total=len(vqa_dataset))):
        text = _make_prompt(item['question'])
        x = {"text": text, "image": os.path.join(args.img_basedir, item['img_path'])}

        score_field = 'score' if args.retrieval_field == 'retrieved_passage' else 'rerank_score'

        if args.include_gt_passage_only:
            z = [{'passage_id': item['pos_item_ids'][0], 'passage_content': item['pos_item_contents'][0], score_field: 1.0}]
        else:
            z = item[args.retrieval_field][:args.retrieval_topk]

        passage_scores = [zk[score_field] for zk in z]
        if args.include_z0_in_ensemble:
            # Add z0 as the last passage with content " " (single space)
            z.append({'passage_id': 'z0', 'passage_content': ' ', score_field: np.mean(passage_scores)})
            passage_scores.append(np.mean(passage_scores))

        retrieved_passage_ids = [zk['passage_id'] for zk in z]
        gt_passage_id = item['pos_item_ids'][0]
        if args.ensure_gt_passage_in_ensemble: 
            if gt_passage_id not in retrieved_passage_ids:
                z[0] = {'passage_id': item['pos_item_ids'][0], 'passage_content': item['pos_item_contents'][0], score_field: 1.0}
                passage_scores[0] = 1.0
                retrieved_passage_ids = [zk['passage_id'] for zk in z] #NOTE update retrieved_passage_ids after replacing the first doc with GT doc

        if args.passage_prior == "log_softmax":
            passage_prior = torch.log_softmax(torch.tensor(passage_scores, device=engine.device), dim=0)
        elif args.passage_prior == "uniform":
            passage_prior = torch.ones(len(z), device=engine.device) / len(z)
        elif args.passage_prior == "prior_head":
            passage_prior = None
        else:
            raise NotImplementedError(f"Passage prior {args.passage_prior} not implemented")

        passage_contents = [_make_evidence_part(zk) for zk in z]

        if args.inference_engine_version == "v1":
            if args.passage_parallel:
                generated_token_ids, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits, inference_stats = engine.generate_passage_parallel(
                    x, passage_contents, passage_prior, max_new_tokens=max_new_tokens, return_stats=True
                )
            else:
                generated_token_ids, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits, inference_stats = engine.generate(
                    x, passage_contents, passage_prior, max_new_tokens=max_new_tokens, return_stats=True
                )
        elif args.inference_engine_version == "v2":
            if args.passage_parallel:
                raise NotImplementedError("Passage parallelism currently supports only inference_engine_version=v1 (greedy).")
            all_generated_token_ids, all_log_all_tokens_llk, all_posterior_logits_over_steps, prior_logits, inference_stats = engine.generate_v2(
                x,
                passage_contents,
                passage_prior,
                max_new_tokens=max_new_tokens,
                return_n_sequences=args.return_n_sequences,
                return_stats=True,
            )
            posterior_logits_over_steps = all_posterior_logits_over_steps[0]
            log_all_tokens_llk = all_log_all_tokens_llk[0]
            generated_token_ids = all_generated_token_ids[0]
            posterior_max_idx = np.argmax(posterior_logits_over_steps) if posterior_logits_over_steps else -1
            prior_max_idx = np.argmax(prior_logits) if prior_logits is not None and len(prior_logits) > 0 else -1
        else:
            raise NotImplementedError(f"Inference engine version {args.inference_engine_version} not implemented")

        if args.return_n_sequences == 1:
            response = engine.backend.processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            generated_answer = response.split('[ANSWER]')[-1].strip()
        else:
            all_responses = engine.backend.processor.tokenizer.batch_decode(all_generated_token_ids, skip_special_tokens=True)
            all_generated_answers = [response.split('[ANSWER]')[-1].strip() for response in all_responses]
            response = all_responses[0]
            generated_answer = all_generated_answers[0]

        # Monitoring
        gt_passage_in_zidx = -1
        if gt_passage_id in retrieved_passage_ids:
            gt_passage_in_zidx = retrieved_passage_ids.index(gt_passage_id)
        
        z_dominant_idx = posterior_max_idx

        # Sort passages by prior_logits for recall computation
        prior_sorted_passage_ids = []
        if prior_logits is not None and len(prior_logits) > 0:
            prior_sorted_indices = np.argsort(prior_logits)[::-1]  # Sort descending
            prior_sorted_passage_ids = [retrieved_passage_ids[i] for i in prior_sorted_indices]

        all_results.append({
            'question_id': item['question_id'],
            'question': item['question'],
            'question_type': item.get('question_type', ''),
            'img_path': item['img_path'],
            'image_id': item['img_id'] if 'img_id' in item else item['image_id'],
            'gold_answer': item['gold_answer'],
            'answers': item['answers'],
            'response': response,
            'generated_answer': generated_answer,
            'passages': z,
            'prompt_text': text,
            'posterior_max_idx': posterior_max_idx,
            'prior_max_idx': prior_max_idx,
            'gt_passage_in_zidx': gt_passage_in_zidx,
            'z_dominant_idx': z_dominant_idx,
            'dominant_passage_is_gt': z_dominant_idx == gt_passage_in_zidx and z_dominant_idx != -1,
            'prior_passage_is_gt': prior_max_idx == gt_passage_in_zidx and prior_max_idx != -1,
            'log_all_tokens_llk': log_all_tokens_llk,
            'prior_logits': prior_logits if prior_logits is not None else None,
            'prior_sorted_passage_ids': prior_sorted_passage_ids,
            'prefill_ms': inference_stats.get('prefill_ms', 0.0),
            'prefill_forward_ms': inference_stats.get('prefill_forward_ms', inference_stats.get('prefill_ms', 0.0)),
            'prior_head_ms': inference_stats.get('prior_head_ms', 0.0),
            'decode_ms': inference_stats.get('decode_ms', 0.0),
            'decode_tokens': inference_stats.get('decode_tokens', 0),
            'decode_tokens_per_ms': inference_stats.get('decode_tokens_per_ms', 0.0),
            'input_tokens': inference_stats.get('input_tokens', 0),
            'output_tokens': inference_stats.get('output_tokens', len(generated_token_ids)),
            'num_prefill_branches': len(passage_contents),
            'dynamic_k_top_p': args.dynamic_k_top_p,
            'retrieval_topk': args.retrieval_topk,
        })

        if args.return_n_sequences > 1:
            all_results[-1].update({
                'all_generated_answers': all_generated_answers,
                'all_posterior_logits_over_steps': all_posterior_logits_over_steps,
                'all_log_all_tokens_llk': all_log_all_tokens_llk,
                'posterior_logits_over_steps': posterior_logits_over_steps,
            })

        # #NOTE: DEBUG DISPLAY
        # print(f"Question ID: {item['question_id']}")
        # print(f"Question: {item['question']}")
        # print(f"Gold answer: {item['gold_answer']}")
        # print(f"Generated response: {response}")
        # print(f"Generated answer: {generated_answer}")
        # print(f"Passage Prior: {passage_prior}")
        # print(f"Passage Posterior: {log_passage_posterior}")
        # print(f"GT Passage idx in Z: {gt_passage_in_zidx}")
        # print(f"Dominant passage is ground-truth: {z_dominant_idx == gt_passage_in_zidx}")

        # print(f"Retrieved passages: {z}")
        # print(f"Dominant passage: {z[z_dominant_idx]}")
        # print(f"Dominant passage content: {passage_contents[z_dominant_idx]}")
        # print(f"Ground-truth passage id: {item['pos_item_ids'][0]}")
        # print(f"Ground-truth passage content: {item['pos_item_contents'][0]}")
        # print(f"Ground-truth passage in retrieved passages: {item['pos_item_ids'][0] in [zk['passage_id'] for zk in z]}")

        
    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset settings
    parser.add_argument("--retrieval_ds_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None) 
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--take_n", type=int, default=-1)
    parser.add_argument("--img_basedir", type=str, default='')
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--prefill_ans_token", action="store_true")
    parser.add_argument("--max_words_per_evidence", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=0)

    # Retrieval settings
    parser.add_argument("--retrieval_field", type=str, default=None, choices=['retrieved_passage', 'reranked_passage'])
    parser.add_argument("--retrieval_topk", type=int, default=5)

    # Model settings
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)

    # Inference/BAPE settings
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=12288) 
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--include_z0_in_ensemble", action="store_true") # if set, provide an empty passage in ensemble
    parser.add_argument("--include_gt_passage_only", action="store_true") # if set, use the ground-truth passage only in generation 
    parser.add_argument("--ensure_gt_passage_in_ensemble", action="store_true") # if set, ensure the ground-truth passage is in the ensemble
    parser.add_argument("--passage_prior", type=str, default="log_softmax")
    parser.add_argument("--prior_head_path", type=str, default=None)
    parser.add_argument("--prior_head_modeling", type=str, default="mlp_head", choices=["mlp_head", "linear_head"])
    parser.add_argument("--prior_head_num_layers", type=int, default=2)
    parser.add_argument("--prior_head_proj_dim", type=int, default=1024)
    parser.add_argument("--dynamic_k_top_p", type=float, default=None)
    parser.add_argument("--max_batch_size_per_forward", type=int, default=5)
    parser.add_argument("--hidden_state_offset", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--inference_engine_version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--return_n_sequences", type=int, default=1)
    parser.add_argument("--passage_parallel", action="store_true")
    parser.add_argument("--pp_backend", type=str, default="nccl")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    # Evaluation setting
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--use_cache", action="store_true")

    # Saving settings
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()
    args = init_passage_parallel_runtime(args)
    main_proc = is_main_process(args)

    output_filepath = f"{args.exp_name}/inference_results.csv"
    if main_proc:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    vqa_dataset = load_from_disk(args.retrieval_ds_path)
    total_inference_runtime_sec = None

    if main_proc and os.path.exists(output_filepath) and args.use_cache:
        all_results = pd.read_csv(output_filepath)
    else:
        # Load VQA dataset
        # vqa_dataset = load_vqa_dataset(args.dataset_name, split=args.split, img_basedir=args.img_basedir, take_n=args.take_n, seed=args.ds_seed)
        if args.take_n > 0:
            print(f"Taking {args.take_n} examples from the dataset, starting from index {args.offset}.")
            vqa_dataset = vqa_dataset.shuffle(seed=42).select([i for i in range(args.offset, args.offset + args.take_n)]) # shuffle
        else:
            print("Using the entire dataset.")

        split_name = 'test'
        if args.dataset_name == 'InfoseekNew_FullPassage':
            split_name = 'valid'
        passages, pid_to_content_map = load_passages(args.dataset_name, split=split_name)

        VLM_PROMPT_FOR_VQA = get_prompt_template(args.prompt_template)
        print("--------------------------------")
        print(f"Read prompt template from {args.prompt_template}.")
        print(f"Prompt template: {VLM_PROMPT_FOR_VQA}.")
        print("--------------------------------")

        def _make_evidence_part(passage_dict):
            pid = passage_dict['passage_id']
            if pid == 'z0':
                return "[EVIDENCE] No passage provided."

            text = pid_to_content_map[pid]
            text = ' '.join(text.split(' ')[:args.max_words_per_evidence])
            # return (
            #     "[EVIDENCE] "
            #     f"{text}\n"
            # ) # version for replicating previous results
            return (
                "[EVIDENCE]"
                f" "
                f"Title: {pid}\t"
                f"Content: {text}"
                f"\n"
            )
        
        def _make_prompt(question):
            prompt = VLM_PROMPT_FOR_VQA + "<<<EVIDENCE>>>"
            prompt += f"\n[QUESTION] {question}"

            # version for previous results. set prefill_ans_token to True
            # prompt = VLM_PROMPT_FOR_VQA + f"\nQuestion: {question}\n" #NOTE give the question first!
            # prompt += ''.join(evidence_parts)
            # prompt += f"\nQuestion: {question}"
            if args.prefill_ans_token:
                prompt += f"\n[ANSWER]"
            # print(prompt)
            # breakpoint()
            return prompt

        backend = HFQwen2VLBackend(
            model_path=args.model_path,
            processor_path=args.processor_path if args.processor_path is not None and args.processor_path != '' else args.model_path,
            adapter_name_or_path=args.adapter_name_or_path,
            max_batch_size_per_forward=args.max_batch_size_per_forward,
            attn_implementation=args.attn_implementation,
            force_single_device_per_rank=args.passage_parallel,
            local_rank=args.pp_local_rank,
        )

        prior_head_config = {
            "modeling": args.prior_head_modeling,
            "num_layers": args.prior_head_num_layers,
            "proj_dim": args.prior_head_proj_dim,
        }
        # Initialize Inference Engine
        engine = BAPEInferenceEngine(backend, prior_head_path=args.prior_head_path, prior_head_config=prior_head_config, dynamic_k_top_p=args.dynamic_k_top_p, hidden_state_offset=args.hidden_state_offset, num_beams=args.num_beams)
        
        inference_start_time = time.perf_counter()
        all_results = process_dataset_with_bape(engine, vqa_dataset, max_new_tokens=args.max_new_tokens, args=args)
        total_inference_runtime_sec = time.perf_counter() - inference_start_time
        all_results = pd.DataFrame(all_results)
        del engine
        torch.cuda.empty_cache()

        # save all_results to a CSV file
        if main_proc:
            all_results.to_csv(output_filepath, index=False)

    # If loading cached inference results, still produce/refresh inference report.
    if main_proc and os.path.exists(output_filepath):
        if not isinstance(all_results, pd.DataFrame):
            all_results = pd.DataFrame(all_results)
        write_inference_report(
            all_results,
            os.path.dirname(output_filepath),
            total_inference_runtime_sec=total_inference_runtime_sec,
        )

    # Evaluation
    if args.do_eval and main_proc:
        if args.dataset_name == 'EVQA':
            sys.path.append('./src/evaluation')
            # df = vqa_dataset.to_pandas()

            from evqa_eval_1004 import process_row as eval_process_row
            from evqa_eval_1004 import process_row_mp as eval_process_row_mp
            from evqa_eval_1004 import extract_queries_and_retrieved_docs

            df = all_results
            import ast
            df['answers'] = df['answers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            def replace_nan_with_unk(x):
                if isinstance(x, float) and np.isnan(x):
                    return 'unk'
                elif isinstance(x, list) or isinstance(x, tuple):
                    return [replace_nan_with_unk(y) for y in x]
                return x
            df['answers'] = df['answers'].apply(replace_nan_with_unk)
            
            # Handle multiple generated answers if present
            if 'all_generated_answers' in df.columns:
                # Convert string representation to list if loading from CSV
                df['all_generated_answers'] = df['all_generated_answers'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                # Store original question_id for later aggregation
                df['original_question_id'] = df['question_id']
                # Explode the dataframe to create one row per generated answer
                df = df.explode('all_generated_answers').reset_index(drop=True)
                df['prediction'] = df['all_generated_answers']
            else:
                df['prediction'] = df['generated_answer']
                df['original_question_id'] = df['question_id']

            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            
            # Original multiprocessing code (commented out due to pickle error with TensorFlow Hub model)
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            
            if False:
                for row in tqdm(df.itertuples(), total=len(df)):
                    eval_result = eval_process_row(row)
                    all_eval_results.append(eval_result)
                dict_to_report = {f"avg_{k}": sum([res[k] for res in all_eval_results])/len(all_eval_results) for k in all_eval_results[0]}
                for k in all_eval_results[0]:
                    df[k] = [res[k] for res in all_eval_results]
            else:
                with multiprocessing.Pool(processes=8) as pool:
                    all_eval_results = list(tqdm(pool.imap(eval_process_row_mp, df.iterrows(), chunksize=1), total=len(df)))
                dict_to_report = {f"avg_score": sum(all_eval_results)/len(all_eval_results)}
                df['score'] = all_eval_results
            
            # Aggregate scores if we have multiple answers per question
            if 'all_generated_answers' in all_results.columns:
                # Group by original question to compute median, best, mean, and list of all scores
                grouped_scores = df.groupby('original_question_id')['score'].agg([
                    ('median_score', 'median'),
                    ('best_score', 'max'),
                    ('mean_score', 'mean'),
                    ('all_scores', list)  # Keep raw scores as a list
                ]).reset_index()
                grouped_scores.columns = ['question_id', 'median_score', 'best_score', 'mean_score', 'all_scores']
                
                # Compute overall metrics
                dict_to_report['avg_median_score'] = grouped_scores['median_score'].mean()
                dict_to_report['avg_best_score'] = grouped_scores['best_score'].mean()
                dict_to_report['avg_mean_score'] = grouped_scores['mean_score'].mean()
                
                print(f"\n[Multiple Sequences Evaluation]")
                print(f"  Average of median scores per question: {dict_to_report['avg_median_score']:.4f}")
                print(f"  Average of best scores per question: {dict_to_report['avg_best_score']:.4f}")
                print(f"  Average of mean scores per question: {dict_to_report['avg_mean_score']:.4f}")
                
                # Merge aggregated scores back to the original dataframe
                all_results_with_agg = all_results.merge(grouped_scores, left_on='question_id', right_on='question_id', how='left')

            torch.cuda.empty_cache()

            # Add Monitoring results
            dominant_passage_hit_rate = df[df['gt_passage_in_zidx'] != -1]['dominant_passage_is_gt'].mean()
            retrieval_hit_rate = len(df[df['gt_passage_in_zidx'] != -1]) / len(df)
            correct_ignore_rate = len(df[(df['gt_passage_in_zidx'] == -1) & (df['z_dominant_idx'] == args.retrieval_topk)]) / len(df[df['gt_passage_in_zidx'] == -1]) if len(df[df['gt_passage_in_zidx'] == -1]) > 0 else 0
            prior_hit_rate = df[df['prior_max_idx'] != -1]['prior_passage_is_gt'].mean()

            dict_to_report['posterior_passage_hit_rate'] = dominant_passage_hit_rate
            dict_to_report['retrieval_hit_rate'] = retrieval_hit_rate
            dict_to_report['prior_passage_hit_rate'] = prior_hit_rate
            dict_to_report['correct_ignore_rate'] = correct_ignore_rate

            # Compute prior recall@K
            prior_recall_at_k = compute_prior_recall_at_k(all_results.to_dict('records'))
            if prior_recall_at_k:
                dict_to_report['prior_recall_at_k'] = prior_recall_at_k
                print(f"\nPrior Recall@K metrics:")
                for k, recall in prior_recall_at_k.items():
                    print(f"  Prior Recall@{k}: {recall:.4f}")

            print("--------------------------------")
            print("Evaluation results:")
            print(dict_to_report)
            print("--------------------------------")

            df.to_csv(f"{os.path.dirname(output_filepath)}/marked_inference_results.csv", index=False)
            
            # Save aggregated results if multiple sequences were generated
            if 'all_generated_answers' in all_results.columns:
                all_results_with_agg.to_csv(f"{os.path.dirname(output_filepath)}/aggregated_scores.csv", index=False)
                print(f"Aggregated scores saved to {os.path.dirname(output_filepath)}/aggregated_scores.csv")
            
            with open(f'{os.path.dirname(output_filepath)}/scores.json', 'w') as f:
                json.dump(dict_to_report, f, indent=2)
            print("Evaluation results saved to", os.path.dirname(output_filepath))
        
        elif args.dataset_name in ['Infoseek', 'InfoseekNew', 'InfoseekNew_FullPassage']:
            sys.path.append("./third_party/infoseek_eval")
            predictions = []
            df = all_results

            def _safe_prediction_text(value):
                if isinstance(value, str):
                    return value
                if isinstance(value, (list, tuple)):
                    return _safe_prediction_text(value[0]) if len(value) > 0 else ""
                if pd.isna(value):
                    return ""
                return str(value)
            
            # Handle multiple generated answers if present - use the first answer
            if 'all_generated_answers' in df.columns:
                import ast
                df['all_generated_answers'] = df['all_generated_answers'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                # Use the first generated answer for evaluation
                df['prediction'] = df['all_generated_answers'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
            else:
                df['prediction'] = df['generated_answer']
            df['prediction'] = df['prediction'].apply(_safe_prediction_text)
            
            # Save inference results before evaluation (like EVQA)
            df.to_csv(f"{os.path.dirname(output_filepath)}/inference_results_with_predictions.csv", index=False)
            print(f"Saved inference results (before scoring) to {os.path.dirname(output_filepath)}/inference_results_with_predictions.csv")
            
            for i, item in enumerate(df.itertuples()):
                predictions.append({
                    'data_id': item.question_id,
                    'prediction': item.prediction
                })
            
            pred_path = f"{os.path.dirname(output_filepath)}/predictions.jsonl"
            with open(pred_path, 'w') as f:
                for pred in predictions:
                    f.write(json.dumps(pred)+'\n')
            
            if args.split in ['test', 'valid', 'valid_m2kr']:
                reference_path = f"third_party/infoseek_eval/infoseek/infoseek_val.jsonl"
                reference_qtype_path = f"third_party/infoseek_eval/infoseek/infoseek_val_qtype.jsonl"
            elif args.split in ['train']:
                reference_path = f"third_party/infoseek_eval/infoseek/infoseek_train.jsonl"
                reference_qtype_path = None
            else:
                raise ValueError(f"Unknown split: {args.split}")

            from infoseek_eval import evaluate
            result = evaluate(pred_path, reference_path, reference_qtype_path)
            final_score = result["final_score"]
            unseen_question_score = result["unseen_question_score"]["score"]
            unseen_entity_score = result["unseen_entity_score"]["score"]
            print(f"\n[Infoseek Evaluation]")
            print(f"  {args.split} final score: {final_score}")
            print(f"  {args.split} unseen question score: {unseen_question_score}")
            print(f"  {args.split} unseen entity score: {unseen_entity_score}")
            
            dict_to_report = {
                'score': final_score,
                'unseen_question_score': unseen_question_score,
                'unseen_entity_score': unseen_entity_score
            }
            
            # Add Monitoring results (like EVQA)
            dominant_passage_hit_rate = df[df['gt_passage_in_zidx'] != -1]['dominant_passage_is_gt'].mean()
            retrieval_hit_rate = len(df[df['gt_passage_in_zidx'] != -1]) / len(df)
            correct_ignore_rate = len(df[(df['gt_passage_in_zidx'] == -1) & (df['z_dominant_idx'] == args.retrieval_topk)]) / len(df[df['gt_passage_in_zidx'] == -1]) if len(df[df['gt_passage_in_zidx'] == -1]) > 0 else 0
            prior_hit_rate = df[df['prior_max_idx'] != -1]['prior_passage_is_gt'].mean()

            dict_to_report['posterior_passage_hit_rate'] = dominant_passage_hit_rate
            dict_to_report['retrieval_hit_rate'] = retrieval_hit_rate
            dict_to_report['prior_passage_hit_rate'] = prior_hit_rate
            dict_to_report['correct_ignore_rate'] = correct_ignore_rate
            
            # Compute prior recall@K
            prior_recall_at_k = compute_prior_recall_at_k(all_results.to_dict('records'))
            if prior_recall_at_k:
                dict_to_report['prior_recall_at_k'] = prior_recall_at_k
                print(f"\nPrior Recall@K metrics:")
                for k, recall in prior_recall_at_k.items():
                    print(f"  Prior Recall@{k}: {recall:.4f}")
            
            # Save inference results with scores (like EVQA marked_inference_results.csv)
            df.to_csv(f"{os.path.dirname(output_filepath)}/marked_inference_results.csv", index=False)
            print(f"Saved inference results (after scoring) to {os.path.dirname(output_filepath)}/marked_inference_results.csv")
            
            print("--------------------------------")
            print("Evaluation results:")
            print(dict_to_report)
            print("--------------------------------")
            
            with open(f'{os.path.dirname(output_filepath)}/scores.json', 'w') as f:
                json.dump(dict_to_report, f, indent=2)
            print("Evaluation results saved to", os.path.dirname(output_filepath))
        
        else:
            raise NotImplementedError(f"Evaluation for {args.dataset_name} not implemented")

    if args.passage_parallel and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()