import sys
sys.path.append('./src')
from hf_backend import HFQwen2VLBackend
from bape_inference_engine import BAPEInferenceEngine
import json
import ast
import collections
from pprint import pprint
from datasets import load_dataset, load_from_disk
from PIL import Image
import argparse

import torch

from tqdm import tqdm
import gc
import os
import re
import numpy as np
import pandas as pd
import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global prompt template for SlideVQA (matching ragk_slidevqa.py)
VLM_PROMPT_FOR_VQA = (
    "Answer the question directly without explanations based on the provided slides."
    "<<<EVIDENCE>>>"
)


def compute_token_f1(gold_answer: str, generated_answer: str) -> float:
    def _normalize(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    gold_tokens = _normalize(gold_answer).split()
    pred_tokens = _normalize(generated_answer).split()

    if len(gold_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_prior_recall_at_k(all_results):
    """
    Compute recall@K based on prior_logits sorted passages.
    Similar to evaluate_retrieval in reranker_inference.py
    """
    if not all_results:
        return {}
    
    # Check if we have prior_sorted_passage_ids
    if 'prior_sorted_passage_ids' not in all_results[0] or not all_results[0]['prior_sorted_passage_ids']:
        return {}
    
    # Determine max K from the data
    def _to_list(val):
        """Parse list-like values that may be stored as strings in cached CSVs."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    # Normalize prior_sorted_passage_ids to proper lists
    for r in all_results:
        r['prior_sorted_passage_ids'] = _to_list(r.get('prior_sorted_passage_ids', []))
        r['passages'] = _to_list(r.get('passages', []))

    max_k = max(len(r['prior_sorted_passage_ids']) for r in all_results if r.get('prior_sorted_passage_ids'))
    
    recall_at_k = {}
    for k in range(1, max_k + 1):
        hits = []
        for result in all_results:
            prior_sorted_ids = _to_list(result.get('prior_sorted_passage_ids', []))
            if not prior_sorted_ids:
                continue
            
            # Get ground truth passage IDs
            gt_passage_in_zidx = result.get('gt_passage_in_zidx', -1)
            try:
                if gt_passage_in_zidx == gt_passage_in_zidx:  # filter out NaN
                    gt_passage_in_zidx = int(gt_passage_in_zidx)
                else:
                    gt_passage_in_zidx = -1
            except Exception:
                gt_passage_in_zidx = -1
            if gt_passage_in_zidx == -1:
                # No ground truth in retrieved passages
                hits.append(0)
                continue
            
            # Get GT passage ID from passages
            passages = _to_list(result.get('passages', []))
            if gt_passage_in_zidx < len(passages):
                # For SlideVQA, passages are dicts with images, use page_num as ID
                gt_passage = passages[gt_passage_in_zidx]
                if isinstance(gt_passage, dict) and 'page_num' in gt_passage:
                    gt_passage_id = gt_passage['page_num']
                else:
                    # Fallback: use index as ID
                    gt_passage_id = gt_passage_in_zidx
                
                # Check if GT is in top-k of prior sorted passages
                top_k_prior_ids_raw = prior_sorted_ids[:k]
                top_k_prior_ids = []
                for pid in top_k_prior_ids_raw:
                    try:
                        top_k_prior_ids.append(int(pid))
                    except Exception:
                        continue
                try:
                    gt_pid_int = int(gt_passage_id)
                except Exception:
                    gt_pid_int = gt_passage_id
                if gt_pid_int in top_k_prior_ids:
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
        return VLM_PROMPT_FOR_VQA
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
    if "input_tokens" in numeric_df.columns:
        numeric_df["input_tokens_total"] = numeric_df["input_tokens"] * numeric_df["num_prefill_branches"]
    else:
        numeric_df["input_tokens_total"] = np.nan
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


def load_image_from_path(img_path):
    """
    Load image from path string or PIL Image object.
    
    Args:
        img_path: Path to image file (str) or PIL Image object
        
    Returns:
        PIL Image object or None
    """
    try:
        # Handle PIL Image objects directly
        if isinstance(img_path, Image.Image):
            # Ensure it's RGB
            if img_path.mode != 'RGB':
                return img_path.convert('RGB')
            return img_path
        
        # Handle string paths
        if isinstance(img_path, str):
            if os.path.exists(img_path):
                return Image.open(img_path).convert('RGB')
            else:
                logging.error(f"Image path does not exist: {img_path}")
                return None
        else:
            logging.error(f"Unsupported image type: {type(img_path)}")
            return None
    except Exception as e:
        logging.error(f"Error loading image from {img_path}: {e}")
        return None


def subdivide_image(image, n_parts):
    """
    Subdivide an image into N sub-images of equal sizes.
    
    Args:
        image: PIL Image object
        n_parts: Number of parts (must be power of 2). 
                 N=2: divide length (height) into 2 parts (2 rows, 1 col)
                 N=4: divide length (height) by 2, width by 2 (2x2 grid)
                 N=8: divide length (height) by 2, width by 4 (2x4 grid)
                 N=16: divide length (height) by 4, width by 4 (4x4 grid)
                 etc.
                 Always divides length (height) first, then width.
    
    Returns:
        List of PIL Image objects in row-wise order (top-left, top-right, ..., bottom-left, bottom-right)
    """
    if n_parts <= 0 or (n_parts & (n_parts - 1)) != 0:
        raise ValueError(f"n_parts must be a power of 2, got {n_parts}")
    
    width, height = image.size
    
    if n_parts == 2:
        rows = 1
        cols = 2
    elif n_parts == 4:
        rows = 2
        cols = 2
    elif n_parts == 9:
        rows = 3
        cols = 3
    else:
        raise ValueError(f"Unsupported n_parts: {n_parts}")
   
    # Calculate sub-image dimensions
    sub_width = width // cols
    sub_height = height // rows
    
    sub_images = []
    
    # Extract sub-images in row-wise order
    # z=0 is top-left, z=1 is top-right, z=2 is bottom-left, z=3 is bottom-right for N=4
    for row in range(rows):
        for col in range(cols):
            left = col * sub_width
            top = row * sub_height
            right = left + sub_width
            bottom = top + sub_height
            
            # Crop the sub-image
            sub_image = image.crop((left, top, right, bottom))
            sub_images.append(sub_image)
    
    return sub_images


def process_dataset_with_bape(engine, slidevqa_dataset, max_new_tokens, args):
    all_results = []
    for idx, item in enumerate(tqdm(slidevqa_dataset, total=len(slidevqa_dataset))):
        qa_id = item['qa_id']
        question = item['question']
        gold_answer = item['answer']
        
        # Determine which pages to use
        if args.use_oracle_slides:
            # Oracle mode: only use ground-truth evidence pages
            evidence_pages = item.get('evidence_pages', [])
            if not isinstance(evidence_pages, list):
                evidence_pages = [evidence_pages] if evidence_pages is not None else []
            
            # Normalize evidence_pages to integers
            evidence_page_nums = []
            for ev_page in evidence_pages:
                if isinstance(ev_page, int):
                    evidence_page_nums.append(ev_page)
                elif isinstance(ev_page, str) and ev_page.startswith('page_'):
                    evidence_page_nums.append(int(ev_page.split('_')[1]))
                else:
                    try:
                        evidence_page_nums.append(int(ev_page))
                    except:
                        continue
            
            pages_to_collect = evidence_page_nums
            if len(pages_to_collect) == 0:
                logging.warning(f"No evidence_pages found for qa_id: {qa_id}, falling back to all pages")
                pages_to_collect = list(range(1, 21))
        else:
            # Normal mode: collect all pages (page_1 to page_20)
            pages_to_collect = list(range(1, 21))
        
        # Collect page images and create passages
        passages = []
        passage_page_nums = []
        split_name = args.split if args.split else 'test'
        
        # Check if we need to subdivide images
        subdivide_n = getattr(args, 'subdivide_image_into_parts', None)
        
        for page_num in pages_to_collect:
            page_key = f'page_{page_num}'
            if page_key in item and item[page_key] is not None:
                page_image = item[page_key]
                
                # Get image path or PIL Image object
                img_path = None
                img_object = None
                if isinstance(page_image, Image.Image):
                    # If it's already a PIL.Image, use it directly
                    # Ensure it's RGB
                    if page_image.mode != 'RGB':
                        img_object = page_image.convert('RGB')
                    else:
                        img_object = page_image
                    # Also try to get the path for tracking purposes
                    img_filename = f"{split_name}_{qa_id}_page_{page_num}.png"
                    img_path = os.path.join(args.img_basedir, img_filename)
                elif isinstance(page_image, str):
                    # If it's a path string
                    if os.path.isabs(page_image):
                        img_path = page_image
                    else:
                        img_path = os.path.join(args.img_basedir, page_image)
                    
                    # Check if path exists
                    if not os.path.exists(img_path):
                        # Try constructing path using standard format
                        img_filename = f"{split_name}_{qa_id}_page_{page_num}.png"
                        constructed_path = os.path.join(args.img_basedir, img_filename)
                        if os.path.exists(constructed_path):
                            img_path = constructed_path
                        else:
                            # Try jpg extension
                            constructed_path_jpg = os.path.join(args.img_basedir, f"{split_name}_{qa_id}_page_{page_num}.jpg")
                            if os.path.exists(constructed_path_jpg):
                                img_path = constructed_path_jpg
                            else:
                                logging.warning(f"Could not find image for {qa_id} page_{page_num}")
                                continue
                else:
                    continue
                
                # Load image as PIL Image if we need to subdivide or if it's not already loaded
                if subdivide_n is not None:
                    # Need to subdivide, so we must have a PIL Image
                    if img_object is None:
                        img_object = load_image_from_path(img_path)
                        if img_object is None:
                            logging.warning(f"Could not load image for subdivision: {qa_id} page_{page_num}")
                            continue
                    
                    # Subdivide the image
                    sub_images = subdivide_image(img_object, subdivide_n)
                    # Create a passage for each sub-image
                    # z index continues across images: z=0,1,2,3 for first image, z=4,5,6,7 for second image, etc.
                    base_z_idx = len(passages)
                    for sub_idx, sub_img in enumerate(sub_images):
                        z_idx = base_z_idx + sub_idx
                        passage_dict = {
                            'images': [sub_img],
                            'text': '',  # No text content for pages
                            'page_num': page_num,  # Store original page number
                            'sub_image_idx': sub_idx,  # Store sub-image index within the original image (0 to n_parts-1)
                            'z_idx': z_idx  # Store global z index across all sub-images
                        }
                        passages.append(passage_dict)
                        passage_page_nums.append(page_num)  # Keep page_num for compatibility
                else:
                    # No subdivision, use original image
                    # Use PIL Image object if available, otherwise use path
                    image_to_use = img_object if img_object is not None else img_path
                    if image_to_use is None:
                        continue
                    
                    passage_dict = {
                        'images': [image_to_use],
                        'text': '',  # No text content for pages
                        'page_num': page_num  # Store page number for tracking
                    }
                    passages.append(passage_dict)
                    passage_page_nums.append(page_num)
        
        # Limit to retrieval_topk
        if len(passages) > args.retrieval_topk:
            passages = passages[:args.retrieval_topk]
            passage_page_nums = passage_page_nums[:args.retrieval_topk]
        
        if len(passages) == 0:
            logging.warning(f"No valid passages found for qa_id: {qa_id}")
            continue
        
        # Get evidence pages for ground truth tracking
        evidence_pages = item.get('evidence_pages', [])
        if not isinstance(evidence_pages, list):
            evidence_pages = [evidence_pages] if evidence_pages is not None else []
        
        # Normalize evidence_pages to integers
        evidence_page_nums = []
        for ev_page in evidence_pages:
            if isinstance(ev_page, int):
                evidence_page_nums.append(ev_page)
            elif isinstance(ev_page, str) and ev_page.startswith('page_'):
                evidence_page_nums.append(int(ev_page.split('_')[1]))
            else:
                try:
                    evidence_page_nums.append(int(ev_page))
                except:
                    continue
        
        # Find gt_passage_in_zidx
        gt_passage_in_zidx = -1
        if evidence_page_nums:
            # Find first evidence page that is in our passages
            for ev_page_num in evidence_page_nums:
                if ev_page_num in passage_page_nums:
                    gt_passage_in_zidx = passage_page_nums.index(ev_page_num)
                    break
        
        # Format prompt
        prompt_template_str = getattr(args, 'prompt_template_str', VLM_PROMPT_FOR_VQA)
        prompt_text = prompt_template_str + "<<<EVIDENCE>>>"
        prompt_text += f"\n[QUESTION] {question}"
        if args.prefill_ans_token:
            prompt_text += f"\n[ANSWER]"
        
        # Create x context (no shared main image in SlideVQA)
        x = {"text": prompt_text, "image": None}
        
        # Prepare passage prior (only prior_head is supported)
        if args.passage_prior == "prior_head":
            passage_prior = None
        else:
            raise NotImplementedError(f"Passage prior {args.passage_prior} not implemented. Only 'prior_head' is supported for SlideVQA.")
        
        # Generate with BAPE (v1 greedy mode)
        if args.inference_engine_version == "v1":
            generated_token_ids, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits, log_posterior_over_steps, inference_stats = engine.generate(
                x,
                passages,
                passage_prior,
                max_new_tokens=max_new_tokens,
                return_stats=True,
                return_posterior_over_steps=True,
            )
        else:
            raise NotImplementedError(f"Inference engine version {args.inference_engine_version} not implemented. Only 'v1' is supported.")
        
        # Decode response
        response = engine.backend.processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        generated_answer = response.split('[ANSWER]')[-1].strip() if '[ANSWER]' in response else response.strip()
        
        # Monitoring
        z_dominant_idx = posterior_max_idx
        log_document_posterior = np.array(log_posterior_over_steps, dtype=float).T.tolist() if log_posterior_over_steps else []
        
        # Sort passages by prior_logits for recall computation
        prior_sorted_passage_ids = []
        if prior_logits is not None and len(prior_logits) > 0:
            prior_sorted_indices = np.argsort(prior_logits)[::-1]  # Sort descending
            prior_sorted_passage_ids = [passage_page_nums[i] for i in prior_sorted_indices]
        
        # Get main image path for output (first evidence page or first available page)
        # Convert PIL Image to path string if needed for CSV output
        img_path = None
        if evidence_page_nums and evidence_page_nums[0] in passage_page_nums:
            ev_idx = passage_page_nums.index(evidence_page_nums[0])
            if passages[ev_idx]['images']:
                img_val = passages[ev_idx]['images'][0]
                if isinstance(img_val, Image.Image):
                    # For CSV, use a placeholder or construct path
                    img_filename = f"{split_name}_{qa_id}_page_{evidence_page_nums[0]}.png"
                    img_path = os.path.join(args.img_basedir, img_filename)
                else:
                    img_path = img_val
        elif passages:
            if passages[0]['images']:
                img_val = passages[0]['images'][0]
                if isinstance(img_val, Image.Image):
                    # For CSV, use a placeholder or construct path
                    img_filename = f"{split_name}_{qa_id}_page_{passage_page_nums[0]}.png"
                    img_path = os.path.join(args.img_basedir, img_filename)
                else:
                    img_path = img_val
        
        all_results.append({
            'qa_id': qa_id,
            'question': question,
            'img_path': img_path,
            'gold_answer': gold_answer,
            'response': response,
            'generated_answer': generated_answer,
            'passages': passages,
            'prompt_text': prompt_text,
            'posterior_max_idx': posterior_max_idx,
            'prior_max_idx': prior_max_idx,
            'gt_passage_in_zidx': gt_passage_in_zidx,
            'z_dominant_idx': z_dominant_idx,
            'dominant_passage_is_gt': z_dominant_idx == gt_passage_in_zidx and z_dominant_idx != -1,
            'prior_passage_is_gt': prior_max_idx == gt_passage_in_zidx and prior_max_idx != -1,
            'log_all_tokens_llk': log_all_tokens_llk,
            'log_posterior_over_steps': log_posterior_over_steps,  # N x K, one posterior vector per generated token
            'log_document_posterior': log_document_posterior,  # K x N, one generated-token trace per document
            'prior_logits': prior_logits if prior_logits is not None else None,
            'prior_sorted_passage_ids': prior_sorted_passage_ids,
            'evidence_page_nums': evidence_page_nums,  # Store all ground-truth evidence pages
            'passage_page_nums': passage_page_nums,  # Store all retrieved passage page numbers
            'prefill_ms': inference_stats.get('prefill_ms', 0.0),
            'prefill_forward_ms': inference_stats.get('prefill_forward_ms', 0.0),
            'prior_head_ms': inference_stats.get('prior_head_ms', 0.0),
            'decode_ms': inference_stats.get('decode_ms', 0.0),
            'decode_tokens': inference_stats.get('decode_tokens', 0),
            'decode_tokens_per_ms': inference_stats.get('decode_tokens_per_ms', 0.0),
            'input_tokens': inference_stats.get('input_tokens', 0),
            'output_tokens': inference_stats.get('output_tokens', len(generated_token_ids)),
            'num_prefill_branches': inference_stats.get('num_prefill_branches', len(passages)),
            'dynamic_k_top_p': args.dynamic_k_top_p,
            'retrieval_topk': args.retrieval_topk,
        })
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset settings
    parser.add_argument("--hf_dataset_path", type=str, default="NTT-hil-insight/SlideVQA",
                        help="HuggingFace dataset path or local path")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--take_n", type=int, default=-1,
                        help="Number of examples to take (-1 for all)")
    parser.add_argument("--img_basedir", type=str, default="../../shared_space/vqa_data/KBVQA_data/SlideVQA",
                        help="Base directory for image paths")
    parser.add_argument("--prompt_template", type=str, default=None,
                        help="Path to prompt template file")
    parser.add_argument("--prefill_ans_token", action="store_true",
                        help="Prefill [ANSWER] token in prompt")
    parser.add_argument("--use_oracle_slides", action="store_true",
                        help="Use only ground-truth slides (evidence_pages) instead of all slides")
    parser.add_argument("--offset", type=int, default=0)

    # Retrieval settings
    parser.add_argument("--retrieval_topk", type=int, default=5,
                        help="Number of pages to use as passages")

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
    parser.add_argument("--passage_prior", type=str, default="prior_head",
                        choices=["prior_head"],
                        help="Only 'prior_head' is supported for SlideVQA")
    parser.add_argument("--prior_head_path", type=str, default=None)
    parser.add_argument("--prior_head_modeling", type=str, default="mlp_head", choices=["mlp_head", "linear_head"])
    parser.add_argument("--prior_head_num_layers", type=int, default=2)
    parser.add_argument("--prior_head_proj_dim", type=int, default=1024)
    parser.add_argument("--auto_resolve_prior_head", action="store_true", default=True, help="Automatically use <adapter>/prior_head.pt when prior_head_path is not provided.")
    parser.add_argument("--dynamic_k_top_p", type=float, default=None)
    parser.add_argument("--max_batch_size_per_forward", type=int, default=5)
    parser.add_argument("--hidden_state_offset", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--inference_engine_version", type=str, default="v1", choices=["v1"],
                        help="Only 'v1' (greedy) is supported for SlideVQA")
    parser.add_argument("--return_n_sequences", type=int, default=1)

    # Evaluation setting
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--use_bem", action="store_true",
                        help="Use BEM (BERT-based Equivalence Model) for evaluation in addition to EM")
    parser.add_argument("--subdivide_image_into_parts", type=int, default=None,
                        help="Subdivide each image into N sub-images (N must be power of 2). N=2 divides height, N=4 divides both height and width (2x2 grid), etc.")

    # Saving settings
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()

    output_filepath = f"{args.exp_name}/inference_results.csv"
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    total_inference_runtime_sec = None
    
    if os.path.exists(output_filepath) and args.use_cache:
        all_results = pd.read_csv(output_filepath)
        print(f"Loaded cached results from {output_filepath}")
    else:
        # Load SlideVQA dataset from HuggingFace
        logger.info(f"Loading SlideVQA dataset from {args.hf_dataset_path}")
        try:
            # Try loading from disk first
            slidevqa_dataset = load_from_disk(args.hf_dataset_path)
            if args.split in slidevqa_dataset:
                slidevqa_dataset = slidevqa_dataset[args.split]
        except:
            # Load from HuggingFace Hub
            slidevqa_dataset = load_dataset(args.hf_dataset_path, split=args.split)
        
        if args.take_n > 0:
            print(f"Taking {args.take_n} examples from the dataset, starting from index {args.offset}.")
            slidevqa_dataset = slidevqa_dataset.shuffle(seed=args.seed).select([i for i in range(args.offset, args.offset + args.take_n)])
        else:
            print("Using the entire dataset.")
        
        print(f"Dataset loaded with {len(slidevqa_dataset)} items")

        # Get prompt template
        prompt_template_str = get_prompt_template(args.prompt_template)
        print("--------------------------------")
        if args.prompt_template:
            print(f"Read prompt template from {args.prompt_template}.")
        else:
            print("Using global VLM_PROMPT_FOR_VQA (matching ragk_slidevqa.py).")
        print(f"Prompt template: {prompt_template_str}")
        print("--------------------------------")
        
        # Store prompt template in args for use in process_dataset_with_bape
        args.prompt_template_str = prompt_template_str

        backend = HFQwen2VLBackend(
            model_path=args.model_path,
            processor_path=args.processor_path if args.processor_path is not None and args.processor_path != '' else args.model_path,
            adapter_name_or_path=args.adapter_name_or_path,
            max_batch_size_per_forward=args.max_batch_size_per_forward
        )

        prior_head_config = {
            "modeling": args.prior_head_modeling,
            "num_layers": args.prior_head_num_layers,
            "proj_dim": args.prior_head_proj_dim,
        }
        # Initialize Inference Engine
        engine = BAPEInferenceEngine(
            backend, 
            prior_head_path=args.prior_head_path, 
            prior_head_config=prior_head_config, 
            dynamic_k_top_p=args.dynamic_k_top_p, 
            hidden_state_offset=args.hidden_state_offset, 
            num_beams=args.num_beams
        )
        
        inference_start_time = time.perf_counter()
        all_results = process_dataset_with_bape(engine, slidevqa_dataset, max_new_tokens=args.max_new_tokens, args=args)
        total_inference_runtime_sec = time.perf_counter() - inference_start_time
        all_results = pd.DataFrame(all_results)
        del engine
        torch.cuda.empty_cache()

        # save all_results to a CSV file
        all_results.to_csv(output_filepath, index=False)
        print(f"Results saved to {output_filepath}")

    if not isinstance(all_results, pd.DataFrame):
        all_results = pd.DataFrame(all_results)
    write_inference_report(all_results, os.path.dirname(output_filepath), total_inference_runtime_sec=total_inference_runtime_sec)

    # Evaluation
    if args.do_eval:
        output_filepath = f"{args.exp_name}/marked_inference_results.csv"
        score_filepath = f"{args.exp_name}/scores.json"
        # Always re-run evaluation, even if cache exists
        df = all_results.copy()
        
        def relaxed_exact_match(gold_answer: str, generated_answer: str) -> bool:
            """
            Relaxed Exact Match: Check if gold_answer appears in generated_answer.
            Uses simple 'in' check with basic normalization (lowercase, strip).
            """
            if not gold_answer or not generated_answer:
                return False
            
            # Basic normalization: lowercase and strip whitespace
            gold_normalized = str(gold_answer).lower().strip()
            generated_normalized = str(generated_answer).lower().strip()
            
            # Check if gold answer is contained in generated answer
            return gold_normalized in generated_normalized
        
        def evidence_em_at_k(gt_pages, pred_pages, k=None):
            """
            Evidence Exact Match: Check if the top-k predicted pages exactly match the ground-truth pages.
            
            Args:
                gt_pages: List of ground-truth page numbers
                pred_pages: List of predicted page numbers (in order)
                k: Number of top pages to consider. If None, uses len(gt_pages)
            
            Returns:
                True if the top-k predicted pages form an exact set match with ground-truth pages.
                - If k=None: checks if top |GT| predicted pages exactly match GT
                - If k is specified: checks if all GT pages are in top k AND top |GT| predicted pages exactly match GT
            """
            if not gt_pages:
                # If no GT pages, EvidenceEM is True only if we're checking for empty match
                return (k == 0) or (k is None and not pred_pages)
            
            if not pred_pages:
                return False
            
            # Remove duplicates while preserving order
            gt_unique = list(dict.fromkeys(gt_pages))
            pred_unique = list(dict.fromkeys(pred_pages))
            
            # Determine k
            if k is None:
                k = len(gt_unique)
            
            if k == 0:
                return len(gt_unique) == 0
            
            # Get top-k predicted pages
            topk_pred = pred_unique[:k]
            
            # Check if all GT pages are in top k
            return set(gt_unique).issubset(set(topk_pred))
            
        # Compute relaxed Exact Match scores
        exact_matches = []
        token_f1_scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing Relaxed Exact Match scores"):
            gold_answer = str(row['gold_answer']) if pd.notna(row['gold_answer']) else ""
            generated_answer = str(row['generated_answer']) if pd.notna(row['generated_answer']) else ""
            
            try:
                em_score = relaxed_exact_match(gold_answer, generated_answer)
                exact_matches.append(1 if em_score else 0)
                token_f1_scores.append(compute_token_f1(gold_answer, generated_answer))
            except Exception as e:
                logging.warning(f"Error computing EM for qa_id {row['qa_id']}: {e}")
                exact_matches.append(0)
                token_f1_scores.append(0.0)
        
        df['exact_match'] = exact_matches
        df['token_f1'] = token_f1_scores
        
        # Compute overall accuracy
        overall_accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        
        dict_to_report = {
            'relaxed_exact_match_accuracy': overall_accuracy,
            'token_f1': float(np.mean(token_f1_scores)) if token_f1_scores else 0.0,
            'total_samples': len(exact_matches),
            'correct_predictions': sum(exact_matches)
        }
        
        # Add Monitoring results
        dominant_passage_hit_rate = df[df['gt_passage_in_zidx'] != -1]['dominant_passage_is_gt'].mean() if len(df[df['gt_passage_in_zidx'] != -1]) > 0 else 0.0
        retrieval_hit_rate = len(df[df['gt_passage_in_zidx'] != -1]) / len(df) if len(df) > 0 else 0.0
        correct_ignore_rate = len(df[(df['gt_passage_in_zidx'] == -1) & (df['z_dominant_idx'] == args.retrieval_topk)]) / len(df[df['gt_passage_in_zidx'] == -1]) if len(df[df['gt_passage_in_zidx'] == -1]) > 0 else 0
        prior_hit_rate = df[df['prior_max_idx'] != -1]['prior_passage_is_gt'].mean() if len(df[df['prior_max_idx'] != -1]) > 0 else 0.0

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

        # Compute EvidenceEM metrics
        prior_evidence_em = []
        prior_evidence_em_at_topk = []
        retrieval_evidence_em = []
        retrieval_evidence_em_at_topk = []
        
        # Initialize lists for prior_evidence_em@k for k=1 to 20
        max_k_for_evidence_em = 20
        prior_evidence_em_at_k = {k: [] for k in range(1, max_k_for_evidence_em + 1)}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing EvidenceEM metrics"):
            # Get ground-truth pages
            evidence_page_nums = row.get('evidence_page_nums', [])
            if isinstance(evidence_page_nums, str):
                # Handle case where it might be stored as string representation
                try:
                    import ast
                    evidence_page_nums = ast.literal_eval(evidence_page_nums)
                except:
                    evidence_page_nums = []
            elif not isinstance(evidence_page_nums, (list, tuple, np.ndarray)):
                # Check if it's a scalar NaN value
                try:
                    if pd.isna(evidence_page_nums):
                        evidence_page_nums = []
                except (ValueError, TypeError):
                    # If it's an array or other type that can't be checked with pd.isna
                    evidence_page_nums = []
            if not isinstance(evidence_page_nums, (list, tuple, np.ndarray)):
                evidence_page_nums = []
            
            # Get predicted pages from prior ranking
            prior_sorted_ids = row.get('prior_sorted_passage_ids', [])
            if isinstance(prior_sorted_ids, str):
                try:
                    import ast
                    prior_sorted_ids = ast.literal_eval(prior_sorted_ids)
                except:
                    prior_sorted_ids = []
            elif not isinstance(prior_sorted_ids, (list, tuple, np.ndarray)):
                # Check if it's a scalar NaN value
                try:
                    if pd.isna(prior_sorted_ids):
                        prior_sorted_ids = []
                except (ValueError, TypeError):
                    # If it's an array or other type that can't be checked with pd.isna
                    prior_sorted_ids = []
            if not isinstance(prior_sorted_ids, (list, tuple, np.ndarray)):
                prior_sorted_ids = []
            
            # Get predicted pages from retrieval order
            passage_page_nums = row.get('passage_page_nums', [])
            if isinstance(passage_page_nums, str):
                try:
                    import ast
                    passage_page_nums = ast.literal_eval(passage_page_nums)
                except:
                    passage_page_nums = []
            elif not isinstance(passage_page_nums, (list, tuple, np.ndarray)):
                # Check if it's a scalar NaN value
                try:
                    if pd.isna(passage_page_nums):
                        passage_page_nums = []
                except (ValueError, TypeError):
                    # If it's an array or other type that can't be checked with pd.isna
                    passage_page_nums = []
            if not isinstance(passage_page_nums, (list, tuple, np.ndarray)):
                passage_page_nums = []
            
            # Compute EvidenceEM for prior ranking
            prior_em = evidence_em_at_k(evidence_page_nums, prior_sorted_ids, k=None)
            prior_em_topk = evidence_em_at_k(evidence_page_nums, prior_sorted_ids, k=args.retrieval_topk)
            
            
            prior_evidence_em.append(1 if prior_em else 0)
            prior_evidence_em_at_topk.append(1 if prior_em_topk else 0)
            
            # Compute EvidenceEM@k for k=1 to 20
            for k in range(1, max_k_for_evidence_em + 1):
                prior_em_k = evidence_em_at_k(evidence_page_nums, prior_sorted_ids, k=k)
                prior_evidence_em_at_k[k].append(1 if prior_em_k else 0)
        
        # Add EvidenceEM columns to dataframe
        df['prior_evidence_em'] = prior_evidence_em
        df['prior_evidence_em_at_topk'] = prior_evidence_em_at_topk
        
        # Compute average EvidenceEM scores
        prior_evidence_em_acc = np.mean(prior_evidence_em) if prior_evidence_em else 0.0
        prior_evidence_em_topk_acc = np.mean(prior_evidence_em_at_topk) if prior_evidence_em_at_topk else 0.0
        
        # Compute average EvidenceEM@k for k=1 to 20
        prior_evidence_em_at_k_avg = {}
        for k in range(1, max_k_for_evidence_em + 1):
            if prior_evidence_em_at_k[k]:
                prior_evidence_em_at_k_avg[k] = float(np.mean(prior_evidence_em_at_k[k]))
            else:
                prior_evidence_em_at_k_avg[k] = 0.0
        
        # Add to report
        dict_to_report['prior_evidence_em@gt_len'] = float(prior_evidence_em_acc)
        dict_to_report['prior_evidence_em@topk'] = float(prior_evidence_em_topk_acc)
        dict_to_report['prior_evidence_em_at_k'] = prior_evidence_em_at_k_avg

        print("--------------------------------")
        print("Evaluation results:")
        print(f"Relaxed Exact Match Accuracy (gold answer in generated answer): {overall_accuracy:.4f} ({sum(exact_matches)}/{len(exact_matches)})")
        print(f"\nEvidenceEM metrics:")
        print(f"  Prior EvidenceEM@GT-len: {prior_evidence_em_acc:.4f} ({sum(prior_evidence_em)}/{len(prior_evidence_em)})")
        print(f"  Prior EvidenceEM@TopK: {prior_evidence_em_topk_acc:.4f} ({sum(prior_evidence_em_at_topk)}/{len(prior_evidence_em_at_topk)})")
        print(f"\nPrior EvidenceEM@K metrics (k=1 to 20):")
        for k in range(1, max_k_for_evidence_em + 1):
            if k in prior_evidence_em_at_k_avg:
                print(f"  Prior EvidenceEM@{k}: {prior_evidence_em_at_k_avg[k]:.4f}")
        
        # BEM Evaluation (optional, only if --use_bem is enabled)
        if args.use_bem:
            sys.path.append('./src/evaluation')
            import evaluation_utils
            import tensorflow as tf
            import multiprocessing
            
            # Disable GPU for multiprocessing
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            tf.config.set_visible_devices([], 'GPU')
            
            def process_row_mp_slidevqa(row):
                """Process single row for BEM evaluation with multiprocessing."""
                idx, row_data = row
                question = str(row_data['question']) if pd.notna(row_data['question']) else ""
                gold_answer = str(row_data['gold_answer']) if pd.notna(row_data['gold_answer']) else ""
                generated_answer = str(row_data['generated_answer']) if pd.notna(row_data['generated_answer']) else ""
                
                # Convert single answer to list (BEM expects reference_list)
                reference_list = [gold_answer] if gold_answer else []
                
                # Use default question_type for SlideVQA
                question_type = 'automatic'
                
                try:
                    score = evaluation_utils.evaluate_example(
                        question=question,
                        reference_list=reference_list,
                        candidate=generated_answer,
                        question_type=question_type
                    )
                    return score
                except Exception as e:
                    logging.warning(f"Error computing BEM for qa_id {row_data['qa_id']}: {e}")
                    return 0.0
            
            # Use multiprocessing with 8 processes (matching EVQA)
            num_processes = 8
            with multiprocessing.Pool(processes=num_processes) as pool:
                bem_scores = list(tqdm(
                    pool.imap(process_row_mp_slidevqa, df.iterrows(), chunksize=1),
                    total=len(df),
                    desc="Computing BEM scores"
                ))
            
            # Add BEM scores to dataframe
            df['bem_score'] = bem_scores
            
            # Calculate average BEM score
            bem_score_accuracy = sum(bem_scores) / len(bem_scores) if bem_scores else 0.0
            
            # Calculate number of correct predictions (BEM score >= 0.5, matching EVQA threshold)
            bem_correct = sum(1 for score in bem_scores if score >= 0.5)
            
            # Add to report
            dict_to_report['bem_score_accuracy'] = bem_score_accuracy
            dict_to_report['bem_correct_predictions'] = bem_correct
            
            print(f"BEM Score Accuracy: {bem_score_accuracy:.4f} ({bem_correct}/{len(bem_scores)} correct, threshold >= 0.5)")
        
        print("--------------------------------")
        print(dict_to_report)
        print("--------------------------------")
        
        # Save results
        df.to_csv(output_filepath, index=False)
        with open(score_filepath, 'w') as f:
            json.dump(dict_to_report, f, indent=2)
        
        print(f"Evaluation results saved to {output_filepath}")
        print(f"Scores saved to {score_filepath}")

