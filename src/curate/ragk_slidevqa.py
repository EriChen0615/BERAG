"""
Script for converting SlideVQA dataset to ShareGPT format for BEFT training.

SlideVQA dataset: https://huggingface.co/datasets/NTT-hil-insight/SlideVQA
    Columns:
    - qa_id: unique ID
    - question: quesiton text
    - answer
    - arithemetic_expression
    - evidence_pages: List 
    - page_1, page_2, ..., page_20
    - answer


ShareGPT format: refer to ragk_answer_ppl.py
NOTE
    - In ragk_answer_ppl.py, the passages are provided as plain text. Here, create a dictionary of the following key-values:
        - 'images': [image_path, ...]
        - 'text': "<text>"
"""

import sys
sys.path.append('./src')
from datasets import load_dataset
import argparse
import json
from tqdm import tqdm
import os
from functools import partial
import gc
import numpy as np
import random
from PIL import Image

VLM_PROMPT_FOR_VQA = (
    "Answer the question directly without explanations based on the provided slides."
    "<<<EVIDENCE>>>"
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
IMG_TOKEN_COUNTS = 400  # (512/28)^2 approx 400

def ensure_json_serializable(obj):
    """Recursively convert objects to JSON-serializable types"""
    if isinstance(obj, Image.Image):
        # Convert Image objects to None or empty string
        return None
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert numpy types to Python native types
        return obj.item()
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to lists
        return obj.tolist()
    elif isinstance(obj, dict):
        # Recursively process dictionaries
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively process lists and tuples
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Already JSON-serializable
        return obj
    else:
        # For unknown types, try to convert to string
        try:
            return str(obj)
        except:
            return None

def convert_to_sharegpt_format(dataset, output_dir, mode='beft', args=None):
    """
    Convert dataset items to ShareGPT format and save as a JSON file.
    Args:
        dataset: HuggingFace dataset with processed items.
        output_dir: Path to save the ShareGPT formatted JSON.
        mode: 'beft' or 'sft' mode
        args: arguments object
    """
    sharegpt_data = []
    print(f"Convert to sharegpt format for {mode} training...")
    
    # Process dataset directly without converting to list
    for item in tqdm(dataset, desc="Converting to ShareGPT format"):
        if mode == 'beft':
            # Ensure img_path is a string, not an Image object
            img_path = item['img_path']
            if isinstance(img_path, Image.Image):
                img_path = None
            elif not isinstance(img_path, str):
                img_path = str(img_path) if img_path is not None else None
            
            # Ensure passages are JSON-serializable
            passages = item['passages']
            if passages is not None:
                # Clean passages to ensure all fields are serializable
                cleaned_passages = []
                for passage in passages:
                    if isinstance(passage, dict):
                        cleaned_passage = {}
                        # Ensure images field is a list of strings
                        if 'images' in passage:
                            images = passage['images']
                            if isinstance(images, list):
                                cleaned_images = []
                                for img in images:
                                    if isinstance(img, Image.Image):
                                        cleaned_images.append(None)
                                    elif isinstance(img, str):
                                        cleaned_images.append(img)
                                    else:
                                        cleaned_images.append(str(img) if img is not None else None)
                                cleaned_passage['images'] = cleaned_images
                            else:
                                cleaned_passage['images'] = []
                        # Ensure text field is a string
                        if 'text' in passage:
                            text = passage['text']
                            if isinstance(text, (str, int, float)):
                                cleaned_passage['text'] = str(text)
                            else:
                                cleaned_passage['text'] = str(text) if text is not None else ''
                        cleaned_passages.append(cleaned_passage)
                    else:
                        # If passage is not a dict, skip it or create a default
                        cleaned_passages.append({'images': [], 'text': ''})
                passages = cleaned_passages
            else:
                passages = []
            
            # Ensure passage_scores are JSON-serializable
            passage_scores = item.get('passage_scores', [])
            if passage_scores is not None:
                cleaned_scores = []
                for score in passage_scores:
                    if isinstance(score, (np.integer, np.floating)):
                        cleaned_scores.append(float(score.item()))
                    elif isinstance(score, (int, float)):
                        cleaned_scores.append(float(score))
                    else:
                        try:
                            cleaned_scores.append(float(score))
                        except:
                            cleaned_scores.append(0.0)
                passage_scores = cleaned_scores
            else:
                passage_scores = []
            
            conversation = {
                "messages": [
                    {
                        "content": f"<image> {item['prompt']}", 
                        "role": "user"
                    },
                    {
                        "content": f"{item['gold_answer']}", 
                        "role": "assistant"
                    }
                ],
                # No main images - each passage has its own image, and there's no shared main image in this setup
                "images": [],
                "gt_passage_idx": item['gt_passage_idx'] if isinstance(item['gt_passage_idx'], list) else [],
                "passages": passages,
                "passage_scores": passage_scores
            }
            if args and args.add_separate_prompt_for_prior:
                prior_prompt_content = item.get('prior_prompt', '')
                question = item.get('question', '')
                if prior_prompt_content and question:
                    prior_content = prior_prompt_content.replace('<<<QUESTION>>>', question)
                    conversation['prior_prompt'] = [
                        {
                            "content": f"<image> {prior_content}",
                            "role": "user"
                        },
                    ]
        elif mode == 'sft':
            # Collect all images from passages
            all_images = []
            passages = item.get('passages', [])
            if passages is not None:
                for passage in passages:
                    if isinstance(passage, dict) and 'images' in passage:
                        images = passage['images']
                        if isinstance(images, list):
                            for img in images:
                                if isinstance(img, str) and img:
                                    all_images.append(img)
                                elif isinstance(img, Image.Image):
                                    # Skip Image objects in SFT mode
                                    pass
                                elif img is not None:
                                    img_str = str(img)
                                    if img_str:
                                        all_images.append(img_str)
            
            # Shuffle the images to randomize order (GT images won't necessarily be first)
            random.shuffle(all_images)
            
            # Ensure all images are strings and valid
            cleaned_images = []
            for img in all_images:
                if isinstance(img, str):
                    cleaned_images.append(img)
                elif img is not None:
                    cleaned_images.append(str(img))
            
            # Add <image> tokens at the beginning of prompt to match the number of images
            num_images = len(cleaned_images)
            image_tokens = " ".join(["<image>"] * num_images) if num_images > 0 else "<image>"
            prompt_with_images = f"{image_tokens} {item['prompt']}"
            
            conversation = {
                "messages": [
                    {
                        "content": prompt_with_images, 
                        "role": "user"
                    },
                    {
                        "content": f"{item['gold_answer']}", 
                        "role": "assistant"
                    }
                ],
                # All passage images are collected into main images field, shuffled
                "images": cleaned_images,
                "gt_passage_idx": item['gt_passage_idx'] if isinstance(item['gt_passage_idx'], list) else [],
                "passages": [],  # Empty in SFT mode since images are in main field
                "passage_scores": []  # Empty in SFT mode
            }
            if args and args.add_separate_prompt_for_prior:
                prior_prompt_content = item.get('prior_prompt', '')
                question = item.get('question', '')
                if prior_prompt_content and question:
                    prior_content = prior_prompt_content.replace('<<<QUESTION>>>', question)
                    conversation['prior_prompt'] = [
                        {
                            "content": f"<image> {prior_content}",
                            "role": "user"
                        },
                    ]
        else:
            raise NotImplementedError(f"convert_to_sharegpt_format {mode}")
        
        # Final cleanup to ensure everything is JSON-serializable
        conversation = ensure_json_serializable(conversation)
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

def save_image_if_needed(page_image, qa_id, page_num, split_name, img_base_dir, skip_check=False):
    """
    Save image to local file if it's an Image object or bytes, using the same naming format as download_slidevqa_images.py
    Format: {split_name}_{qa_id}_page_{page_num}.png
    
    Returns the absolute file path if saved or found, or None if not needed/available
    Note: If file exists but is 0 bytes, it will be re-saved.
    
    Args:
        skip_check: If True, skip file existence checks and assume all files are valid.
                   This significantly speeds up processing when images are already saved.
    """
    if page_image is None:
        return None
    
    # Ensure img_base_dir is absolute
    img_base_dir = os.path.abspath(img_base_dir)
    img_filename = f"{split_name}_{qa_id}_page_{page_num}.png"
    img_path = os.path.join(img_base_dir, img_filename)
    
    # Helper function to check if file exists and is valid (non-zero size)
    def is_valid_file(filepath):
        """Check if file exists and has non-zero size"""
        if skip_check:
            # Skip disk check, assume file is valid
            return True
        if not os.path.exists(filepath):
            return False
        return os.path.getsize(filepath) > 0
    
    # Check if it's already a string path
    if isinstance(page_image, str):
        # Check if the file exists at the expected location and is valid
        if is_valid_file(img_path):
            # File exists and is valid, return absolute path
            return os.path.abspath(img_path)
        # If it's already a path string, check if that file exists and is valid
        if is_valid_file(page_image):
            return os.path.abspath(page_image)
        # If it's a relative path, try with img_base_dir
        if not os.path.isabs(page_image):
            full_path = os.path.join(img_base_dir, page_image)
            if is_valid_file(full_path):
                return os.path.abspath(full_path)
            # Try to resolve relative path from current working directory
            try:
                resolved_path = os.path.abspath(page_image)
                if is_valid_file(resolved_path):
                    return resolved_path
            except:
                pass
        # Return as absolute path if possible (even if invalid, we'll try to save it)
        try:
            return os.path.abspath(page_image)
        except:
            return page_image
    
    # If it's an Image object, save it
    if isinstance(page_image, Image.Image):
        # Re-save if file doesn't exist or is 0 bytes
        if skip_check:
            # Skip saving if check is disabled (assume file already exists)
            return os.path.abspath(img_path)
        should_save = not os.path.exists(img_path) or os.path.getsize(img_path) == 0
        
        if should_save:
            # Save the image
            os.makedirs(img_base_dir, exist_ok=True)
            try:
                page_image.save(img_path)
                return os.path.abspath(img_path)
            except Exception as e:
                print(f"Warning: Could not save Image object for {qa_id} page {page_num}: {e}")
                return None
        else:
            # File exists and is valid
            return os.path.abspath(img_path)
    
    # If it's bytes, try to load as image and save
    if isinstance(page_image, bytes):
        try:
            from io import BytesIO
            img = Image.open(BytesIO(page_image))
            
            # Re-save if file doesn't exist or is 0 bytes
            if skip_check:
                # Skip saving if check is disabled (assume file already exists)
                return os.path.abspath(img_path)
            should_save = not os.path.exists(img_path) or os.path.getsize(img_path) == 0
            
            if should_save:
                # Save the image
                os.makedirs(img_base_dir, exist_ok=True)
                img.save(img_path)
                return os.path.abspath(img_path)
            else:
                # File exists and is valid
                return os.path.abspath(img_path)
        except Exception as e:
            print(f"Warning: Could not save image from bytes for {qa_id} page {page_num}: {e}")
            return None
    
    # Unknown type - try to convert to string
    try:
        path_str = str(page_image)
        if is_valid_file(path_str):
            return os.path.abspath(path_str)
        return path_str
    except:
        return None

def subdivide_image_for_augmentation(image, rows, cols):
    """
    Subdivide an image into a grid of sub-images.
    
    Args:
        image: PIL Image object
        rows: Number of rows in the grid
        cols: Number of columns in the grid
    
    Returns:
        List of PIL Image objects in row-wise order (top-left, top-right, ..., bottom-left, bottom-right)
    """
    width, height = image.size
    
    # Calculate sub-image dimensions
    sub_width = width // cols
    sub_height = height // rows
    
    sub_images = []
    
    # Extract sub-images in row-wise order
    # For 2x2: top-left=0, top-right=1, bottom-left=2, bottom-right=3
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

def save_subdivided_image(sub_image, qa_id, page_num, part_idx, split_name, img_base_dir):
    """
    Save a subdivided image with proper naming convention.
    
    Args:
        sub_image: PIL Image object to save
        qa_id: Question-answer ID
        page_num: Page number
        part_idx: Part index (0, 1, 2, 3, ...)
        split_name: Dataset split name (train, test, etc.)
        img_base_dir: Base directory for images
    
    Returns:
        Absolute path to saved image, or None if saving failed
    """
    if sub_image is None:
        return None
    
    # Ensure img_base_dir is absolute
    img_base_dir = os.path.abspath(img_base_dir)
    img_filename = f"{split_name}_{qa_id}_page_{page_num}_part-{part_idx}.png"
    img_path = os.path.join(img_base_dir, img_filename)
    
    try:
        # Save the image
        os.makedirs(img_base_dir, exist_ok=True)
        sub_image.save(img_path)
        return os.path.abspath(img_path)
    except Exception as e:
        print(f"Warning: Could not save subdivided image for {qa_id} page {page_num} part {part_idx}: {e}")
        return None

def add_prefix_and_form_prompt(batch, args):
    """Process a batch of data efficiently using HuggingFace batched processing"""
    batch_size = len(batch['qa_id'])
    
    # Process image paths - get the first page as the main image
    img_paths = []
    prompts = [''] * batch_size
    all_passage_contents = [''] * batch_size
    all_passage_scores = [-1] * batch_size
    all_gt_passage_idx = [[] for _ in range(batch_size)]  # Changed to list of lists
    
    # Get output directory for calculating relative paths
    output_dir = getattr(args, 'output_dir', None)
    if output_dir:
        output_dir = os.path.abspath(output_dir)
    
    # Process each item in the batch
    for idx in range(batch_size):
        # Get evidence pages (ground truth pages)
        evidence_pages = batch['evidence_pages'][idx]
        if not isinstance(evidence_pages, list):
            evidence_pages = [evidence_pages] if evidence_pages is not None else []
        
        # Get all available pages (page_1 to page_20)
        # Use the same IMG_BASE_DIR as download_slidevqa_images.py
        img_base_dir = args.img_basedir if args.img_basedir else "../../shared_space/vqa_data/KBVQA_data/SlideVQA"
        img_base_dir_abs = os.path.abspath(img_base_dir)
        split_name = args.split if hasattr(args, 'split') and args.split else 'train'
        
        available_pages = []
        for page_num in range(1, 21):
            page_key = f'page_{page_num}'
            if page_key in batch and batch[page_key][idx] is not None:
                page_image = batch[page_key][idx]
                qa_id = batch['qa_id'][idx]
                
                # Try to save image if needed (if it's Image object or bytes)
                skip_check = getattr(args, 'skip_image_path_exist_check', False)
                saved_path = save_image_if_needed(page_image, qa_id, page_num, split_name, img_base_dir_abs, skip_check=skip_check)
                
                if saved_path:
                    # saved_path is now always an absolute path from save_image_if_needed
                    # Convert to relative path from output_dir for the dataset file
                    if os.path.isabs(saved_path):
                        if output_dir:
                            try:
                                # Calculate relative path from output_dir to the image file
                                page_path = os.path.relpath(saved_path, output_dir).replace('\\', '/')
                            except ValueError:
                                # If paths are on different drives (Windows), keep absolute or use img_basedir
                                if args.img_basedir:
                                    # Extract filename and use img_basedir
                                    filename = os.path.basename(saved_path)
                                    page_path = f"{args.img_basedir}/{filename}"
                                else:
                                    # Keep absolute path as fallback
                                    page_path = saved_path
                        else:
                            # No output_dir, calculate relative path from current working directory
                            try:
                                page_path = os.path.relpath(saved_path).replace('\\', '/')
                            except ValueError:
                                # Keep absolute path as fallback
                                page_path = saved_path
                    else:
                        # Already a relative path, use as is
                        page_path = saved_path
                elif isinstance(page_image, str):
                    # Already a path string - convert to relative path from output_dir if absolute
                    if os.path.isabs(page_image):
                        if output_dir:
                            try:
                                page_path = os.path.relpath(page_image, output_dir).replace('\\', '/')
                            except ValueError:
                                # Keep absolute path as fallback
                                page_path = page_image
                        else:
                            try:
                                page_path = os.path.relpath(page_image).replace('\\', '/')
                            except ValueError:
                                page_path = page_image
                    else:
                        # Already a relative path, use as is
                        page_path = page_image
                else:
                    # Skip if we can't handle it
                    continue
                
                available_pages.append({
                    'page_num': page_num,
                    'page_path': page_path
                })
        
        # Create passages as dictionaries with images and text
        # Randomly select args.topk_docs pages from available_pages
        passages = []
        passage_scores = []
        
        if len(available_pages) > 0:
            # Randomly sample topk_docs pages from available pages
            num_pages_to_select = min(args.topk_docs, len(available_pages))
            selected_pages = random.sample(available_pages, num_pages_to_select)
            
            for avail_page in selected_pages:
                # Ensure passage image path is absolute
                passage_img_path = avail_page['page_path']
                if not os.path.isabs(passage_img_path):
                    # Convert relative path to absolute path
                    if output_dir:
                        # If path is relative to output_dir, convert to absolute
                        passage_img_path = os.path.abspath(os.path.join(output_dir, passage_img_path))
                    else:
                        # If no output_dir, convert relative to absolute from current working directory
                        passage_img_path = os.path.abspath(passage_img_path)
                
                passage_dict = {
                    'images': [passage_img_path],
                    'text': ''  # No text content for pages
                }
                passages.append(passage_dict)
                passage_scores.append(0.0)
        
        # Handle ensure_gt_passage_in_topk: ensure ALL evidence_pages are in passages
        if args.ensure_gt_passage_in_topk and evidence_pages:
            # Normalize evidence_pages to integers
            evidence_page_nums = []
            for ev_page in evidence_pages:
                if isinstance(ev_page, int):
                    evidence_page_nums.append(ev_page)
                elif isinstance(ev_page, str) and ev_page.startswith('page_'):
                    evidence_page_nums.append(int(ev_page.split('_')[1]))
                else:
                    # Try to convert to int
                    try:
                        evidence_page_nums.append(int(ev_page))
                    except:
                        continue
            
            # Check which evidence pages are already in passages
            # We need to match by comparing the page numbers
            existing_evidence_page_nums = set()
            for passage in passages:
                if passage['images'] and len(passage['images']) > 0:
                    passage_page_path = passage['images'][0]
                    
                    # Find which page number this passage corresponds to
                    # passage_page_path is absolute, but avail_page['page_path'] is relative
                    # Convert both to absolute for comparison, or compare by normalizing paths
                    for avail_page in available_pages:
                        # Convert avail_page path to absolute for comparison
                        avail_page_path_abs = avail_page['page_path']
                        if not os.path.isabs(avail_page_path_abs):
                            if output_dir:
                                avail_page_path_abs = os.path.abspath(os.path.join(output_dir, avail_page_path_abs))
                            else:
                                avail_page_path_abs = os.path.abspath(avail_page_path_abs)
                        
                        # Compare normalized absolute paths
                        if os.path.normpath(passage_page_path) == os.path.normpath(avail_page_path_abs):
                            # Check if this page number is in evidence_pages
                            if avail_page['page_num'] in evidence_page_nums:
                                existing_evidence_page_nums.add(avail_page['page_num'])
                            break
            
            # Add missing evidence pages to passages
            for ev_page_num in evidence_page_nums:
                if ev_page_num not in existing_evidence_page_nums:
                    # Find the page path from available_pages
                    page_path = None
                    for avail_page in available_pages:
                        if avail_page['page_num'] == ev_page_num:
                            page_path = avail_page['page_path']
                            break
                    
                    if page_path:
                        # Ensure passage image path is absolute
                        passage_img_path = page_path
                        if not os.path.isabs(passage_img_path):
                            # Convert relative path to absolute path
                            if output_dir:
                                # If path is relative to output_dir, convert to absolute
                                passage_img_path = os.path.abspath(os.path.join(output_dir, passage_img_path))
                            else:
                                # If no output_dir, convert relative to absolute from current working directory
                                passage_img_path = os.path.abspath(passage_img_path)
                        
                        passage_dict = {
                            'images': [passage_img_path],
                            'text': ''
                        }
                        # Insert at the beginning to prioritize evidence pages
                        passages.insert(0, passage_dict)
                        passage_scores.insert(0, 1.0)  # High score for evidence pages
                        existing_evidence_page_nums.add(ev_page_num)
            
            # Trim to topk_docs if needed
            if len(passages) > args.topk_docs:
                passages = passages[:args.topk_docs]
                passage_scores = passage_scores[:args.topk_docs]
        
        # Find gt_passage_idx: which passages contain evidence pages (can be multiple)
        # Find ALL passages that match any evidence page
        gt_passage_idx_list = []
        if evidence_pages:
            # Normalize all evidence pages to integers
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
            
            # Find page paths for all evidence pages
            # Convert to absolute paths for comparison (passages have absolute paths)
            evidence_page_paths_abs = set()
            for ev_page_num in evidence_page_nums:
                for avail_page in available_pages:
                    if avail_page['page_num'] == ev_page_num:
                        # Convert to absolute path for comparison
                        ev_path = avail_page['page_path']
                        if not os.path.isabs(ev_path):
                            if output_dir:
                                ev_path = os.path.abspath(os.path.join(output_dir, ev_path))
                            else:
                                ev_path = os.path.abspath(ev_path)
                        evidence_page_paths_abs.add(os.path.normpath(ev_path))
                        break
            
            # Find which passages contain any of these evidence pages
            for i, passage in enumerate(passages):
                if passage['images'] and len(passage['images']) > 0:
                    passage_page_path = passage['images'][0]  # This is already absolute
                    if os.path.normpath(passage_page_path) in evidence_page_paths_abs:
                        gt_passage_idx_list.append(i)
        
        # If no evidence pages found, use empty list (or could use [-1] for backward compatibility)
        gt_passage_idx = gt_passage_idx_list if gt_passage_idx_list else []
        
        # Get main image (first evidence page, or first available page)
        img_path = None
        if evidence_pages and len(evidence_pages) > 0:
            # Try to use the first evidence page
            ev_page = evidence_pages[0]
            if isinstance(ev_page, int):
                ev_page_num = ev_page
            elif isinstance(ev_page, str) and ev_page.startswith('page_'):
                ev_page_num = int(ev_page.split('_')[1])
            else:
                try:
                    ev_page_num = int(ev_page)
                except:
                    ev_page_num = None
            
            # Find the page path from available_pages
            if ev_page_num is not None:
                for avail_page in available_pages:
                    if avail_page['page_num'] == ev_page_num:
                        img_path = avail_page['page_path']
                        break
        
        # Fallback to first available page if no evidence page found
        if img_path is None and available_pages:
            img_path = available_pages[0]['page_path']
        
        # Ensure img_path is absolute (same as passages images)
        if img_path and not os.path.isabs(img_path):
            # Convert relative path to absolute path
            if output_dir:
                # If path is relative to output_dir, convert to absolute
                img_path = os.path.abspath(os.path.join(output_dir, img_path))
            else:
                # If no output_dir, convert relative to absolute from current working directory
                img_path = os.path.abspath(img_path)
        
        # Form prompt
        question_part = f"\n[QUESTION] {batch['question'][idx]}"
        prompt = VLM_PROMPT_FOR_VQA + question_part
        
        img_paths.append(img_path)
        prompts[idx] = prompt
        all_passage_contents[idx] = passages
        all_passage_scores[idx] = passage_scores
        all_gt_passage_idx[idx] = gt_passage_idx
    
    # Update batch with processed data
    batch['img_path'] = img_paths
    batch['prompt'] = prompts
    batch['gt_passage_idx'] = all_gt_passage_idx
    batch['passages'] = all_passage_contents
    batch['passage_scores'] = all_passage_scores
    batch['gold_answer'] = batch.get('answer', [''] * batch_size)

    if args.add_separate_prompt_for_prior:
        prior_prompts = [VLM_PROMPT_FOR_PRIOR for _ in range(batch_size)]
        batch['prior_prompt'] = prior_prompts

    return batch

def create_augmented_instances(dataset, args):
    """
    Create augmented training instances by subdividing ground-truth slides.
    Augmented instances are separate from original instances and have empty gt_passage_idx.
    
    Args:
        dataset: Processed HuggingFace dataset
        args: Arguments object with topk_docs, img_basedir, split, etc.
    
    Returns:
        Dataset with original + augmented instances
    """
    # Only augment if enabled and topk_docs == 4
    if not getattr(args, 'enable_data_augmentation', False):
        return dataset
    
    if args.topk_docs != 4:
        print(f"Warning: Data augmentation requires topk_docs=4, but got {args.topk_docs}. Skipping augmentation.")
        return dataset
    
    print("Creating augmented instances with image subdivision...")
    img_base_dir = args.img_basedir if args.img_basedir else "../../shared_space/vqa_data/KBVQA_data/SlideVQA"
    img_base_dir_abs = os.path.abspath(img_base_dir)
    split_name = args.split if hasattr(args, 'split') and args.split else 'train'
    output_dir = getattr(args, 'output_dir', None)
    if output_dir:
        output_dir = os.path.abspath(output_dir)
    
    augmented_instances = []
    
    for item in tqdm(dataset, desc="Creating augmented instances"):
        # Get evidence pages (ground-truth slides)
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
        
        # Skip if >2 GT slides
        if len(evidence_page_nums) > 2:
            continue
        
        # Skip if no GT slides
        if len(evidence_page_nums) == 0:
            continue
        
        # Get original batch data to access page images
        qa_id = item['qa_id']
        question = item['question']
        answer = item.get('answer', '')
        prompt = item.get('prompt', '')
        
        # Collect GT slide images
        gt_slide_images = []
        for ev_page_num in evidence_page_nums:
            page_key = f'page_{ev_page_num}'
            page_image = None
            
            # Try to get from item (page fields should still be in dataset)
            if page_key in item and item[page_key] is not None:
                page_image = item[page_key]
                
                # Handle different types
                if isinstance(page_image, Image.Image):
                    if page_image.mode != 'RGB':
                        page_image = page_image.convert('RGB')
                elif isinstance(page_image, str):
                    # It's a path - try to load
                    if os.path.exists(page_image):
                        try:
                            page_image = Image.open(page_image).convert('RGB')
                        except Exception as e:
                            print(f"Warning: Could not load image from path {page_image}: {e}")
                            continue
                    else:
                        # Try standard naming convention
                        img_filename = f"{split_name}_{qa_id}_page_{ev_page_num}.png"
                        img_path = os.path.join(img_base_dir_abs, img_filename)
                        if os.path.exists(img_path):
                            try:
                                page_image = Image.open(img_path).convert('RGB')
                            except Exception as e:
                                print(f"Warning: Could not load image {img_path}: {e}")
                                continue
                        else:
                            continue
                elif isinstance(page_image, bytes):
                    # Handle bytes
                    try:
                        from io import BytesIO
                        page_image = Image.open(BytesIO(page_image)).convert('RGB')
                    except Exception as e:
                        print(f"Warning: Could not load image from bytes: {e}")
                        continue
                else:
                    continue
            else:
                # Try to load from standard path
                img_filename = f"{split_name}_{qa_id}_page_{ev_page_num}.png"
                img_path = os.path.join(img_base_dir_abs, img_filename)
                if os.path.exists(img_path):
                    try:
                        page_image = Image.open(img_path).convert('RGB')
                    except Exception as e:
                        print(f"Warning: Could not load image {img_path}: {e}")
                        continue
            
            if page_image is not None:
                gt_slide_images.append((ev_page_num, page_image))
        
        # Skip if we couldn't load GT slide images
        if len(gt_slide_images) == 0:
            continue
        
        # Create augmented passages
        augmented_passages = []
        part_counter = 0
        
        if len(evidence_page_nums) == 2:
            # Case 1: 2 GT slides - split each by width (1 row, 2 cols)
            for ev_page_num, gt_image in gt_slide_images:
                sub_images = subdivide_image_for_augmentation(gt_image, rows=1, cols=2)
                
                for part_idx, sub_img in enumerate(sub_images):
                    # Save subdivided image
                    saved_path = save_subdivided_image(
                        sub_img, qa_id, ev_page_num, part_idx, split_name, img_base_dir_abs
                    )
                    
                    if saved_path:
                        # Convert to relative path from output_dir if needed
                        passage_img_path = saved_path
                        if output_dir and os.path.isabs(saved_path):
                            try:
                                passage_img_path = os.path.relpath(saved_path, output_dir).replace('\\', '/')
                            except ValueError:
                                # Different drives, keep absolute or use img_basedir
                                if args.img_basedir:
                                    filename = os.path.basename(saved_path)
                                    passage_img_path = f"{args.img_basedir}/{filename}"
                                else:
                                    passage_img_path = saved_path
                        
                        # Ensure absolute path for passage
                        if not os.path.isabs(passage_img_path):
                            if output_dir:
                                passage_img_path = os.path.abspath(os.path.join(output_dir, passage_img_path))
                            else:
                                passage_img_path = os.path.abspath(passage_img_path)
                        
                        passage_dict = {
                            'images': [passage_img_path],
                            'text': ''
                        }
                        augmented_passages.append(passage_dict)
                        part_counter += 1
        
        elif len(evidence_page_nums) == 1:
            # Case 2: 1 GT slide - split into 2x2 grid
            ev_page_num, gt_image = gt_slide_images[0]
            sub_images = subdivide_image_for_augmentation(gt_image, rows=2, cols=2)
            
            for part_idx, sub_img in enumerate(sub_images):
                # Save subdivided image
                saved_path = save_subdivided_image(
                    sub_img, qa_id, ev_page_num, part_idx, split_name, img_base_dir_abs
                )
                
                if saved_path:
                    # Convert to relative path from output_dir if needed
                    passage_img_path = saved_path
                    if output_dir and os.path.isabs(saved_path):
                        try:
                            passage_img_path = os.path.relpath(saved_path, output_dir).replace('\\', '/')
                        except ValueError:
                            # Different drives, keep absolute or use img_basedir
                            if args.img_basedir:
                                filename = os.path.basename(saved_path)
                                passage_img_path = f"{args.img_basedir}/{filename}"
                            else:
                                passage_img_path = saved_path
                    
                    # Ensure absolute path for passage
                    if not os.path.isabs(passage_img_path):
                        if output_dir:
                            passage_img_path = os.path.abspath(os.path.join(output_dir, passage_img_path))
                        else:
                            passage_img_path = os.path.abspath(passage_img_path)
                    
                    passage_dict = {
                        'images': [passage_img_path],
                        'text': ''
                    }
                    augmented_passages.append(passage_dict)
                    part_counter += 1
        
        # Create augmented instance if we have passages
        if len(augmented_passages) > 0:
            # Get main image (first passage image)
            img_path = None
            if augmented_passages and augmented_passages[0]['images']:
                img_path = augmented_passages[0]['images'][0]
            
            # Ensure img_path is absolute
            if img_path and not os.path.isabs(img_path):
                if output_dir:
                    img_path = os.path.abspath(os.path.join(output_dir, img_path))
                else:
                    img_path = os.path.abspath(img_path)
            
            # Create passage scores (all 0.0 for augmented)
            passage_scores = [0.0] * len(augmented_passages)
            
            # Create augmented instance
            augmented_instance = {
                'qa_id': qa_id,  # Keep same qa_id (or could add suffix)
                'question': question,
                'answer': answer,
                'prompt': prompt,
                'img_path': img_path,
                'passages': augmented_passages,
                'passage_scores': passage_scores,
                'gt_passage_idx': [-1],  # Use -1 as placeholder - no GT annotation
                'gold_answer': answer,
                'evidence_pages': evidence_pages  # Keep original for reference
            }
            
            # Add prior prompt if needed
            if args.add_separate_prompt_for_prior:
                prior_prompt_content = item.get('prior_prompt', VLM_PROMPT_FOR_PRIOR)
                if question:
                    prior_content = prior_prompt_content.replace('<<<QUESTION>>>', question)
                    augmented_instance['prior_prompt'] = prior_content
            
            augmented_instances.append(augmented_instance)
    
    print(f"Created {len(augmented_instances)} augmented instances")
    
    # Concatenate original dataset with augmented instances
    if len(augmented_instances) > 0:
        from datasets import Dataset
        augmented_dataset = Dataset.from_list(augmented_instances)
        
        # Concatenate datasets
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets([dataset, augmented_dataset])
        print(f"Combined dataset: {len(dataset)} original + {len(augmented_instances)} augmented = {len(combined_dataset)} total")
        return combined_dataset
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_path", type=str, required=True, help='Path to SlideVQA dataset (HuggingFace dataset name or path)')
    parser.add_argument("--mode", type=str, default='beft', choices=['beft', 'sft'])
    parser.add_argument("--img_basedir", type=str, default="../../shared_space/vqa_data/KBVQA_data/SlideVQA", 
                        help='Base directory for image paths (should match download_slidevqa_images.py IMG_BASE_DIR)')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--report_token_length", action='store_true')
    parser.add_argument("--drop_max_tokens", type=int, default=0)
    parser.add_argument("--topk_docs", type=int, default=5, help='Number of top passages to use')
    parser.add_argument("--random_sample_1passage_from_topk", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8, help='Number of workers for parallel processing')
    parser.add_argument("--batch_size", type=int, default=100, help='Batch size for dataset processing')
    parser.add_argument("--ensure_gt_passage_in_topk", action='store_true', help='Ensure all evidence_pages are in passages')
    parser.add_argument("--add_separate_prompt_for_prior", action='store_true')
    parser.add_argument("--prior_prompt", type=str, default=None)
    parser.add_argument("--split", type=str, default='train', help='Dataset split to use')
    parser.add_argument("--skip_image_path_exist_check", action='store_true', 
                        help='Skip disk checks for image file existence. Assumes all images are already saved correctly. Significantly speeds up processing.')
    parser.add_argument("--enable_data_augmentation", action='store_true',
                        help='Enable data augmentation by subdividing GT slides into sub-images. Requires topk_docs=4. Creates separate training instances with empty gt_passage_idx.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    # Try to load from disk first, then try loading from HuggingFace Hub
    try:
        from datasets import load_from_disk
        ds = load_from_disk(args.hf_dataset_path)
    except:
        ds = load_dataset(args.hf_dataset_path, split=args.split)
    
    if args.sample_size > 0 or args.sample_offset > 0:
        print(f"Sampling {args.sample_size} items from {args.sample_offset} to {args.sample_offset + args.sample_size}")
        ds = ds.shuffle(seed=args.seed).select(range(args.sample_offset, min(args.sample_offset + args.sample_size, len(ds))))
    print(f"Dataset loaded with {len(ds)} items")

    # Only keep the necessary fields
    keep_fields = ['qa_id', 'question', 'answer', 'evidence_pages']
    # Add page fields
    for i in range(1, 21):
        keep_fields.append(f'page_{i}')
    
    ds = ds.remove_columns([col for col in ds.column_names if col not in keep_fields])
    
    # Use HuggingFace's native batched processing with multiprocessing
    print("Processing dataset with batched operations...")
    
    # Create a partial function with the required arguments
    process_func = partial(add_prefix_and_form_prompt, args=args)
    
    # Process the dataset using HuggingFace's map with multiprocessing
    ds = ds.map(
        process_func,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        desc="Processing batches",
        load_from_cache_file=False  # Disable caching for fresh processing
    )
    
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
            # For beft mode, count tokens in the full conversation
            # Get the longest passage for token counting
            max_len_passage = None
            max_len = 0
            for passage in example['passages']:
                passage_text = passage.get('text', '')
                if len(passage_text) > max_len:
                    max_len = len(passage_text)
                    max_len_passage = passage_text
            
            if max_len_passage:
                user_content = example['prompt'].replace('<<<EVIDENCE>>>', max_len_passage)
            else:
                user_content = example['prompt'].replace('<<<EVIDENCE>>>', '')
            
            assistant_content = f"[ANSWER] {example['gold_answer']}"
            full_text = user_content + assistant_content
            
            # Count tokens without padding
            tokens = tokenizer.encode(full_text, add_special_tokens=True)
            token_count = len(tokens) + IMG_TOKEN_COUNTS  # add space for image tokens
            
            return {"token_count": token_count}
        
        # Count tokens for all examples
        ds_with_tokens = ds.map(
            count_tokens_for_example,
            desc="Counting tokens",
            num_proc=8
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
        
        def filter_by_token_count(batch):
            return [tc <= args.drop_max_tokens for tc in batch['token_count']]

        ds_filtered = ds_with_tokens.filter(
            filter_by_token_count,
            batched=True,
            num_proc=8,
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

    # Create augmented instances (separate training examples) if enabled
    if args.enable_data_augmentation:
        ds = create_augmented_instances(ds, args)

    print("Converting to ShareGPT format...")
    convert_to_sharegpt_format(ds, args.output_dir, mode=args.mode, args=args)
    print(f"Processing complete! Output saved to {args.output_dir}")
