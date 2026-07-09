"""
Script for downloading SlideVQA dataset images from HuggingFace and saving to local directory.

This script:
1. Loads SlideVQA dataset from HuggingFace
2. Downloads all page images (page_1 to page_20) to local directory
3. Updates the dataset with local image paths
4. Saves the updated dataset
"""

import sys
sys.path.append('./src')
from datasets import load_dataset, Dataset, DatasetDict
import argparse
import os
from tqdm import tqdm
from PIL import Image, ImageFile
import hashlib

# Allow loading truncated images for verification
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_image_hash(image):
    """Get a hash of the image for filename"""
    if hasattr(image, 'tobytes'):
        return hashlib.md5(image.tobytes()).hexdigest()[:16]
    return hashlib.md5(str(image).encode()).hexdigest()[:16]

def is_image_corrupted(image_path):
    """
    Check if an image file is corrupted or truncated.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        (is_corrupted, error_msg): (bool, str) tuple
        is_corrupted: True if image is corrupted, False if valid
        error_msg: Error message if corrupted, None if valid
    """
    if not os.path.exists(image_path):
        return True, "File does not exist"
    
    if os.path.getsize(image_path) == 0:
        return True, "File is empty (0 bytes)"
    
    try:
        # Try to open and verify the image
        with Image.open(image_path) as img:
            # Verify image integrity (this will raise an exception if corrupted)
            img.verify()
        
        # Re-open to do a full load test (verify() closes the image)
        with Image.open(image_path) as img:
            # Try to convert to RGB to ensure it can be processed
            img.convert('RGB')
        
        return False, None
    except (OSError, IOError) as e:
        # Image is corrupted or truncated
        return True, f"Image corrupted/truncated: {str(e)}"
    except Exception as e:
        return True, f"Unexpected error: {str(e)}"

def save_image(image, save_path, skip_if_exists=True, verbose=False, check_corruption=True):
    """
    Save an image to the specified path with corruption checking.
    
    Args:
        image: PIL Image object to save
        save_path: Path where to save the image
        skip_if_exists: If True, skip saving if file exists and is valid (non-zero size and not corrupted)
        verbose: If True, print status messages for each image
        check_corruption: If True, check if existing images are corrupted and re-download if needed
    
    Returns:
        save_path: The path where the image was saved
        status: One of 'skipped' (already exists and valid), 'redownloaded' (was corrupted/0 bytes), 'downloaded' (first time), 'error' (failed)
    """
    # Check if image already exists and is valid
    file_existed = os.path.exists(save_path)
    was_corrupted = False
    was_zero_bytes = False
    
    if file_existed:
        file_size = os.path.getsize(save_path)
        if file_size > 0:
            # File exists and has non-zero size, check if it's corrupted
            if check_corruption:
                is_corrupted, error_msg = is_image_corrupted(save_path)
                if is_corrupted:
                    was_corrupted = True
                    print(f"  [REDOWNLOAD] {os.path.basename(save_path)} - File exists but is corrupted: {error_msg}")
                    # Remove the corrupted file
                    try:
                        os.remove(save_path)
                    except Exception as e:
                        if verbose:
                            print(f"    Warning: Could not remove corrupted file: {e}")
                else:
                    # File exists, is valid size, and is not corrupted - skip (no print)
                    return save_path, 'skipped'
            else:
                # Skip corruption check, just check file size (no print)
                return save_path, 'skipped'
        else:
            # File exists but is 0 bytes, need to re-download
            was_zero_bytes = True
            print(f"  [REDOWNLOAD] {os.path.basename(save_path)} - File exists but is 0 bytes, re-downloading...")
            # Remove the 0-byte file
            try:
                os.remove(save_path)
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not remove 0-byte file: {e}")
    else:
        # File doesn't exist, this will be a first-time download
        print(f"  [DOWNLOAD] {os.path.basename(save_path)} - File does not exist, downloading...")
    
    # File doesn't exist, was corrupted, or was 0 bytes - save it
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if isinstance(image, Image.Image):
                image.save(save_path)
            else:
                # If it's a path or URL, we might need to handle it differently
                # For now, assume it's already an Image object
                image.save(save_path)
            
            # Verify the saved file exists and has non-zero size
            if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
                retry_count += 1
                if verbose:
                    print(f"  [RETRY {retry_count}/{max_retries}] {os.path.basename(save_path)} - Saved but file is 0 bytes or missing, retrying...")
                if retry_count < max_retries:
                    continue
                else:
                    if verbose:
                        print(f"  [ERROR] {os.path.basename(save_path)} - Failed after {max_retries} retries: file is still 0 bytes or missing")
                    return save_path, 'error'
            
            # Verify image integrity after saving
            if check_corruption:
                is_corrupted, error_msg = is_image_corrupted(save_path)
                if is_corrupted:
                    retry_count += 1
                    if verbose:
                        print(f"  [RETRY {retry_count}/{max_retries}] {os.path.basename(save_path)} - Saved but image is corrupted: {error_msg}, retrying...")
                    # Remove corrupted file and retry
                    try:
                        os.remove(save_path)
                    except:
                        pass
                    if retry_count < max_retries:
                        continue
                    else:
                        if verbose:
                            print(f"  [ERROR] {os.path.basename(save_path)} - Failed after {max_retries} retries: image is still corrupted")
                        return save_path, 'error'
            
            # Successfully saved and verified
            file_size = os.path.getsize(save_path)
            # Determine status
            if was_corrupted:
                status = 'redownloaded'
                status_label = "[REDOWNLOADED]"
            elif was_zero_bytes:
                status = 'redownloaded'
                status_label = "[REDOWNLOADED]"
            else:
                status = 'downloaded'
                status_label = "[DOWNLOADED]"
            
            # Always print when downloading or redownloading
            print(f"  {status_label} {os.path.basename(save_path)} - Saved and verified successfully (size: {file_size} bytes)")
            return save_path, status
            
        except Exception as e:
            retry_count += 1
            if verbose:
                print(f"  [RETRY {retry_count}/{max_retries}] {os.path.basename(save_path)} - Failed to save: {e}, retrying...")
            if retry_count < max_retries:
                # Try to remove partial file before retry
                try:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                except:
                    pass
                continue
            else:
                if verbose:
                    print(f"  [ERROR] {os.path.basename(save_path)} - Failed after {max_retries} retries: {e}")
                return save_path, 'error'
    
    # Should not reach here, but just in case
    return save_path, 'error'

def process_batch(batch, img_base_dir, split_name, relative_img_dir=None, verbose=False, check_corruption=True):
    """Process a batch of examples: download images and update paths using batched operations"""
    batch_size = len(batch['qa_id'])
    updated_batch = {key: list(values) for key, values in batch.items()}
    
    # Process each page (page_1 to page_20)
    for page_num in range(1, 21):
        page_key = f'page_{page_num}'
        if page_key not in batch:
            continue
        
        page_images = batch[page_key]
        
        # Process each example in the batch
        for idx in range(batch_size):
            page_image = page_images[idx]
            qa_id = batch['qa_id'][idx]
            
            # Skip if image is None or empty
            if page_image is None:
                continue
            
            # Determine image format and save
            try:
                # If it's a PIL Image object
                if isinstance(page_image, Image.Image):
                    # Create a unique filename based on qa_id and page number
                    img_filename = f"{split_name}_{qa_id}_page_{page_num}.png"
                    img_path = os.path.join(img_base_dir, img_filename)
                    
                    # Save the image (with corruption check and verbose output)
                    save_path, status = save_image(page_image, img_path, skip_if_exists=True, verbose=verbose, check_corruption=check_corruption)
                    
                    # Update the field with relative path (for portability)
                    if relative_img_dir:
                        # Use relative path from a base directory
                        updated_batch[page_key][idx] = os.path.join(relative_img_dir, img_filename)
                    else:
                        # Use absolute path
                        updated_batch[page_key][idx] = img_path
                elif isinstance(page_image, str):
                    # If it's already a path, check if it's a URL or local path
                    if page_image.startswith('http://') or page_image.startswith('https://'):
                        # TODO: Download from URL if needed
                        # For now, keep as is
                        updated_batch[page_key][idx] = page_image
                    else:
                        # It's already a local path, check if it's valid and not corrupted
                        img_filename = os.path.basename(page_image)
                        if os.path.exists(page_image):
                            file_size = os.path.getsize(page_image)
                            if file_size > 0:
                                # Check if image is corrupted (if corruption checking is enabled)
                                if check_corruption:
                                    is_corrupted, error_msg = is_image_corrupted(page_image)
                                    if is_corrupted:
                                        # Set to None so it will be re-downloaded if the image is available
                                        # No print here - will be printed when actually downloading
                                        updated_batch[page_key][idx] = None
                                    else:
                                        # It's already a local path and valid, keep as is (no print)
                                        updated_batch[page_key][idx] = page_image
                                else:
                                    # It's already a local path, keep as is (no print)
                                    updated_batch[page_key][idx] = page_image
                            else:
                                # Set to None so it will be re-downloaded if the image is available
                                # No print here - will be printed when actually downloading
                                updated_batch[page_key][idx] = None
                        else:
                            # Set to None so it will be re-downloaded if the image is available
                            # No print here - will be printed when actually downloading
                            updated_batch[page_key][idx] = None
                else:
                    # Unknown type - convert to None to avoid serialization issues
                    # This handles any remaining Image objects that weren't caught
                    if verbose:
                        print(f"  [WARNING] Unknown image type {type(page_image)} for {page_key} in example {qa_id}, setting to None")
                    updated_batch[page_key][idx] = None
            except Exception as e:
                # Keep original value on error, but convert Image to None to avoid serialization issues
                if verbose:
                    print(f"  [ERROR] Processing {page_key} for example {qa_id}: {e}")
                if isinstance(page_image, Image.Image):
                    updated_batch[page_key][idx] = None
                continue
    
    return updated_batch

def convert_image_to_string(example):
    """Convert Image objects to string paths in an example"""
    for key in list(example.keys()):
        if key.startswith('page_'):
            value = example[key]
            # If it's an Image object (dict with 'path' or 'bytes'), extract the path
            if isinstance(value, dict):
                # Image type is a dict with 'path' and 'bytes' fields
                if 'path' in value:
                    example[key] = value['path']
                elif 'bytes' in value:
                    # If only bytes, set to None (shouldn't happen after process_batch)
                    example[key] = None
            # If it's already a string or None, keep it as is
            elif isinstance(value, str) or value is None:
                pass
            # If it's a PIL Image object, it should have been converted in process_batch
            # But just in case, set to None
            elif hasattr(value, 'save'):
                example[key] = None
    return example

def ensure_string_features(dataset):
    """Ensure all page fields are strings, not Images, to avoid JSON serialization errors"""
    from datasets import Features, Value
    
    # First, convert any remaining Image objects to strings
    print("Converting Image objects to string paths...")
    if isinstance(dataset, DatasetDict):
        updated_splits = {}
        for split_name, split_data in dataset.items():
            # Convert Image objects to strings first
            split_data = split_data.map(
                convert_image_to_string,
                desc=f"Converting images to strings for {split_name}",
                load_from_cache_file=False
            )
            # Then change feature types
            features = split_data.features
            new_features = {}
            for key, feature in features.items():
                if key.startswith('page_'):
                    new_features[key] = Value('string')
                else:
                    new_features[key] = feature
            updated_splits[split_name] = split_data.cast(Features(new_features))
        return DatasetDict(updated_splits)
    elif isinstance(dataset, Dataset):
        # Convert Image objects to strings first
        dataset = dataset.map(
            convert_image_to_string,
            desc="Converting images to strings",
            load_from_cache_file=False
        )
        # Then change feature types
        features = dataset.features
        new_features = {}
        for key, feature in features.items():
            if key.startswith('page_'):
                new_features[key] = Value('string')
            else:
                new_features[key] = feature
        return dataset.cast(Features(new_features))
    return dataset

def download_and_update_dataset(hf_dataset_path, output_dir, img_base_dir, split=None, batch_size=100, num_proc=8, relative_img_dir=None, hf_token=None, verbose=False, check_corruption=True):
    """
    Download images from HuggingFace dataset and update paths
    
    Args:
        hf_dataset_path: Path to HuggingFace dataset (name or local path)
        output_dir: Directory to save the updated dataset
        img_base_dir: Base directory to save images
        split: Specific split to process (None for all splits)
        batch_size: Batch size for processing (default: 100)
        num_proc: Number of processes for parallel processing (default: 8)
        relative_img_dir: Relative directory path to use in dataset (for portability)
        hf_token: HuggingFace token for authentication (optional, can also use HF_TOKEN env var)
    """
    # Login to HuggingFace if token provided
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("Logged in to HuggingFace using provided token")
        except Exception as e:
            print(f"Warning: Could not login with token: {e}")
    else:
        # Try to use token from environment variable
        hf_token_env = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if hf_token_env:
            try:
                from huggingface_hub import login
                login(token=hf_token_env)
                print("Logged in to HuggingFace using token from environment variable")
            except Exception as e:
                print(f"Warning: Could not login with token from env: {e}")
    
    print(f"Loading dataset from {hf_dataset_path}...")
    
    # Load dataset
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(hf_dataset_path)
    except:
        # Use token if available for loading gated datasets
        load_kwargs = {}
        if hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'):
            load_kwargs['token'] = hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        
        if split:
            dataset = load_dataset(hf_dataset_path, split=split, **load_kwargs)
        else:
            dataset = load_dataset(hf_dataset_path, **load_kwargs)
    
    # Create image base directory
    os.makedirs(img_base_dir, exist_ok=True)
    
    # If relative_img_dir not provided, use a default relative path
    if relative_img_dir is None:
        # Use a path relative to the output directory
        # Calculate relative path from output_dir to img_base_dir
        try:
            relative_img_dir = os.path.relpath(img_base_dir, output_dir)
        except:
            # If they're on different drives, use absolute path
            relative_img_dir = None
    
    # Create a partial function with fixed arguments for batched processing
    from functools import partial
    process_batch_func = partial(
        process_batch,
        img_base_dir=img_base_dir,
        relative_img_dir=relative_img_dir,
        verbose=verbose,
        check_corruption=check_corruption
    )
    
    # Process dataset using batched operations
    if isinstance(dataset, DatasetDict):
        # Multiple splits
        print(f"Processing dataset with batched operations (batch_size={batch_size}, num_proc={num_proc})...")
        updated_datasets = {}
        
        for split_name, split_dataset in dataset.items():
            print(f"Processing split: {split_name} ({len(split_dataset)} examples)...")
            
            # Use batched map operation
            def process_batch_with_split(batch):
                return process_batch(batch, img_base_dir, split_name, relative_img_dir, verbose=verbose, check_corruption=check_corruption)
            
            updated_dataset = split_dataset.map(
                process_batch_with_split,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                desc=f"Processing {split_name}",
                load_from_cache_file=False
            )
            
            updated_datasets[split_name] = updated_dataset
        
        # Save updated dataset
        updated_dataset = DatasetDict(updated_datasets)
        
        # Ensure all Image features are converted to strings before saving
        print("Converting Image features to string type...")
        updated_dataset = ensure_string_features(updated_dataset)
        
        print(f"Saving updated dataset to {output_dir}...")
        updated_dataset.save_to_disk(output_dir)
        
    elif isinstance(dataset, Dataset):
        # Single split
        split_name = split if split else 'train'
        print(f"Processing dataset ({len(dataset)} examples) with batched operations (batch_size={batch_size}, num_proc={num_proc})...")
        
        # Use batched map operation
        def process_batch_with_split(batch):
            return process_batch(batch, img_base_dir, split_name, relative_img_dir, verbose=verbose, check_corruption=check_corruption)
        
        updated_dataset = dataset.map(
            process_batch_with_split,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Processing",
            load_from_cache_file=False
        )
        
        # Ensure all Image features are converted to strings before saving
        print("Converting Image features to string type...")
        updated_dataset = ensure_string_features(updated_dataset)
        
        # Save updated dataset
        print(f"Saving updated dataset to {output_dir}...")
        updated_dataset.save_to_disk(output_dir)
    else:
        raise ValueError(f"Unknown dataset type: {type(dataset)}")
    
    print(f"Done! Updated dataset saved to {output_dir}")
    print(f"Images saved to {img_base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SlideVQA images from HuggingFace and update dataset")
    parser.add_argument("--hf_dataset_path", type=str, default="NTT-hil-insight/SlideVQA",
                        help="HuggingFace dataset path or name")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the updated dataset")
    parser.add_argument("--img_base_dir", type=str, 
                        default="../../shared_space/vqa_data/KBVQA_data/SlideVQA",
                        help="Base directory to save images (root directory for SlideVQA images)")
    parser.add_argument("--split", type=str, default=None,
                        help="Specific split to process (train, validation, test). None for all splits")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for batched processing (default: 100)")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Number of processes for parallel processing (default: 8)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for authentication (or set HF_TOKEN env var)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information for each downloaded image")
    parser.add_argument("--check_corruption", action='store_true', default=True,
                        help="Check if existing images are corrupted and re-download if needed (default: True)")
    parser.add_argument("--no_check_corruption", dest='check_corruption', action='store_false',
                        help="Disable corruption checking (faster but may keep corrupted images)")
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    img_base_dir = os.path.abspath(args.img_base_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Use absolute path for images in dataset (as specified by user)
    # Images will be saved to img_base_dir, and paths in dataset will point there
    download_and_update_dataset(
        hf_dataset_path=args.hf_dataset_path,
        output_dir=output_dir,
        img_base_dir=img_base_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        relative_img_dir=None,  # Use absolute paths in dataset
        hf_token=args.hf_token,
        verbose=args.verbose,
        check_corruption=args.check_corruption
    )

