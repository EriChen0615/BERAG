import sys
sys.path.append('./src')
import json
from pprint import pprint
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from PIL import Image
import argparse

import torch
# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

from transformers import AutoTokenizer
from tqdm import tqdm
import gc
import os
import numpy as np
import pandas as pd

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global prompt template for SlideVQA (matching ragk_slidevqa.py)
VLM_PROMPT_FOR_VQA = (
    "Answer the question directly without explanations based on the provided slides."
    "<<<EVIDENCE>>>"
)


class VLLMSlideVQAInferenceEngine:
    """vLLM-based inference engine for SlideVQA with support for multiple images (20 slides)."""
    
    def __init__(self, 
                 model_path: str = None,
                 base_model_path: str = None,
                 processor_path: str = None,
                 adapter_name_or_path: str = None,
                 tensor_parallel_size: int = None,
                 pipeline_parallel_size: int = 1,
                 max_model_len: int = 8192,
                 max_pixels: int = None,
                 dtype: str = "bfloat16",
                 trust_remote_code: bool = True,
                 **vllm_kwargs):
        """
        Initialize the vLLM inference engine for SlideVQA.
        
        Args:
            model_path: Path to the model or model name
            base_model_path: Path to base model (for LoRA adapters)
            processor_path: Path to processor (for tokenizer)
            adapter_name_or_path: Path to LoRA adapter
            tensor_parallel_size: Number of GPUs for tensor parallelism
            pipeline_parallel_size: Number of GPUs for pipeline parallelism
            max_model_len: Maximum model context length
            max_pixels: Maximum pixels for image processing
            dtype: Model data type
            trust_remote_code: Whether to trust remote code
            **vllm_kwargs: Additional vLLM engine arguments
        """
        
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.processor_path = processor_path or model_path
        self.adapter_name_or_path = adapter_name_or_path
        self.max_pixels = max_pixels
        self.max_model_len = max_model_len
        
        # Determine GPU configuration
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        logger.info(f"Initializing vLLM engine for SlideVQA...")
        logger.info(f"Model: {model_path}. NOTE: DEPRECATED!. Use base_model_path instead.")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Adapter: {adapter_name_or_path}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"Pipeline parallel size: {pipeline_parallel_size}")
        logger.info(f"Max model length: {max_model_len}")
        
        # Determine which model to load
        model_name_or_path = base_model_path
        if adapter_name_or_path and adapter_name_or_path != "":
            # Use base model for LoRA
            enable_lora = True
        else:
            # Use the main model path
            enable_lora = False
        print(model_name_or_path)
        # Initialize vLLM engine
        engine_args = {
            "model": model_name_or_path,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "disable_log_stats": True,
            "enable_lora": enable_lora,
            "limit_mm_per_prompt": {"image": 20, "video": 1, "audio": 1},  # Support up to 20 images for SlideVQA
            "max_lora_rank": 128,
            **vllm_kwargs
        }
        
        self.llm = LLM(**engine_args)
        
        # Initialize tokenizer for chat template processing
        logger.info("Loading tokenizer for chat template processing...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.processor_path, 
            trust_remote_code=trust_remote_code
        )
        # Set truncation side to 'left' to keep the rightmost content (保留最右内容)
        self.tokenizer.truncation_side = 'left'
        
        # Initialize sampling parameters for greedy decoding
        self.sampling_params = SamplingParams(
            repetition_penalty=1.0,
            temperature=0.0,  # Greedy decoding
            top_p=1.0,
            top_k=-1,
            max_tokens=1024,
            skip_special_tokens=True,
        )
        
        # Initialize LoRA request if adapter is provided
        self.lora_request = None
        if enable_lora:
            self.lora_request = LoRARequest("adapter_0", 1, adapter_name_or_path)
            print(f"Loaded LoRA from {adapter_name_or_path}")
        
        logger.info("vLLM engine initialized successfully")
    
    def _estimate_image_tokens(self, image: Image.Image) -> int:
        """
        Estimate the number of tokens for an image based on its size.
        For Qwen2-VL, images are processed with patch size 14x14.
        The actual token count depends on the processed image size after resizing.
        
        Args:
            image: PIL Image object
            
        Returns:
            Estimated number of tokens for this image
        """
        if image is None:
            return 0
        
        width, height = image.size
        
        # Qwen2-VL uses patch size 14x14
        # The image will be resized to fit within max_pixels if specified
        # For estimation, we use a typical processing size
        # Qwen2-VL typically processes images to around 1344x1344 or similar
        # But it depends on max_pixels and aspect ratio
        
        # Conservative estimation: assume images are processed to a reasonable size
        # For slides, typical size might be around 1920x1080 or similar
        # After processing, tokens = (processed_width / 14) * (processed_height / 14)
        
        # If max_pixels is set, estimate based on that
        if self.max_pixels:
            # Estimate processed size based on max_pixels while maintaining aspect ratio
            # Formula: width * height <= max_pixels, maintaining aspect_ratio = width/height
            # So: width = height * aspect_ratio, and height * (height * aspect_ratio) <= max_pixels
            # Therefore: height^2 * aspect_ratio <= max_pixels
            aspect_ratio = width / height if height > 0 else 1.0
            estimated_height = int((self.max_pixels / aspect_ratio) ** 0.5)
            estimated_width = int(estimated_height * aspect_ratio)
            # Ensure we don't exceed original dimensions
            estimated_width = min(estimated_width, width)
            estimated_height = min(estimated_height, height)
        else:
            # Default: Qwen2-VL typically processes images with a max dimension of 1344
            # But if image is smaller, use original size
            max_dim = max(width, height)
            if max_dim <= 1344:
                estimated_width, estimated_height = width, height
            else:
                # Scale proportionally to fit max_dim = 1344
                scale = 1344 / max_dim
                estimated_width = int(width * scale)
                estimated_height = int(height * scale)
        
        # Calculate tokens: (width / 14) * (height / 14), rounded up
        # Qwen2-VL uses patch size 14x14
        patch_size = 14
        tokens_per_image = ((estimated_width + patch_size - 1) // patch_size) * ((estimated_height + patch_size - 1) // patch_size)
        
        # Add some overhead for special tokens and processing (typically +1 per image)
        tokens_per_image += 1
        
        return tokens_per_image
    
    def _prepare_conversation_format(self, prompt_text: str, images: list) -> tuple[str, list]:
        """
        Prepare conversation format for vLLM with Qwen2-VL using chat template.
        Supports multiple images (up to 20 for SlideVQA).
        Truncates images (from the front, keeping the back) if needed, but never truncates text.
        
        Args:
            prompt_text: The text prompt
            images: List of PIL Image objects (up to 20)
            
        Returns:
            Tuple of (formatted_prompt_string, filtered_images_list)
        """
        # Remove the <image> token from the prompt template since we'll handle it through messages
        text_prompt = prompt_text.replace("<image>", "").strip()
        
        # Filter out None images
        valid_images = [img for img in images if img is not None] if images else []
        
        # Calculate text tokens (without truncation)
        # First, create a text-only message to estimate text tokens
        text_only_messages = [
            {
                "role": "user",
                "content": text_prompt
            }
        ]
        text_only_prompt = self.tokenizer.apply_chat_template(
            text_only_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        encoded_text = self.tokenizer(
            text_only_prompt,
            return_tensors=None,
            add_special_tokens=False,
            truncation=False
        )
        text_tokens = len(encoded_text['input_ids'])
        
        # Reserve space for text, generation, and safety margin
        # Generation: 1024 tokens, safety margin: 2048 tokens for chat template overhead
        reserved_for_text_and_generation = text_tokens + 1024 + 2048
        available_tokens_for_images = max(0, self.max_model_len - reserved_for_text_and_generation)
        
        # Calculate tokens for each image and filter from front if needed
        image_token_counts = []
        total_image_tokens = 0
        for image in valid_images:
            img_tokens = self._estimate_image_tokens(image)
            image_token_counts.append((image, img_tokens))
            total_image_tokens += img_tokens
        
        # If total image tokens exceed available space, remove images from the front (保留后面的slides)
        filtered_images = []
        accumulated_tokens = 0
        for image, img_tokens in reversed(image_token_counts):  # Process from back to front
            if accumulated_tokens + img_tokens <= available_tokens_for_images:
                filtered_images.insert(0, image)  # Insert at beginning to maintain order
                accumulated_tokens += img_tokens
            else:
                logger.warning(
                    f"Dropping image from front: estimated {img_tokens} tokens "
                    f"(total image tokens: {total_image_tokens}, available: {available_tokens_for_images})"
                )
        
        # Log truncation info if images were dropped
        if len(filtered_images) < len(valid_images):
            dropped_count = len(valid_images) - len(filtered_images)
            logger.warning(
                f"Dropped {dropped_count} image(s) from the front to fit within token limit. "
                f"Kept {len(filtered_images)} image(s) from the back. "
                f"(Text tokens: {text_tokens}, Image tokens: {accumulated_tokens}/{total_image_tokens}, "
                f"Max model len: {self.max_model_len})"
            )
        
        # Create conversation messages with filtered images
        if filtered_images:
            content = []
            # Add all filtered images first
            for image in filtered_images:
                content.append({"type": "image", "image": image})
            # Add text at the end
            content.append({"type": "text", "text": text_prompt})
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        else:
            # Text-only conversation
            messages = [
                {
                    "role": "user",
                    "content": text_prompt
                }
            ]
        
        # Apply chat template - this is crucial for vLLM multimodal support
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return formatted_prompt, filtered_images
    
    def _prepare_multimodal_data(self, images: list) -> dict:
        """
        Prepare multimodal data for vLLM input with multiple images.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            Dictionary with image data for vLLM
        """
        if not images or len(images) == 0:
            return None
        
        # Filter out None images
        valid_images = [img for img in images if img is not None]
        
        if len(valid_images) == 0:
            return None
        
        # For vLLM with Qwen2-VL, when there are multiple images,
        # vLLM expects a list of images in the "image" key
        # The chat template already handles the ordering in the prompt,
        # so we just need to provide all images
        return {"image": valid_images}  # List of images (vLLM handles multiple images)
    
    def load_image_from_path(self, img_path):
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
    
    def prepare_all_requests(self, dataloader):
        """
        Prepare all requests upfront for batch processing.
        
        Args:
            dataloader: DataLoader for the SlideVQA data
            
        Returns:
            Tuple of (all_requests, request_metadata)
        """
        all_requests = []
        request_metadata = []
        
        logging.info("Preparing all requests for batch processing...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Preparing requests")):
            all_images_list, texts, qa_ids, gold_answers, questions = batch
            
            for idx, (images_list, text, qa_id, gold_answer, question) in enumerate(
                zip(all_images_list, texts, qa_ids, gold_answers, questions)
            ):
                # Load all images (up to 20)
                # Note: images_list may contain PIL.Image objects (from HF dataset) or paths (strings)
                loaded_images = []
                for img_item in images_list:
                    if img_item is not None:
                        # If it's already a PIL.Image, use it directly (from HF dataset)
                        if isinstance(img_item, Image.Image):
                            # Ensure it's RGB
                            if img_item.mode != 'RGB':
                                img_item = img_item.convert('RGB')
                            loaded_images.append(img_item)
                        else:
                            # Otherwise, it's a path string, load from disk
                            image = self.load_image_from_path(img_item)
                            if image is not None:
                                loaded_images.append(image)
                
                if len(loaded_images) == 0:
                    logging.error(f"Failed to load any images for {qa_id}")
                    continue
                
                # Use the working chat template approach
                # This will filter images from front if needed, keeping the back slides
                formatted_prompt, filtered_images = self._prepare_conversation_format(text, loaded_images)
                
                # Prepare multimodal data using the filtered images
                multi_modal_data = self._prepare_multimodal_data(filtered_images)
                
                # Create vLLM input format
                vllm_input = {
                    "prompt": formatted_prompt,
                    "multi_modal_data": multi_modal_data
                }
                    
                all_requests.append(vllm_input)
                request_metadata.append({
                    "qa_id": qa_id,
                    "text": text,
                    "gold_answer": gold_answer,
                    "question": question,
                    "batch_idx": batch_idx,
                    "idx_in_batch": idx,
                    "num_images": len(filtered_images)  # Use filtered images count
                })
        
        logging.info(f"Prepared {len(all_requests)} requests for vLLM batch processing")
        return all_requests, request_metadata
    
    def process_requests_in_batches(self, all_requests, request_metadata, batch_size):
        """
        Process all requests in batches using vLLM.
        
        Args:
            all_requests: List of vLLM input requests
            request_metadata: List of metadata for each request
            batch_size: Batch size for processing
            
        Returns:
            List of result dictionaries
        """
        all_results = []
        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(all_requests), batch_size), desc="Processing vLLM batches"):
            batch_requests = all_requests[i:i + batch_size]
            batch_metadata = request_metadata[i:i + batch_size]
            
            # Generate responses for this batch
            batch_results = self.llm.generate(batch_requests, self.sampling_params, lora_request=self.lora_request)

            # Extract generated text and process responses
            for result, metadata in zip(batch_results, batch_metadata):
                response = result.outputs[0].text
                result_dict = {
                    'qa_id': metadata['qa_id'],
                    'gold_answer': metadata['gold_answer'],
                    'question': metadata['question'],
                    'prompt': metadata['text'],
                    'response': response,
                    'generated_answer': response.split('[ANSWER]')[1].strip() if '[ANSWER]' in response else response,
                    'num_images': metadata['num_images']
                }
                all_results.append(result_dict)

        return all_results


def process_dataloader_with_vllm(inference_engine, dataloader, batch_size):
    """
    Process a complete data split using vLLM with true batch processing.
    
    Args:
        inference_engine: VLLMSlideVQAInferenceEngine instance
        dataloader: DataLoader for the split
        batch_size: Batch size for processing
        
    Returns:
        List of result dictionaries
    """
    # Step 1: Prepare all requests upfront
    all_requests, request_metadata = inference_engine.prepare_all_requests(dataloader)
    
    if not all_requests:
        raise ValueError("No valid requests to process")
    
    # Step 2: Process all requests in true batches
    all_results = inference_engine.process_requests_in_batches(
        all_requests, request_metadata, batch_size
    )
    
    return all_results 

def get_prompt_template(prompt_template: str):
    """
    Get prompt template for SlideVQA.
    If prompt_template is None, uses the global VLM_PROMPT_FOR_VQA.
    """
    if prompt_template is None:
        # Use global prompt template (matching ragk_slidevqa.py format)
        base_prompt = VLM_PROMPT_FOR_VQA
        # Add question part
        return base_prompt + "\n[QUESTION] {question}"
    else:
        with open(prompt_template, 'r') as f:
            template_content = f.read()
            # If template doesn't have {question} placeholder, add it
            if "{question}" not in template_content:
                template_content += "\n[QUESTION] {question}"
            return template_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset settings
    parser.add_argument("--hf_dataset_path", type=str, default="NTT-hil-insight/SlideVQA",
                        help="HuggingFace dataset path or local path")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--take_n", type=int, default=-1,
                        help="Number of examples to take (-1 for all)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Number of examples to offset")
    parser.add_argument("--img_basedir", type=str, default="../../shared_space/vqa_data/KBVQA_data/SlideVQA",
                        help="Base directory for image paths")
    parser.add_argument("--prompt_template", type=str, default=None,
                        help="Path to prompt template file")
    parser.add_argument("--prefill_ans_token", action="store_true",
                        help="Prefill [ANSWER] token in prompt")
    parser.add_argument("--use_oracle_slides", action="store_true",
                        help="Use only ground-truth slides (evidence_pages) instead of all slides")

    # Model settings
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)

    # Inference/VLLM settings
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Maximum model context length. Note: Qwen2-VL models typically support up to 32768 tokens, "
                             "but this can be increased for multimodal inputs with many images.") 
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation setting
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--use_bem", action="store_true",
                        help="Use BEM (BERT-based Equivalence Model) for evaluation in addition to EM")

    # Saving settings
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()

    output_filepath = f"{args.exp_name}/inference_results.csv"
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

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
            print(f"Taking {args.take_n} examples from the dataset.")
            slidevqa_dataset = slidevqa_dataset.shuffle(seed=args.seed).select([i for i in range(args.offset, args.offset + args.take_n)])
        else:
            print("Using the entire dataset.")
        
        print(f"Dataset loaded with {len(slidevqa_dataset)} items")

        # Get prompt template (use global VLM_PROMPT_FOR_VQA if no custom template)
        prompt_template_str = get_prompt_template(args.prompt_template)
        print("--------------------------------")
        if args.prompt_template:
            print(f"Read prompt template from {args.prompt_template}.")
        else:
            print("Using global VLM_PROMPT_FOR_VQA (matching ragk_slidevqa.py).")
        print(f"Prompt template: {prompt_template_str}")
        print("--------------------------------")

        def _make_prompt(question, num_images=20):
            """
            Create prompt with question, matching ragk_slidevqa.py format.
            Includes <image> tokens to mark image positions (up to 20 images).
            """
            # Use the prompt template (either from file or global variable)
            # Format with question
            prompt = prompt_template_str.format(question=question)
            
            # Add <image> tokens at the beginning to mark image positions
            # This ensures all images are properly associated with the prompt
            # Qwen2-VL uses <image> tokens to indicate where images should be placed
            image_tokens = " ".join(["<image>"] * num_images)
            prompt = f"{image_tokens} {prompt}"
            
            if args.prefill_ans_token:
                prompt += f"\n[ANSWER]"
            return prompt

        def collate_fn(batch):
            """Collate function for SlideVQA dataset."""
            all_images_list = []
            texts = []
            qa_ids = []
            gold_answers = []
            questions = []
            
            split_name = args.split if args.split else 'test'
            
            for b in batch:
                qa_id = b['qa_id']
                
                # Determine which pages to use
                if args.use_oracle_slides:
                    # Oracle mode: only use ground-truth evidence pages
                    evidence_pages = b.get('evidence_pages', [])
                    if not isinstance(evidence_pages, list):
                        evidence_pages = [evidence_pages] if evidence_pages is not None else []
                    
                    # Normalize evidence_pages to integers (matching ragk_slidevqa.py logic)
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
                    
                    # Only collect pages that are in evidence_pages
                    pages_to_collect = evidence_page_nums
                    if len(pages_to_collect) == 0:
                        logging.warning(f"No evidence_pages found for qa_id: {qa_id}, falling back to all pages")
                        pages_to_collect = list(range(1, 21))
                else:
                    # Normal mode: collect all pages (page_1 to page_20)
                    pages_to_collect = list(range(1, 21))
                
                # Collect page images
                images_for_item = []
                for page_num in pages_to_collect:
                    page_key = f'page_{page_num}'
                    if page_key in b and b[page_key] is not None:
                        page_image = b[page_key]
                        
                        # Priority 1: If it's already a PIL.Image object (from HF dataset), use it directly
                        if isinstance(page_image, Image.Image):
                            # Directly use the PIL.Image object, no disk I/O needed
                            images_for_item.append(page_image)
                            continue
                        
                        # Priority 2: If it's a path string, try to load from disk
                        if isinstance(page_image, str):
                            img_path = None
                            # If it's an absolute path
                            if os.path.isabs(page_image):
                                img_path = page_image
                            else:
                                # Try relative to img_basedir
                                img_path = os.path.join(args.img_basedir, page_image)
                            
                            # Check if path exists
                            if img_path and os.path.exists(img_path):
                                images_for_item.append(img_path)
                            else:
                                # Try constructing path using the standard format: {split_name}_{qa_id}_page_{page_num}.png
                                constructed_path = os.path.join(args.img_basedir, f"{split_name}_{qa_id}_page_{page_num}.png")
                                if os.path.exists(constructed_path):
                                    images_for_item.append(constructed_path)
                                else:
                                    # Try jpg extension
                                    constructed_path_jpg = os.path.join(args.img_basedir, f"{split_name}_{qa_id}_page_{page_num}.jpg")
                                    if os.path.exists(constructed_path_jpg):
                                        images_for_item.append(constructed_path_jpg)
                                    else:
                                        # Try without split_name prefix
                                        constructed_path_no_split = os.path.join(args.img_basedir, f"{qa_id}_page_{page_num}.png")
                                        if os.path.exists(constructed_path_no_split):
                                            images_for_item.append(constructed_path_no_split)
                                        else:
                                            logging.warning(f"Could not find image for {qa_id} page_{page_num}")
                
                if len(images_for_item) == 0:
                    logging.warning(f"No images found for qa_id: {qa_id}")
                
                all_images_list.append(images_for_item)
                
                # Create prompt with actual number of images loaded
                question = b['question']
                num_images_loaded = len(images_for_item)
                text = _make_prompt(question, num_images=num_images_loaded)
                texts.append(text)
                
                qa_ids.append(qa_id)
                gold_answers.append(b['answer'])
                questions.append(question)
            
            return all_images_list, texts, qa_ids, gold_answers, questions
        
        slidevqa_dataloader = DataLoader(
            slidevqa_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )

        # Check if max_model_len exceeds Qwen2-VL typical limit and warn
        if args.max_model_len > 32768:
            qwen2vl_models = ["Qwen2-VL", "qwen2-vl", "Qwen/Qwen2-VL"]
            is_qwen2vl = any(model_name in (args.base_model_path or args.model_path or "").lower() 
                           for model_name in qwen2vl_models)
            if is_qwen2vl:
                logger.warning(
                    f"max_model_len={args.max_model_len} exceeds Qwen2-VL's typical limit of 32768 tokens. "
                    f"This may cause issues. Consider reducing the number of images or using a smaller max_model_len."
                )
        
        # Initialize VLLMSlideVQAInferenceEngine
        engine = VLLMSlideVQAInferenceEngine(
            model_path=args.model_path,
            base_model_path=args.base_model_path,
            processor_path=args.processor_path,
            adapter_name_or_path=args.adapter_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_pixels=args.max_pixels,
            dtype=args.dtype,
        )
        
        all_results = process_dataloader_with_vllm(engine, slidevqa_dataloader, args.batch_size)
        all_results = pd.DataFrame(all_results)
        del engine
        torch.cuda.empty_cache()

        # save all_results to a CSV file
        all_results.to_csv(output_filepath, index=False)
        print(f"Results saved to {output_filepath}")

    # Evaluation
    if args.do_eval:
        output_filepath = f"{args.exp_name}/marked_inference_results.csv"
        score_filepath = f"{args.exp_name}/scores.json"
        # Always re-run evaluation, even if cache exists
        # This ensures evaluation uses the latest results and any updated evaluation logic
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
        
        # Compute relaxed Exact Match scores
        exact_matches = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing Relaxed Exact Match scores"):
            gold_answer = str(row['gold_answer']) if pd.notna(row['gold_answer']) else ""
            generated_answer = str(row['generated_answer']) if pd.notna(row['generated_answer']) else ""
            
            try:
                em_score = relaxed_exact_match(gold_answer, generated_answer)
                exact_matches.append(1 if em_score else 0)
            except Exception as e:
                logging.warning(f"Error computing EM for qa_id {row['qa_id']}: {e}")
                exact_matches.append(0)
        
        df['exact_match'] = exact_matches
        
        # Compute overall accuracy
        overall_accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        
        dict_to_report = {
            'relaxed_exact_match_accuracy': overall_accuracy,
            'total_samples': len(exact_matches),
            'correct_predictions': sum(exact_matches)
        }
        
        print("--------------------------------")
        print("Evaluation results:")
        print(f"Relaxed Exact Match Accuracy (gold answer in generated answer): {overall_accuracy:.4f} ({sum(exact_matches)}/{len(exact_matches)})")
        
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
        
        # Save results
        df.to_csv(output_filepath, index=False)
        with open(score_filepath, 'w') as f:
            json.dump(dict_to_report, f, indent=2)
        
        print(f"Evaluation results saved to {output_filepath}")
        print(f"Scores saved to {score_filepath}")

