import sys
sys.path.append('./src')
from vqa_datasets import load_passages
import json
from pprint import pprint
from datasets import load_from_disk
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




class VLLMInferenceEngine:
    """vLLM-based inference engine for hateful meme classification."""
    
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
                 model_family: str = "auto",
                 max_tokens: int = 1024,
                 **vllm_kwargs):
        """
        Initialize the vLLM inference engine.
        
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
        self.model_family = self._resolve_model_family(model_family, base_model_path or model_path)
        self.max_tokens = max_tokens
        
        # Determine GPU configuration
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        logger.info(f"Initializing vLLM engine...")
        logger.info(f"Model: {model_path}. NOTE: DEPRECATED!. Use base_model_path instead.")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Adapter: {adapter_name_or_path}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"Pipeline parallel size: {pipeline_parallel_size}")
        logger.info(f"Max model length: {max_model_len}")
        logger.info(f"Model family: {self.model_family}")
        
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
            "limit_mm_per_prompt": {"image": 1, "video": 1, "audio": 1},  # Each request has only 1 image
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
        if self.model_family == "llava":
            # Keep the same left-truncation policy used by the Qwen path.
            self.tokenizer.truncation_side = "left"
        else:
            # Qwen path keeps the rightmost content, which usually contains the final question.
            self.tokenizer.truncation_side = "left"
        
        stop_token_ids = self._get_stop_token_ids()
        if stop_token_ids:
            logger.info(f"Using stop token ids: {stop_token_ids}")

        # Initialize sampling parameters for greedy decoding (like the original)
        self.sampling_params = SamplingParams(
            repetition_penalty=1.0,
            temperature=0.0,  # Greedy decoding
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            skip_special_tokens=True,
            stop_token_ids=stop_token_ids,
        )
        
        # Initialize LoRA request if adapter is provided
        self.lora_request = None
        if enable_lora:
            self.lora_request = LoRARequest("adapter_0", 1, adapter_name_or_path)
            print(f"Loaded LoRA from {adapter_name_or_path}")
        
        logger.info("vLLM engine initialized successfully")

    @staticmethod
    def _resolve_model_family(model_family: str, model_path: str = None) -> str:
        if model_family != "auto":
            return model_family
        model_path = (model_path or "").lower()
        if "llava" in model_path:
            return "llava"
        if "qwen" in model_path:
            return "qwen2_vl"
        return "qwen2_vl"

    def _get_stop_token_ids(self) -> list[int]:
        stop_token_ids = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int) and eos_token_id >= 0:
            stop_token_ids.append(eos_token_id)

        if self.model_family == "llava":
            # LLaVA-Llama3 uses the Llama-3 chat template, where assistant turns
            # usually terminate with <|eot_id|> rather than the global EOS token.
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_token_id, int) and eot_token_id >= 0:
                stop_token_ids.append(eot_token_id)

        return list(dict.fromkeys(stop_token_ids))

    @staticmethod
    def _llava_default_system() -> str:
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )

    def _format_llava_fallback_prompt(self, prompt_text: str, has_image: bool) -> str:
        image_prefix = "<image>\n" if has_image else ""
        return f"{self._llava_default_system()}\nUSER: {image_prefix}{prompt_text} ASSISTANT:"

    def _apply_text_only_template(self, text_prompt: str) -> str:
        messages = [{"role": "user", "content": text_prompt}]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            if self.model_family == "llava":
                logger.warning(f"Falling back to default LLaVA text prompt template: {e}")
                return self._format_llava_fallback_prompt(text_prompt, has_image=False)
            raise
    
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
        # If max_pixels is set, estimate based on that
        if self.max_pixels:
            # Estimate processed size based on max_pixels while maintaining aspect ratio
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
    
    def _prepare_conversation_format(self, prompt_text: str, image: Image.Image) -> tuple[str, Image.Image]:
        """
        Prepare conversation format for vLLM using chat template.
        Includes truncation logic to ensure at least 1000 tokens are reserved for image
        and input length doesn't exceed max_model_len.
        Truncates text from the left, keeping the rightmost content.
        
        Args:
            prompt_text: The text prompt
            image: PIL Image object
            
        Returns:
            Tuple of (formatted_prompt_string, image)
        """
        # Remove the <image> token from the prompt template since we'll handle it through messages
        text_prompt = prompt_text.replace("<image>", "").strip()
        
        # Estimate image tokens (reserve at least 1000 tokens for image)
        image_tokens = self._estimate_image_tokens(image) if image is not None else 0
        reserved_for_image = max(1000, image_tokens)  # Reserve at least 1000 tokens for image
        
        # Reserve space for generation and safety margin
        # Generation: 1024 tokens, safety margin: 2048 tokens for chat template overhead
        reserved_for_generation = 1024
        reserved_for_safety = 2048
        total_reserved = reserved_for_image + reserved_for_generation + reserved_for_safety
        
        # Calculate available tokens for text
        available_tokens_for_text = max(0, self.max_model_len - total_reserved)
        
        # Estimate text tokens by creating a text-only message and applying chat template
        # This gives us a more accurate token count that accounts for chat template overhead
        text_only_prompt = self._apply_text_only_template(text_prompt)
        encoded_text = self.tokenizer(
            text_only_prompt,
            return_tensors=None,
            add_special_tokens=False,
            truncation=False
        )
        text_tokens = len(encoded_text['input_ids'])
        
        truncation_side = getattr(self.tokenizer, "truncation_side", "left")
        truncation_label = "right" if truncation_side == "right" else "left"

        # Truncate text if needed.
        if text_tokens > available_tokens_for_text:
            # We need to account for the chat template overhead, so we'll truncate more conservatively
            # Estimate chat template overhead (typically around 20-50 tokens)
            chat_template_overhead = 50
            available_for_text_content = max(0, available_tokens_for_text - chat_template_overhead)
            
            # Tokenize just the text content (without chat template)
            encoded_text_content = self.tokenizer(
                text_prompt,
                return_tensors=None,
                add_special_tokens=False,
                truncation=False
            )
            text_content_tokens = len(encoded_text_content['input_ids'])
            
            if text_content_tokens > available_for_text_content:
                if available_for_text_content == 0:
                    truncated_token_ids = []
                elif truncation_side == "right":
                    truncated_token_ids = encoded_text_content['input_ids'][:available_for_text_content]
                else:
                    truncated_token_ids = encoded_text_content['input_ids'][-available_for_text_content:]
                # Decode back to text after applying the model-family truncation policy.
                truncated_text_prompt = self.tokenizer.decode(
                    truncated_token_ids,
                    skip_special_tokens=False
                )
                logger.warning(
                    f"Truncating text from {truncation_label}: {text_content_tokens} tokens -> {available_for_text_content} tokens "
                    f"(Text length: {len(text_prompt)} chars -> {len(truncated_text_prompt)} chars, "
                    f"Total text tokens with template: {text_tokens}, Available: {available_tokens_for_text}, "
                    f"Image tokens: {image_tokens}, Reserved: {reserved_for_image}, "
                    f"Max model len: {self.max_model_len})"
                )
            else:
                truncated_text_prompt = text_prompt
        else:
            truncated_text_prompt = text_prompt
        
        if self.model_family == "llava":
            # LLaVA checkpoints converted from XTuner may not ship a HF chat template.
            # Try the tokenizer template first; fall back to LLaMA-Factory's llava template.
            if image is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": truncated_text_prompt}
                        ]
                    }
                ]
            else:
                messages = [{"role": "user", "content": truncated_text_prompt}]

            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                if image is not None and "<image>" not in formatted_prompt:
                    logger.warning("LLaVA chat template did not emit <image>; falling back to default LLaVA prompt template.")
                    formatted_prompt = self._format_llava_fallback_prompt(
                        truncated_text_prompt,
                        has_image=True,
                    )
            except Exception as e:
                logger.warning(f"Falling back to default LLaVA multimodal prompt template: {e}")
                formatted_prompt = self._format_llava_fallback_prompt(
                    truncated_text_prompt,
                    has_image=image is not None,
                )
        else:
            # Create conversation messages with image and (possibly truncated) text
            if image is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": truncated_text_prompt}
                        ]
                    }
                ]
            else:
                # Text-only conversation
                messages = [
                    {
                        "role": "user",
                        "content": truncated_text_prompt
                    }
                ]
            
            # Apply chat template - this is crucial for vLLM multimodal support
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        return formatted_prompt, image
    
    def _prepare_multimodal_data(self, image: Image.Image) -> dict:
        """
        Prepare multimodal data for vLLM input.
        This matches exactly the working implementation in generate_dpo_data_vllm.py
        """
        if image is None:
            return None
        
        # For vLLM with Qwen2.5-VL, we need to prepare the image in the expected format
        return {
            "image": image  # vLLM expects a single PIL image for Qwen2.5-VL
        }
    
    def load_image_from_dataloader(self, img_tensor):
        """
        Convert dataloader image tensor or path to PIL Image.
        """
        try:
            # Handle string paths (this is what we're getting from the dataset)
            if isinstance(img_tensor, str):
                if os.path.exists(img_tensor):
                    return Image.open(img_tensor).convert('RGB')
                else:
                    logging.error(f"Image path does not exist: {img_tensor}")
                    return None
            # Convert tensor image to PIL if necessary
            elif torch.is_tensor(img_tensor):
                # Handle different tensor formats
                if img_tensor.dim() == 3:  # CHW format
                    img_tensor = img_tensor.permute(1, 2, 0)  # Convert to HWC
                elif img_tensor.dim() == 4:  # BCHW format - take first image
                    img_tensor = img_tensor[0].permute(1, 2, 0)  # Convert to HWC
                
                img_np = img_tensor.cpu().numpy()
                
                # Normalize to 0-255 if needed
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0:  # Normalized to [0,1]
                        img_np = (img_np * 255).astype(np.uint8)
                    else:  # Already in [0,255] range
                        img_np = img_np.astype(np.uint8)
                
                img = Image.fromarray(img_np)
                return img
            elif isinstance(img_tensor, Image.Image):
                return img_tensor
            else:
                logging.error(f"Unsupported image type: {type(img_tensor)}")
                return None
        except Exception as e:
            logging.error(f"Error converting image: {e}")
            return None
    
    def prepare_all_requests(self, dataloader, ensure_gt_passage_in_ensemble):
        """
        Prepare all requests upfront for batch processing.
        Uses the exact same approach as the working generate_dpo_data_vllm.py script.
        
        Args:
            dataloader: DataLoader for the data
            query: Query template to use
            
        Returns:
            Tuple of (all_requests, request_metadata)
        """
        all_requests = []
        request_metadata = []
        
        logging.info("Preparing all requests for batch processing...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Preparing requests")):
            images, texts, gold_answers, question_ids, answers, question_types, questions = batch
            
            for idx, (img_tensor, text, question_id, gold_answer, ans, question_type, question) in enumerate(zip(images, texts, question_ids, gold_answers, answers, question_types, questions)):
                formatted_question = f"<image>{text}"
                
                # Convert dataloader tensor to PIL Image using the new method
                image = self.load_image_from_dataloader(img_tensor)
                if image is None:
                    logging.error(f"Failed to convert image tensor for {question_id}")
                    exit()
                
                # Use the working chat template approach with truncation
                formatted_prompt, filtered_image = self._prepare_conversation_format(f"{formatted_question}", image)
                
                # Prepare multimodal data using the filtered image
                multi_modal_data = self._prepare_multimodal_data(filtered_image)
                
                # Create vLLM input format exactly like the working script
                # if multi_modal_data is not None:
                    # For multimodal inputs, use the formatted prompt with multimodal data
                vllm_input = {
                    "prompt": formatted_prompt,
                    "multi_modal_data": multi_modal_data
                }
                    # else:
                    #     # For text-only inputs, just use the formatted prompt
                    #     vllm_input = {
                    #         "prompt": formatted_prompt
                    #     }
                    
                all_requests.append(vllm_input)
                request_metadata.append({
                    "question_id": question_id,
                    "text": text,
                    "gold_answer": gold_answer,
                    "batch_idx": batch_idx,
                    "idx_in_batch": idx,
                    "answers": ans,
                    "question_type": question_type,
                    "question": question
                })
                    
                # except Exception as e:
                    # logging.error(f"Error preparing request for image {question_id}: {e}")
                    # continue
        
        logging.info(f"Prepared {len(all_requests)} requests for vLLM batch processing")
        return all_requests, request_metadata
    
    def process_requests_in_batches(self, all_requests, request_metadata, batch_size, ensure_gt_passage_in_ensemble):
        """
        Process all requests in batches using vLLM.
        Uses the exact same approach as the working generate_dpo_data_vllm.py script.
        
        Args:
            all_requests: List of vLLM input requests
            request_metadata: List of metadata for each request
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (batch_results, batch_text_preds, batch_labels, batch_rewards, batch_ids)
        """
        all_results = []
        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(all_requests), batch_size), desc="Processing vLLM batches"):
            batch_requests = all_requests[i:i + batch_size]
            batch_metadata = request_metadata[i:i + batch_size]
            
            # try:
                # Generate responses for this batch - this is where the true batching happens!
            batch_results = self.llm.generate(batch_requests, self.sampling_params, lora_request=self.lora_request)

            # Extract generated text and process responses
            for result, metadata in zip(batch_results, batch_metadata):
                response = result.outputs[0].text
                result_dict = {
                    'question_id': metadata['question_id'],
                    'gold_answer': metadata['gold_answer'],
                    'prompt': metadata['text'],
                    'response': response,
                    'generated_answer': response.split('[ANSWER]')[1].strip() if '[ANSWER] ' in response else response,
                    'answers': metadata['answers'],
                    'question_type': metadata['question_type'],
                    'question': metadata['question']
                }
                all_results.append(result_dict)

                # del batch_results
                # gc.collect()
                
            # except Exception as e:
            #     logging.error(f"Error processing batch {i//batch_size + 1}: {e}")
            #     continue

        return all_results


def process_dataloader_with_vllm(inference_engine, dataloader, batch_size, ensure_gt_passage_in_ensemble):
    """
    Process a complete data split using vLLM with true batch processing.
    
    Args:
        inference_engine: VLLMInferenceEngine instance
        dataloader: DataLoader for the split
        split_name: Name of the split being processed
        args: Command line arguments
        
    Returns:
        Dictionary with results and metrics
    """
    # Step 1: Prepare all requests upfront (like the working sampling script)
    all_requests, request_metadata = inference_engine.prepare_all_requests(dataloader, ensure_gt_passage_in_ensemble)
    
    if not all_requests:
        raise ValueError("No valid requests to process")
    
    # Step 2: Process all requests in true batches (this is where the performance gain happens!)
    all_results = inference_engine.process_requests_in_batches(
        all_requests, request_metadata, args.batch_size, ensure_gt_passage_in_ensemble
    )
    
    return all_results 

def get_prompt_template(prompt_template: str):
    if prompt_template is None:
        return (
            "Answer the question after [QUESTION] about the image."
            "A retriever has retrieved a relevant document for you and provided it after [EVIDENCE]."
            "Give your answer after [ANSWER]\n"
        )
    else:
        with open(prompt_template, 'r') as f:
            return f.read()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset settings
    parser.add_argument("--retrieval_ds_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None) 
    parser.add_argument("--take_n", type=int, default=-1)
    parser.add_argument("--img_basedir", type=str, default='')
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--prefill_ans_token", action="store_true")
    parser.add_argument("--max_words_per_evidence", type=int, default=1024)
    parser.add_argument("--ensure_gt_passage_in_ensemble", action="store_true")

    # Retrieval settings
    parser.add_argument("--retrieval_field", type=str, default=None, choices=['retrieved_passage', 'reranked_passage'])
    parser.add_argument("--retrieval_topk", type=int, default=5)

    # Model settings
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)
    parser.add_argument("--model_family", type=str, default="auto", choices=["auto", "qwen2_vl", "llava"])

    # Inference/VLLM settings
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_model_len", type=int, default=32768) 
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation setting
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--use_cache", action="store_true")

    # Saving settings
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()

    output_filepath = f"{args.exp_name}/inference_results.csv"
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    vqa_dataset = load_from_disk(args.retrieval_ds_path)

    if os.path.exists(output_filepath) and args.use_cache:
        all_results = pd.read_csv(output_filepath)
    else:
        # Load VQA dataset
        # vqa_dataset = load_vqa_dataset(args.dataset_name, split=args.split, img_basedir=args.img_basedir, take_n=args.take_n, seed=args.ds_seed)
        if args.take_n > 0:
            print(f"Taking {args.take_n} examples from the dataset.")
            vqa_dataset = vqa_dataset.shuffle(seed=42).select([i for i in range(args.take_n)])
        else:
            print("Using the entire dataset.")

        passages, pid_to_content_map = load_passages(args.dataset_name, split='test')

        VLM_PROMPT_FOR_VQA = get_prompt_template(args.prompt_template)
        print("--------------------------------")
        print(f"Read prompt template from {args.prompt_template}.")
        print(f"Prompt template: {VLM_PROMPT_FOR_VQA}.")
        print("--------------------------------")

        def _make_evidence_part(passage_dict):
            pid = passage_dict['passage_id']
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
        
        def _make_prompt(question, passage_dicts):
            evidence_parts = [_make_evidence_part(passage_dict) for passage_dict in passage_dicts]
            prompt = VLM_PROMPT_FOR_VQA + ' '.join(evidence_parts)
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

        def collate_fn(batch):
            images = [os.path.join(args.img_basedir, b['img_path']) for b in batch]
            
            # Process each item to handle ensure_gt_passage_in_ensemble
            processed_texts = []
            for b in batch:
                # Get retrieved passages
                retrieved_passages = b[args.retrieval_field][:args.retrieval_topk]
                
                # Handle ensure_gt_passage_in_ensemble
                if args.ensure_gt_passage_in_ensemble:
                    gt_passage_id = b['pos_item_ids'][0]
                    retrieved_passage_ids = [p['passage_id'] for p in retrieved_passages]
                    
                    # If GT passage is not in retrieved passages, replace the first one
                    if gt_passage_id not in retrieved_passage_ids:
                        score_field = 'score' if args.retrieval_field == 'retrieved_passage' else 'rerank_score'
                        gt_passage = {
                            'passage_id': gt_passage_id,
                            'passage_content': b['pos_item_contents'][0],
                            score_field: 1.0
                        }
                        retrieved_passages[0] = gt_passage
                
                # Create prompt with processed passages
                text = _make_prompt(b['question'], retrieved_passages)
                processed_texts.append(text)
            
            gold_answers = [b['gold_answer'] for b in batch]
            question_ids = [b['question_id'] for b in batch]
            answers = [b['answers'] for b in batch]
            question_types = [b['question_type'] for b in batch]
            questions = [b['question'] for b in batch]
            return images, processed_texts, gold_answers, question_ids, answers, question_types, questions
        
        vqa_dataloader = DataLoader(vqa_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        # Initialize VLLMInferenceEngine
        engine = VLLMInferenceEngine(
            model_path=args.model_path,
            base_model_path=args.base_model_path,
            processor_path=args.processor_path,
            adapter_name_or_path=args.adapter_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_pixels=args.max_pixels,
            dtype=args.dtype,
            model_family=args.model_family,
            max_tokens=args.max_tokens,
        )
        
        all_results = process_dataloader_with_vllm(engine, vqa_dataloader, args.batch_size, args.ensure_gt_passage_in_ensemble)
        all_results = pd.DataFrame(all_results)
        del engine
        torch.cuda.empty_cache()

        # save all_results to a CSV file
        all_results.to_csv(output_filepath, index=False)

    # Evaluation
    if args.do_eval:
        output_filepath = f"{args.exp_name}/marked_inference_results.csv"
        score_filepath = f"{args.exp_name}/scores.json"
        if os.path.exists(output_filepath) and os.path.exists(score_filepath) and args.use_cache:
            dict_to_report = json.load(open(score_filepath, 'r'))
            print("Evaluation results loaded from cache.")
            print("Evaluation results:")
            print(dict_to_report)
            print("--------------------------------")
            exit()

        if args.dataset_name == 'EVQA':
            sys.path.append('./src/evaluation')
            # df = vqa_dataset.to_pandas()

            from evqa_eval_1004 import process_row as eval_process_row
            from evqa_eval_1004 import process_row_mp as eval_process_row_mp
            from evqa_eval_1004 import extract_queries_and_retrieved_docs

            df = all_results
            df['prediction'] = df['generated_answer']

            # df['prediction'] = all_results['generated_answer']
            # df['prompt'] = all_results['prompt']
            # columns_to_keep = ['question_id', 'question','question_type', 'prompt','gold_answer', 'answers', 'prediction']
            # df = df[columns_to_keep]
            # breakpoint()
            all_eval_results = []
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')

            if False:
                for row in tqdm(df.itertuples(), total=len(df)):
                    eval_result = eval_process_row(row)
                    all_eval_results.append(eval_result)
                dict_to_report = {f"avg_{k}": sum([res[k] for res in all_eval_results])/len(all_eval_results) for k in all_eval_results[0]}
                for k in all_eval_results[0]:
                    df[k] = [res[k] for res in all_eval_results]
            else:
                import multiprocessing
                with multiprocessing.Pool(processes=8) as pool:
                    all_eval_results = list(tqdm(pool.imap(eval_process_row_mp, df.iterrows(), chunksize=1), total=len(df)))
                dict_to_report = {f"avg_score": sum(all_eval_results)/len(all_eval_results)}
                df['score'] = all_eval_results
            
            torch.cuda.empty_cache()
            
            print("--------------------------------")
            print("Evaluation results:")
            print(dict_to_report)
            print("--------------------------------")

            df.to_csv(output_filepath)
            with open(f'{os.path.dirname(output_filepath)}/scores.json', 'w') as f:
                json.dump(dict_to_report, f)
            print("Evaluation results saved to", os.path.dirname(output_filepath))
        else:
            raise NotImplementedError(f"Evaluation for {args.dataset_name} not implemented")