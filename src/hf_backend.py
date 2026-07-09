import atexit
import json
import os
import re
import shutil
import tempfile
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoConfig, DynamicCache
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None
try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLForConditionalGeneration = None
try:
    from transformers import LlavaForConditionalGeneration
except Exception:
    LlavaForConditionalGeneration = None
try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None
from qwen_vl_utils import process_vision_info
import torch

try:
    from . import cache_compat
except ImportError:
    import cache_compat


def _make_peft_export_dir(model_path: str, adapter_path: str):
    """
    Create a temporary directory with a "full" PEFT layout so that a single
    from_pretrained(peft_export_dir) loads base from model_path and adapter from
    this dir (avoids adapter-on-CPU when using device_map="auto").
    - adapter_config.json with base_model_name_or_path set to model_path (absolute)
    - adapter weights copied from adapter_path (adapter_model.safetensors or .bin)
    Returns the temp dir path. Registers atexit to remove it on process exit.
    """
    adapter_path = os.path.abspath(os.path.expanduser(adapter_path))
    # Use absolute path only when model_path is an existing local dir (so loading is cwd-independent).
    # Otherwise keep model_path as-is so Hub ids like "Qwen/Qwen2.5-VL-3B-Instruct" work.
    model_path_expanded = os.path.expanduser(model_path)
    if os.path.isdir(model_path_expanded) or os.path.isfile(
        os.path.join(model_path_expanded, "config.json")
    ):
        base_model_for_config = os.path.abspath(model_path_expanded)
    else:
        base_model_for_config = model_path
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")

    tmpdir = tempfile.mkdtemp(prefix="peft_export_")
    try:
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        adapter_config["base_model_name_or_path"] = base_model_for_config
        adapter_config = _sanitize_target_modules_if_needed(adapter_config, base_model_for_config)
        with open(os.path.join(tmpdir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

        for name in ("adapter_model.safetensors", "adapter_model.bin"):
            src = os.path.join(adapter_path, name)
            if os.path.isfile(src):
                dst = os.path.join(tmpdir, name)
                # LLaVA checkpoints may need key-space normalization for PEFT loading.
                if name == "adapter_model.safetensors":
                    _rewrite_llava_adapter_safetensors_if_needed(src, dst, base_model_for_config)
                else:
                    shutil.copy2(src, dst)
                break
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise FileNotFoundError(
                f"No adapter weights (adapter_model.safetensors or .bin) in {adapter_path}"
            )

        def _cleanup():
            shutil.rmtree(tmpdir, ignore_errors=True)

        atexit.register(_cleanup)
        return tmpdir
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def _sanitize_target_modules_if_needed(adapter_config: dict, base_model_name_or_path: str) -> dict:
    """
    Some older/specialized LLaVA LoRA checkpoints contain malformed target_modules entries
    (e.g. "29.self_attn.k_proj" or "model.layers.11.self_attn.q_proj") that do not map
    to current module names and cause large MISSING init at load time.
    For LLaVA only, normalize to canonical module suffixes.
    """
    target_modules = adapter_config.get("target_modules")
    if not isinstance(target_modules, list) or len(target_modules) == 0:
        return adapter_config

    model_type = ""
    try:
        cfg = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        model_type = str(getattr(cfg, "model_type", "")).lower()
    except Exception:
        model_type = ""

    is_llava = (model_type == "llava") or ("llava" in str(base_model_name_or_path).lower())
    if not is_llava:
        return adapter_config

    malformed_entry = any(
        isinstance(m, str)
        and (
            re.match(r"^\d+\.", m) is not None
            or re.match(r"^model\.layers\.\d+\.", m) is not None
        )
        for m in target_modules
    )

    if malformed_entry:
        canonical = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        adapter_config["target_modules"] = canonical
        # Avoid accidentally targeting vision tower attention projections.
        adapter_config["exclude_modules"] = ["vision_tower"]
        print(
            "[HF Backend] Detected malformed LLaVA LoRA target_modules; "
            f"fallback to canonical targets: {canonical}"
        )

    return adapter_config


def _rewrite_llava_adapter_safetensors_if_needed(src_path: str, dst_path: str, base_model_name_or_path: str) -> None:
    """
    Rewrite known-bad LLaVA adapter key prefix variants to the PEFT-expected namespace.
    If no rewrite is needed, performs a direct copy.
    """
    model_type = ""
    try:
        cfg = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        model_type = str(getattr(cfg, "model_type", "")).lower()
    except Exception:
        model_type = ""

    is_llava = (model_type == "llava") or ("llava" in str(base_model_name_or_path).lower())
    if not is_llava:
        shutil.copy2(src_path, dst_path)
        return

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except Exception:
        shutil.copy2(src_path, dst_path)
        return

    old_prefix = "base_model.model.language_model.model."
    new_prefix = "base_model.model.model.language_model."

    with safe_open(src_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        needs_rewrite = any(k.startswith(old_prefix) for k in keys)
        if not needs_rewrite:
            shutil.copy2(src_path, dst_path)
            return

        rewritten = {}
        for k in keys:
            nk = k
            if k.startswith(old_prefix):
                nk = new_prefix + k[len(old_prefix) :]
            rewritten[nk] = f.get_tensor(k)

    save_file(rewritten, dst_path)
    print("[HF Backend] Rewrote LLaVA adapter safetensors keys for PEFT namespace compatibility.")


def _get_config_attr(config, attr):
    """Get attribute from config, with fallback to text_config for VL configs (e.g. Qwen2_5_VLConfig)."""
    val = getattr(config, attr, None)
    if val is not None:
        return val
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return getattr(text_config, attr, None)
    return None


def _make_dynamic_cache(num_hidden_layers=None):
    """Create a DynamicCache compatible with both old (num_hidden_layers=) and new (no-arg) transformers API."""
    try:
        if num_hidden_layers is not None:
            return DynamicCache(num_hidden_layers=num_hidden_layers)
    except TypeError:
        pass
    return DynamicCache()


def _token_to_id(tok, token_str):
    """Get token id from PreTrainedTokenizer or tokenizers.Tokenizer."""
    if hasattr(tok, "convert_tokens_to_ids"):
        out = tok.convert_tokens_to_ids(token_str)
        return out if isinstance(out, int) else (out[0] if out else None)
    try:
        enc = tok.encode(token_str, add_special_tokens=False)
    except TypeError:
        enc = tok.encode(token_str)
    ids = enc.ids if hasattr(enc, "ids") else list(enc)
    return ids[0] if ids else None


def _get_special_id(tok, attr):
    """Get eos_token_id or pad_token_id from PreTrainedTokenizer or tokenizers.Tokenizer."""
    if hasattr(tok, attr):
        val = getattr(tok, attr)
        if val is not None:
            return val
    if attr == "eos_token_id":
        return _token_to_id(tok, "</s>") or _token_to_id(tok, "<|endoftext|>") or _token_to_id(tok, "<|im_end|>")
    if attr == "pad_token_id":
        return _token_to_id(tok, "<|pad|>") or _token_to_id(tok, "<pad>") or 0
    return None


class HFQwen2VLBackend:
    _VISION_KEYS = ("pixel_values", "image_grid_thw", "image_sizes")

    def __init__(
        self,
        model_path,
        processor_path,
        adapter_name_or_path=None,
        max_batch_size_per_forward=None,
        attn_implementation="flash_attention_2",
        force_single_device_per_rank=False,
        local_rank=0,
    ):
        print("Initialize HF Backend for QWen2VL")
        self.processor = AutoProcessor.from_pretrained(processor_path)
        model_cls = self._select_model_cls(model_path)

        adapter_path = (adapter_name_or_path or "").strip()
        if force_single_device_per_rank:
            # For torchrun/DDP-style passage-parallel runs, each rank must bind to a unique GPU.
            # Using device_map="auto" can shard across multiple visible GPUs per rank and causes
            # NCCL duplicate-GPU errors when collectives are initialized.
            device_map = {"": int(local_rank)}
            print(f"[HF Backend] Force single-device mode enabled, local_rank={local_rank}, device_map={device_map}")
        else:
            device_map = "auto"
        if adapter_path:
            # Single from_pretrained(peft_export_dir) so base and adapter use the same device_map
            peft_export_dir = _make_peft_export_dir(model_path, adapter_path)
            self.model = model_cls.from_pretrained(
                peft_export_dir,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
            print(f"Loaded LoRA from {adapter_path} (via PEFT export)")
        else:
            self.model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )

        self.model.eval()
        self.model_type = getattr(getattr(self.model, "config", None), "model_type", "")
        self._patch_llava_processor_defaults()
        
        # Memory management: max batch size per forward pass
        self.max_batch_size_per_forward = max_batch_size_per_forward
        if max_batch_size_per_forward is not None:
            print(f"[HF Backend] Max batch size per forward: {max_batch_size_per_forward}")

        # Stop tokens (support both PreTrainedTokenizer and tokenizers.Tokenizer).
        # LLaVA-Llama3 uses Llama-3 chat turns, which often terminate with <|eot_id|>
        # rather than the global EOS token.
        tok = self.tokenizer
        stop_token_list = [
            _token_to_id(tok, "<|im_end|>"),
            _get_special_id(tok, "eos_token_id"),
            _get_special_id(tok, "pad_token_id"),
        ]
        if "llava" in str(self.model_type).lower():
            stop_token_list.append(_token_to_id(tok, "<|eot_id|>"))
        self.stop_token_list = list(dict.fromkeys(t for t in stop_token_list if t is not None))
        print(f"[HF Backend] Stop token ids: {self.stop_token_list}")

    def _patch_llava_processor_defaults(self):
        """
        Patch missing LlavaProcessor attributes that can be absent in some checkpoints.
        Mirrors LLaMA-Factory fallback logic to avoid NoneType errors in processor math.
        """
        p = self.processor
        cls_name = p.__class__.__name__.lower()
        is_llava = "llava" in cls_name or "llava" in str(self.model_type).lower()
        if not is_llava:
            return

        vision_cfg = getattr(getattr(self.model, "config", None), "vision_config", None)
        image_processor = getattr(p, "image_processor", None)

        if getattr(p, "patch_size", None) is None:
            patch_size = (
                getattr(vision_cfg, "patch_size", None)
                or getattr(image_processor, "patch_size", None)
                or 14
            )
            setattr(p, "patch_size", patch_size)
            print(f"[HF Backend] Missing LlavaProcessor.patch_size, fallback to {patch_size}.")

        if getattr(p, "vision_feature_select_strategy", None) is None:
            strategy = getattr(getattr(self.model, "config", None), "vision_feature_select_strategy", None) or "default"
            setattr(p, "vision_feature_select_strategy", strategy)
            print(
                "[HF Backend] Missing LlavaProcessor.vision_feature_select_strategy, "
                f"fallback to {strategy}."
            )

        if getattr(p, "num_additional_image_tokens", None) in [None, 0]:
            setattr(p, "num_additional_image_tokens", 1)
            print("[HF Backend] Missing LlavaProcessor.num_additional_image_tokens, fallback to 1.")

        tok = self.tokenizer
        if hasattr(tok, "truncation_side"):
            tok.truncation_side = "left"
            print("[HF Backend] Set LLaVA tokenizer truncation_side='left'.")
    
    @property
    def tokenizer(self):
        """Tokenizer from processor; supports .tokenizer or ._tokenizer (e.g. Qwen2Tokenizer)."""
        p = self.processor
        return getattr(p, "tokenizer", None) or getattr(p, "_tokenizer", p)

    @property
    def device(self):
        return self.model.device

    @staticmethod
    def _select_model_cls(model_path: str):
        """
        Select appropriate model class from config/model type.

        - If model_path clearly refers to a *vision-language* checkpoint (contains 'VL'
          or '-VL-'), use the Qwen2 VL classes.
        - Otherwise, assume a text-only Qwen/Qwen2.5 checkpoint and fall back to
          AutoModelForCausalLM so we don't apply VL-specific assumptions like
          rope_parameters['mrope_section'] to text models.
        """
        model_type = ""
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", "")
        except Exception:
            model_type = ""

        # Qwen-VL family
        if model_type in {"qwen2_5_vl"} or any(token in model_path for token in ["Qwen2.5-VL", "qwen2_5_vl"]):
            if Qwen2_5_VLForConditionalGeneration is None:
                raise ImportError("Qwen2_5_VLForConditionalGeneration not available in transformers.")
            return Qwen2_5_VLForConditionalGeneration
        if model_type in {"qwen2_vl"} or any(token in model_path for token in ["Qwen2-VL", "qwen2_vl"]):
            if Qwen2VLForConditionalGeneration is None:
                raise ImportError("Qwen2VLForConditionalGeneration not available in transformers.")
            return Qwen2VLForConditionalGeneration

        # LLaVA family
        if model_type in {"llava"} or "llava" in model_path.lower():
            if LlavaForConditionalGeneration is not None:
                return LlavaForConditionalGeneration
            if AutoModelForVision2Seq is not None:
                return AutoModelForVision2Seq
            raise ImportError("No Llava-compatible model class found in this transformers version.")

        # Text-only Qwen/Qwen2.5: use generic causal LM
        return AutoModelForCausalLM

    def _get_prompt_only_inputs(self, x, zk, image_inputs=None, video_inputs=None):
        """
        Build model inputs for the prompt only (user message with evidence, no assistant reply).
        Same zk/message logic as prepare_input; returns inputs dict for tokenizing prompt + generation prompt.
        """
        # Handle zk as either string (VQA) or dict (SlideVQA)
        if isinstance(zk, dict):
            passage_text = zk.get('text', '')
            passage_images = zk.get('images', [])
            passage_images = [img for img in passage_images if img is not None]
            input_text = x["text"].replace("<<<EVIDENCE>>>", passage_text)
            images_to_use = passage_images
        else:
            input_text = x["text"].replace("<<<EVIDENCE>>>", zk)
            images_to_use = [x["image"]] if x.get("image") is not None else []

        model_type_str = str(self.model_type).lower()
        is_qwen_vl = "qwen" in model_type_str
        is_llava = "llava" in model_type_str

        has_images = any(img is not None for img in images_to_use)
        if has_images:
            content = []
            for img in images_to_use:
                if img is not None:
                    if is_qwen_vl:
                        content.append({"type": "image", "image": img})
                    else:
                        # LLaVA-style processors only need a typed image marker in messages.
                        content.append({"type": "image"})
            content.append({"type": "text", "text": input_text})
            user_content = content
        else:
            user_content = input_text

        messages = [{"role": "user", "content": user_content}]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except ValueError as e:
            # Some LLaVA checkpoints (e.g., xtuner conversions) do not ship HF chat_template metadata.
            # Mirror LLaMA-Factory template:llava formatting only for LLaVA in this fallback path.
            if is_llava and "does not have a chat template" in str(e):
                llava_default_system = (
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions."
                )
                # IMPORTANT: LLaVA expects explicit <image> placeholders in text when image
                # features are provided; otherwise model raises token/feature mismatch.
                user_text = input_text
                if has_images:
                    image_prefix = "".join("<image>\n" for _ in images_to_use if _ is not None)
                    user_text = f"{image_prefix}{input_text}"
                text = f"{llava_default_system}\nUSER: {user_text} ASSISTANT:"
            else:
                raise

        if image_inputs is None and video_inputs is None:
            if is_qwen_vl:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [img for img in images_to_use if img is not None]
                video_inputs = None

        # LLaVA fast image processor crashes on images=[]; normalize empty image/video
        # containers to None so text-only (z0) passages are handled correctly.
        if isinstance(image_inputs, (list, tuple)) and len(image_inputs) == 0:
            image_inputs = None
        if isinstance(video_inputs, (list, tuple)) and len(video_inputs) == 0:
            video_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        return inputs.to(device=self.device)

    def _normalize_generated_tokens_to_ids(self, generated_tokens):
        """Normalize generated_tokens to a list of int token ids."""
        if not generated_tokens:
            return []
        raw = getattr(generated_tokens, "ids", generated_tokens)
        ids = list(raw)
        if ids and isinstance(ids[0], int):
            return ids
        return [int(t) for t in ids]

    def prepare_input(self, x, generated_tokens, zk, image_inputs=None, video_inputs=None, past_key_values=None):
        """
        x: {"text": ..., "image": <img_path>} (image can be None for SlideVQA where passages have their own images)
        generated_tokens: list of token ids (continuation only, or full context when using cache)
        zk: passage text (str) or passage dict (dict with {'images': [...], 'text': '...'}) for SlideVQA
        past_key_values: if not None, incremental step: only the last token is used as input (KV cache reuse).
        """
        ids = self._normalize_generated_tokens_to_ids(generated_tokens)

        if past_key_values is not None:
            # Incremental step: pass only the new token; prefix is in the cache.
            last_id = ids[-1] if ids else 0
            try:
                k, _ = cache_compat._cache_get_layer(past_key_values, 0)
                past_length = k.shape[2]
                # Full-length attention mask: past tokens + current token (required for KV cache)
                inputs = {
                    "input_ids": torch.tensor([[last_id]], device=self.device, dtype=torch.long),
                    "attention_mask": torch.ones(1, past_length + 1, device=self.device, dtype=torch.long),
                    "position_ids": torch.tensor([[past_length]], device=self.device, dtype=torch.long),
                }
            except Exception:
                inputs = {
                    "input_ids": torch.tensor([[last_id]], device=self.device, dtype=torch.long),
                    "attention_mask": torch.ones(1, 1, device=self.device, dtype=torch.long),
                }
            return inputs

        # Full sequence: prompt token ids + continuation token ids (no decode round-trip)
        prompt_inputs = self._get_prompt_only_inputs(x, zk, image_inputs, video_inputs)
        input_ids = prompt_inputs["input_ids"]
        attention_mask = prompt_inputs["attention_mask"]

        if ids:
            continuation = torch.tensor([ids], device=self.device, dtype=torch.long)
            input_ids = torch.cat([input_ids, continuation], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, len(ids), device=self.device, dtype=attention_mask.dtype),
            ], dim=1)

        out = dict(prompt_inputs)
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        return out
    
    def prepare_batched_input(self, x, generated_tokens, passages, image_inputs=None, video_inputs=None):
        """
        Prepare batched inputs for multiple passages with proper padding.
        
        Args:
            x: {"text": ..., "image": <img_path>} (image can be None for SlideVQA)
            generated_tokens: list of tokens (or token ids)
            passages: list of passage texts (str) or passage dicts (dict with {'images': [...], 'text': '...'}) for SlideVQA
            image_inputs: Optional preprocessed image inputs
            video_inputs: Optional preprocessed video inputs
            
        Returns:
            Batched inputs ready for forward pass
        """
        # Prepare individual inputs first
        individual_inputs = []
        for zk in passages:
            inputs = self.prepare_input(x, generated_tokens, zk, image_inputs, video_inputs)
            individual_inputs.append(inputs)
        
        # Find maximum sequence length for padding
        max_seq_len = max(inp["input_ids"].shape[1] for inp in individual_inputs)
        
        # Pad and batch the inputs
        batched_inputs = {}
        sample_meta = [
            {
                "orig_idx": i,
                "has_image": False,
                "vision_counts": {},
            }
            for i in range(len(individual_inputs))
        ]
        all_keys = set()
        for inp in individual_inputs:
            all_keys.update(inp.keys())

        for key in sorted(all_keys):
            if key in ["input_ids", "attention_mask"]:
                # Pad sequences to max length using LEFT padding
                padded_tensors = []
                for inp in individual_inputs:
                    tensor = inp[key]
                    seq_len = tensor.shape[1]
                    if seq_len < max_seq_len:
                        # Create LEFT padding tensor
                        if key == "input_ids":
                            pad_value = _get_special_id(self.tokenizer, "pad_token_id")
                        else:  # attention_mask
                            pad_value = 0
                        padding = torch.full(
                            (tensor.shape[0], max_seq_len - seq_len), 
                            pad_value, 
                            device=tensor.device, 
                            dtype=tensor.dtype
                        )
                        # LEFT padding: pad BEFORE the sequence
                        padded_tensor = torch.cat([padding, tensor], dim=1)
                    else:
                        padded_tensor = tensor
                    padded_tensors.append(padded_tensor)
                
                batched_inputs[key] = torch.cat(padded_tensors, dim=0)
            
            elif key == "pixel_values":
                # For mixed text-only + multimodal samples (e.g., z0), keep only real image tensors.
                # Qwen2-VL cannot consume synthetic zero-length visual rows.
                tensors = []
                for i, inp in enumerate(individual_inputs):
                    if (
                        key in inp
                        and torch.is_tensor(inp[key])
                        and inp[key].shape[0] > 0
                        and inp[key].numel() > 0
                    ):
                        tensors.append(inp[key])
                        sample_meta[i]["vision_counts"][key] = int(inp[key].shape[0])
                        sample_meta[i]["has_image"] = True
                    else:
                        sample_meta[i]["vision_counts"][key] = 0
                if tensors:
                    batched_inputs[key] = torch.cat(tensors, dim=0)

            elif key == "image_grid_thw":
                # Keep only rows for actual images; text-only samples contribute no rows.
                tensors = []
                for i, inp in enumerate(individual_inputs):
                    if (
                        key in inp
                        and torch.is_tensor(inp[key])
                        and inp[key].shape[0] > 0
                        and inp[key].numel() > 0
                    ):
                        tensors.append(inp[key])
                        sample_meta[i]["vision_counts"][key] = int(inp[key].shape[0])
                    else:
                        sample_meta[i]["vision_counts"][key] = 0
                if tensors:
                    batched_inputs[key] = torch.cat(tensors, dim=0)

            elif key == "image_sizes":
                # LLaVA may return image_sizes; drop empty rows from text-only passages.
                tensors = []
                for i, inp in enumerate(individual_inputs):
                    if (
                        key in inp
                        and torch.is_tensor(inp[key])
                        and inp[key].shape[0] > 0
                        and inp[key].numel() > 0
                    ):
                        tensors.append(inp[key])
                        sample_meta[i]["vision_counts"][key] = int(inp[key].shape[0])
                    else:
                        sample_meta[i]["vision_counts"][key] = 0
                if tensors:
                    batched_inputs[key] = torch.cat(tensors, dim=0)
            
            else:
                # For other keys, concatenate along batch dimension
                if all(key in inp for inp in individual_inputs):
                    batched_inputs[key] = torch.cat([inp[key] for inp in individual_inputs], dim=0)
        batched_inputs["__sample_meta__"] = sample_meta
        return batched_inputs

    def _get_sample_meta(self, inputs, batch_size):
        sample_meta = inputs.get("__sample_meta__")
        if not isinstance(sample_meta, list) or len(sample_meta) != batch_size:
            return None
        return sample_meta

    def _slice_vision_tensor_by_meta(self, tensor, sample_meta, key, start_idx, end_idx):
        if tensor is None:
            return None
        ranges = []
        cursor = 0
        for i, meta in enumerate(sample_meta):
            count = int(meta.get("vision_counts", {}).get(key, 0))
            next_cursor = cursor + count
            if i >= start_idx and i < end_idx and count > 0:
                ranges.append((cursor, next_cursor))
            cursor = next_cursor
        if not ranges:
            return None
        if len(ranges) == 1:
            s, e = ranges[0]
            return tensor[s:e]
        return torch.cat([tensor[s:e] for s, e in ranges], dim=0)
    
    @torch.no_grad()
    def forward(self, inputs, past_key_values=None, return_hidden_states=False):
        """
        Run forward pass and return log probabilities for next token.
        Supports automatic batch splitting for memory management.
        
        Args:
            inputs: Prepared model inputs from prepare_input() or batched inputs
            past_key_values: Optional past key-value cache from previous forward pass
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Tuple of (log_probs, past_key_values, hidden_states) where:
            - log_probs: Tensor of shape (batch_size, vocab_size) with log probabilities for next token
            - past_key_values: Past key-value cache for next iteration
            - hidden_states: Hidden states (or None if not requested)
        """
        batch_size = inputs["input_ids"].shape[0]
        sample_meta = self._get_sample_meta(inputs, batch_size)
        
        # Check if batch splitting is needed.
        can_split = True
        if sample_meta is None:
            # Mixed multimodal batches without metadata are unsafe to shard.
            if "image_grid_thw" in inputs and inputs["image_grid_thw"].shape[0] != batch_size:
                can_split = False
            if "pixel_values" in inputs and "image_grid_thw" not in inputs:
                can_split = False
        if self.max_batch_size_per_forward is not None and batch_size > self.max_batch_size_per_forward and can_split:
            return self._forward_with_batch_splitting(inputs, past_key_values, return_hidden_states)

        # Standard forward pass without splitting
        model_inputs = {k: v for k, v in inputs.items() if not str(k).startswith("__")}
        # Safety: never pass empty vision tensors to model.
        for k in self._VISION_KEYS:
            if k in model_inputs and torch.is_tensor(model_inputs[k]) and model_inputs[k].numel() == 0:
                del model_inputs[k]
        if "pixel_values" in model_inputs and "image_grid_thw" in model_inputs:
            if model_inputs["image_grid_thw"].shape[0] == 0:
                del model_inputs["pixel_values"]
                del model_inputs["image_grid_thw"]
        if "pixel_values" in model_inputs and "image_sizes" in model_inputs:
            if model_inputs["image_sizes"].shape[0] == 0:
                del model_inputs["pixel_values"]
                del model_inputs["image_sizes"]

        outputs = self.model(**model_inputs, past_key_values=past_key_values, use_cache=True, output_hidden_states=return_hidden_states, return_dict=True)
        last_token_logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        
        # Convert to log probabilities
        log_probs = torch.log_softmax(last_token_logits, dim=-1)
        
        return log_probs, outputs.past_key_values, outputs.hidden_states
    
    @torch.no_grad()
    def _forward_with_batch_splitting(self, inputs, past_key_values=None, return_hidden_states=False):
        """
        Forward pass with batch splitting for memory efficiency.
        
        Args:
            inputs: Batched inputs to split
            past_key_values: Past KV cache to split
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Same as forward(), but with results concatenated from multiple sub-batches
        """
        batch_size = inputs["input_ids"].shape[0]
        max_batch = self.max_batch_size_per_forward
        sample_meta = self._get_sample_meta(inputs, batch_size)
        
        num_hidden_layers = _get_config_attr(self.model.config, "num_hidden_layers")
        if num_hidden_layers is None:
            raise AttributeError("Cannot get num_hidden_layers from model config (tried config and config.text_config)")
        
        all_log_probs = []
        all_past_key_values = []
        all_hidden_states = [] if return_hidden_states else None
        covered = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        
        # Process in chunks
        for start_idx in range(0, batch_size, max_batch):
            end_idx = min(start_idx + max_batch, batch_size)
            chunk_size = end_idx - start_idx
            
            # Split inputs
            chunk_inputs = {}
            for key, value in inputs.items():
                if str(key).startswith("__"):
                    continue
                # Slice all batch-major tensors, including position_ids for KV-cache incremental decoding.
                if key in ["input_ids", "attention_mask", "position_ids"] and value.shape[0] == batch_size:
                    chunk_inputs[key] = value[start_idx:end_idx]
                elif key in self._VISION_KEYS:
                    if sample_meta is None:
                        # Backward-compatible fallback.
                        if key == "pixel_values":
                            if "image_grid_thw" in inputs:
                                image_grid_thw = inputs["image_grid_thw"]
                                patches_per_sample = (
                                    image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
                                ).cpu().tolist()
                                start_patch_idx = sum(patches_per_sample[:start_idx])
                                end_patch_idx = sum(patches_per_sample[:end_idx])
                                chunk_inputs[key] = value[start_patch_idx:end_patch_idx, :]
                            else:
                                single_image_size = value.shape[0] // batch_size
                                chunk_inputs[key] = value[
                                    start_idx * single_image_size : end_idx * single_image_size, :
                                ]
                        elif value.shape[0] == batch_size:
                            chunk_inputs[key] = value[start_idx:end_idx]
                        else:
                            chunk_inputs[key] = value
                    else:
                        chunk_tensor = self._slice_vision_tensor_by_meta(
                            value, sample_meta, key, start_idx, end_idx
                        )
                        if chunk_tensor is not None:
                            chunk_inputs[key] = chunk_tensor
                elif isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
                    chunk_inputs[key] = value[start_idx:end_idx]
                else:
                    chunk_inputs[key] = value

            # Assertions/guardrails for mixed passage batches.
            if sample_meta is not None:
                has_any_image = any(bool(m.get("has_image", False)) for m in sample_meta[start_idx:end_idx])
                if not has_any_image:
                    for k in self._VISION_KEYS:
                        assert k not in chunk_inputs, f"Text-only chunk unexpectedly contains vision key {k}"
                if has_any_image and "pixel_values" in chunk_inputs:
                    assert chunk_inputs["pixel_values"].numel() > 0, "Image chunk has empty pixel_values"

            # Split past_key_values if present (use compat layer; cache __iter__ shape differs by API)
            chunk_past_kv = _make_dynamic_cache(num_hidden_layers)
            if past_key_values is not None:
                for i in range(cache_compat._cache_num_layers(past_key_values)):
                    k, v = cache_compat._cache_get_layer(past_key_values, i)
                    cache_compat._cache_append_layer(
                        chunk_past_kv, k[start_idx:end_idx], v[start_idx:end_idx]
                    )
            
            # Forward pass for this chunk
            outputs = self.model(
                **chunk_inputs,
                past_key_values=chunk_past_kv,
                use_cache=True,
                output_hidden_states=return_hidden_states,
            )

            # Get logits and convert to log probs
            last_token_logits = outputs.logits[:, -1, :]
            chunk_log_probs = torch.log_softmax(last_token_logits, dim=-1)
            assert chunk_log_probs.shape[0] == chunk_size, "Chunk logits batch size mismatch"
            all_log_probs.append((start_idx, end_idx, chunk_log_probs))
            covered[start_idx:end_idx] = True
            
            # Collect past_key_values
            all_past_key_values.append((start_idx, end_idx, outputs.past_key_values))
            
            # Collect hidden states if requested (only last layer to minimize VRAM)
            if return_hidden_states:
                all_hidden_states.append(outputs.hidden_states[-1])
        
        # Reassemble results in original sample order.
        vocab_size = all_log_probs[0][2].shape[1]
        log_probs = torch.empty((batch_size, vocab_size), device=self.device, dtype=all_log_probs[0][2].dtype)
        for s, e, t in all_log_probs:
            log_probs[s:e] = t
        assert log_probs.shape[0] == batch_size, "Concatenated logits size mismatch"
        assert bool(covered.all().item()), "Some samples were not covered during split forward"
        
        # Reassemble past_key_values in original sample order.
        concatenated_past_kv = _make_dynamic_cache(num_hidden_layers)
        num_layers = cache_compat._cache_num_layers(all_past_key_values[0][2])
        for i in range(num_layers):
            layer_kv = [cache_compat._cache_get_layer(c[2], i) for c in all_past_key_values]
            k0 = layer_kv[0][0]
            v0 = layer_kv[0][1]
            k_full = torch.empty((batch_size,) + tuple(k0.shape[1:]), device=k0.device, dtype=k0.dtype)
            v_full = torch.empty((batch_size,) + tuple(v0.shape[1:]), device=v0.device, dtype=v0.dtype)
            for (s, e, _), (k_chunk, v_chunk) in zip(all_past_key_values, layer_kv):
                k_full[s:e] = k_chunk
                v_full[s:e] = v_chunk
            cache_compat._cache_append_layer(
                concatenated_past_kv,
                k_full,
                v_full,
            )

        # Concatenate hidden states if requested (only last layer)
        concatenated_hidden_states = None
        if return_hidden_states:
            # Concatenate the last layer's hidden states (already extracted above)
            last_layer_hidden_states = torch.cat(all_hidden_states, dim=0)
            concatenated_hidden_states = (last_layer_hidden_states,)

        return log_probs, concatenated_past_kv, concatenated_hidden_states

    @torch.no_grad()
    def forward_sharded(self, inputs, past_key_values_sharded=None, return_hidden_states=False):
        """
        Forward pass for passage-parallel decoding.
        Unlike _forward_with_batch_splitting(), this path keeps chunk-local KV caches
        and never reassembles full-batch k/v tensors.
        """
        batch_size = inputs["input_ids"].shape[0]
        sample_meta = self._get_sample_meta(inputs, batch_size)
        can_split = True
        if sample_meta is None:
            if "image_grid_thw" in inputs and inputs["image_grid_thw"].shape[0] != batch_size:
                can_split = False
            if "pixel_values" in inputs and "image_grid_thw" not in inputs:
                can_split = False

        if self.max_batch_size_per_forward is None or batch_size <= self.max_batch_size_per_forward or not can_split:
            # Fall back to standard forward for small local shards.
            return self.forward(inputs, past_key_values=past_key_values_sharded, return_hidden_states=return_hidden_states)

        max_batch = self.max_batch_size_per_forward
        num_hidden_layers = _get_config_attr(self.model.config, "num_hidden_layers")
        if num_hidden_layers is None:
            raise AttributeError("Cannot get num_hidden_layers from model config (tried config and config.text_config)")

        all_log_probs = []
        all_hidden_states = [] if return_hidden_states else None
        next_past_key_values_sharded = []
        shard_idx = 0

        for start_idx in range(0, batch_size, max_batch):
            end_idx = min(start_idx + max_batch, batch_size)
            chunk_size = end_idx - start_idx

            chunk_inputs = {}
            for key, value in inputs.items():
                if str(key).startswith("__"):
                    continue
                if key in ["input_ids", "attention_mask", "position_ids"] and value.shape[0] == batch_size:
                    chunk_inputs[key] = value[start_idx:end_idx]
                elif key in self._VISION_KEYS:
                    if sample_meta is None:
                        if key == "pixel_values":
                            if "image_grid_thw" in inputs:
                                image_grid_thw = inputs["image_grid_thw"]
                                patches_per_sample = (
                                    image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
                                ).cpu().tolist()
                                start_patch_idx = sum(patches_per_sample[:start_idx])
                                end_patch_idx = sum(patches_per_sample[:end_idx])
                                chunk_inputs[key] = value[start_patch_idx:end_patch_idx, :]
                            else:
                                single_image_size = value.shape[0] // batch_size
                                chunk_inputs[key] = value[
                                    start_idx * single_image_size : end_idx * single_image_size, :
                                ]
                        elif value.shape[0] == batch_size:
                            chunk_inputs[key] = value[start_idx:end_idx]
                        else:
                            chunk_inputs[key] = value
                    else:
                        chunk_tensor = self._slice_vision_tensor_by_meta(
                            value, sample_meta, key, start_idx, end_idx
                        )
                        if chunk_tensor is not None:
                            chunk_inputs[key] = chunk_tensor
                elif isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
                    chunk_inputs[key] = value[start_idx:end_idx]
                else:
                    chunk_inputs[key] = value

            if sample_meta is not None:
                has_any_image = any(bool(m.get("has_image", False)) for m in sample_meta[start_idx:end_idx])
                if not has_any_image:
                    for k in self._VISION_KEYS:
                        assert k not in chunk_inputs, f"Text-only chunk unexpectedly contains vision key {k}"
                if has_any_image and "pixel_values" in chunk_inputs:
                    assert chunk_inputs["pixel_values"].numel() > 0, "Image chunk has empty pixel_values"

            if isinstance(past_key_values_sharded, list):
                if shard_idx < len(past_key_values_sharded):
                    chunk_past_kv = past_key_values_sharded[shard_idx]
                else:
                    chunk_past_kv = _make_dynamic_cache(num_hidden_layers)
            else:
                # Backward-compatible fallback if a single cache is passed.
                chunk_past_kv = _make_dynamic_cache(num_hidden_layers)
                if past_key_values_sharded is not None:
                    for i in range(cache_compat._cache_num_layers(past_key_values_sharded)):
                        k, v = cache_compat._cache_get_layer(past_key_values_sharded, i)
                        cache_compat._cache_append_layer(
                            chunk_past_kv, k[start_idx:end_idx], v[start_idx:end_idx]
                        )

            outputs = self.model(
                **chunk_inputs,
                past_key_values=chunk_past_kv,
                use_cache=True,
                output_hidden_states=return_hidden_states,
            )

            last_token_logits = outputs.logits[:, -1, :]
            chunk_log_probs = torch.log_softmax(last_token_logits, dim=-1)
            assert chunk_log_probs.shape[0] == chunk_size, "Chunk logits batch size mismatch"
            all_log_probs.append(chunk_log_probs)
            next_past_key_values_sharded.append(outputs.past_key_values)
            if return_hidden_states:
                all_hidden_states.append(outputs.hidden_states[-1])
            shard_idx += 1

        log_probs = torch.cat(all_log_probs, dim=0)
        hidden_states = None
        if return_hidden_states:
            hidden_states = (torch.cat(all_hidden_states, dim=0),)
        return log_probs, next_past_key_values_sharded, hidden_states
    
    @torch.no_grad()
    def forward_single(self, inputs, past_key_values=None):
        """
        Run forward pass for a single input (backward compatibility).
        
        Args:
            inputs: Prepared model inputs from prepare_input()
            past_key_values: Optional past key-value cache from previous forward pass
            
        Returns:
            Tuple of (log_probs, past_key_values) where:
            - log_probs: Tensor of shape (vocab_size,) with log probabilities for next token
            - past_key_values: Past key-value cache for next iteration
        """
        log_probs, past_key_values = self.forward(inputs, past_key_values)
        return log_probs[0], past_key_values  # Return single batch item
    
    def generate(self, inputs, max_new_tokens):
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True)
        return outputs.logits[0][-1]

    @torch.no_grad()
    def generate_standard(self, inputs, max_new_tokens, **kwargs):
        """
        Full autoregressive generation. Returns only the newly generated token ids (no prompt).
        Use with inputs from _get_prompt_only_inputs(x, zk) or prepare_input(x, [], zk).
        """
        pad_token_id = kwargs.pop("pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = _get_special_id(self.tokenizer, "pad_token_id")
        eos_token_id = kwargs.pop("eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = _get_special_id(self.tokenizer, "eos_token_id")
        input_length = inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
            **kwargs,
        )
        # Return only new tokens (strip prompt)
        return outputs.sequences[0][input_length:].cpu().tolist()

    def is_stop_token(self, token_idx):
        return token_idx in self.stop_token_list


            



    