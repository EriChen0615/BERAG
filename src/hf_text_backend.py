from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from .hf_backend import _get_special_id, _make_dynamic_cache
    from . import cache_compat
except ImportError:
    from hf_backend import _get_special_id, _make_dynamic_cache
    import cache_compat


class HFTextBackend:
    """
    Text-only HF backend for BERAG.

    Provides the minimal interface expected by BAPEInferenceEngine:
      - device
      - model
      - processor.tokenizer (for decoding and special ids)
      - prepare_input(x, generated_tokens, zk, past_key_values=None)
      - prepare_batched_input(x, generated_tokens, passages)
      - forward(inputs, past_key_values=None, return_hidden_states=False)
      - is_stop_token(token_idx)
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        max_batch_size_per_forward: int = None,
    ):
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Ensure pad/eos ids are set so batching and stopping work reliably.
        if self.tokenizer.pad_token_id is None:
            pad_id = _get_special_id(self.tokenizer, "pad_token_id")
            if pad_id is not None:
                self.tokenizer.pad_token_id = pad_id
        if self.tokenizer.eos_token_id is None:
            eos_id = _get_special_id(self.tokenizer, "eos_token_id")
            if eos_id is not None:
                self.tokenizer.eos_token_id = eos_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # Expose a processor-like object with .tokenizer so BERAG code that
        # references backend.processor.tokenizer continues to work.
        class _ProcessorWrapper:
            def __init__(self, tok):
                self.tokenizer = tok

        self.processor = _ProcessorWrapper(self.tokenizer)

        self.max_batch_size_per_forward = max_batch_size_per_forward
        if max_batch_size_per_forward is not None:
            print(f"[HFTextBackend] Max batch size per forward: {max_batch_size_per_forward}")

        tok = self.tokenizer
        self.stop_token_list = [
            _get_special_id(tok, "eos_token_id"),
            _get_special_id(tok, "pad_token_id"),
        ]

    @property
    def device(self):
        return self.model.device

    def is_stop_token(self, token_idx: int) -> bool:
        return token_idx in self.stop_token_list

    def _normalize_generated_tokens_to_ids(self, generated_tokens):
        if not generated_tokens:
            return []
        raw = getattr(generated_tokens, "ids", generated_tokens)
        ids = list(raw)
        if ids and isinstance(ids[0], int):
            return ids
        return [int(t) for t in ids]

    def _get_prompt_only_inputs(self, x, zk):
        """
        Build inputs for the prompt only (no assistant reply tokens).
        x: {"text": "...<<<EVIDENCE>>>..."}
        zk: passage text (str)
        """
        input_text = x["text"].replace("<<<EVIDENCE>>>", zk)
        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        return enc.to(device=self.device)

    def prepare_input(self, x, generated_tokens, zk, past_key_values=None):
        """
        x: {"text": ...}
        generated_tokens: list of token ids (continuation only, or full context when using cache)
        zk: passage text (str)
        past_key_values: if not None, incremental step: only the last token is used as input.
        """
        ids = self._normalize_generated_tokens_to_ids(generated_tokens)

        if past_key_values is not None:
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

        prompt_inputs = self._get_prompt_only_inputs(x, zk)
        input_ids = prompt_inputs["input_ids"]
        attention_mask = prompt_inputs["attention_mask"]

        if ids:
            continuation = torch.tensor([ids], device=self.device, dtype=torch.long)
            input_ids = torch.cat([input_ids, continuation], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(1, len(ids), device=self.device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )

        out = dict(prompt_inputs)
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        return out

    def prepare_batched_input(self, x, generated_tokens, passages):
        """
        Prepare batched inputs for multiple passages with proper left padding.
        x: {"text": ...}
        generated_tokens: list of token ids
        passages: list of passage texts (str)
        """
        individual_inputs = []
        for zk in passages:
            inputs = self.prepare_input(x, generated_tokens, zk)
            individual_inputs.append(inputs)

        max_seq_len = max(inp["input_ids"].shape[1] for inp in individual_inputs)

        batched_inputs = {}
        for key in individual_inputs[0].keys():
            if key in ["input_ids", "attention_mask"]:
                padded_tensors = []
                for inp in individual_inputs:
                    tensor = inp[key]
                    seq_len = tensor.shape[1]
                    if seq_len < max_seq_len:
                        if key == "input_ids":
                            pad_value = _get_special_id(self.tokenizer, "pad_token_id")
                        else:
                            pad_value = 0
                        padding = torch.full(
                            (tensor.shape[0], max_seq_len - seq_len),
                            pad_value,
                            device=tensor.device,
                            dtype=tensor.dtype,
                        )
                        padded_tensor = torch.cat([padding, tensor], dim=1)
                    else:
                        padded_tensor = tensor
                    padded_tensors.append(padded_tensor)
                batched_inputs[key] = torch.cat(padded_tensors, dim=0)
            else:
                batched_inputs[key] = torch.cat([inp[key] for inp in individual_inputs], dim=0)

        return batched_inputs

    def prepare_batched_incremental_input(self, new_token_ids: List[int], past_key_values) -> dict:
        """
        Prepare batched inputs for incremental decoding (one new token per sequence).
        new_token_ids: list of token ids, one per batch element.
        past_key_values: KV cache from previous forward.
        """
        batch_size = len(new_token_ids)
        try:
            k, _ = cache_compat._cache_get_layer(past_key_values, 0)
            past_length = k.shape[2]
        except Exception:
            past_length = 0
        inputs = {
            "input_ids": torch.tensor([new_token_ids], device=self.device, dtype=torch.long).T,
            "attention_mask": torch.ones(batch_size, past_length + 1, device=self.device, dtype=torch.long),
            "position_ids": torch.full((batch_size, 1), past_length, device=self.device, dtype=torch.long),
        }
        return inputs

    @torch.no_grad()
    def forward(self, inputs, past_key_values=None, return_hidden_states=False, return_full_logits=False):
        """
        Run forward pass and return log probabilities for next token.
        Mirrors HFQwen2VLBackend.forward but without vision-specific keys.
        If return_full_logits=True, returns (logits, past_key_values, hidden_states) where logits
        has shape [batch, seq_len, vocab_size] for extracting token-level log probs.
        """
        batch_size = inputs["input_ids"].shape[0]
        if self.max_batch_size_per_forward is not None and batch_size > self.max_batch_size_per_forward:
            return self._forward_with_batch_splitting(inputs, past_key_values, return_hidden_states, return_full_logits)

        outputs = self.model(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )
        if return_full_logits:
            return outputs.logits, outputs.past_key_values, outputs.hidden_states
        last_token_logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(last_token_logits, dim=-1)
        return log_probs, outputs.past_key_values, outputs.hidden_states

    @torch.no_grad()
    def _forward_with_batch_splitting(self, inputs, past_key_values=None, return_hidden_states=False, return_full_logits=False):
        """
        Forward pass with batch splitting for memory efficiency.
        """
        batch_size = inputs["input_ids"].shape[0]
        max_batch = self.max_batch_size_per_forward

        num_hidden_layers = _get_config_attr(self.model.config, "num_hidden_layers")
        if num_hidden_layers is None:
            raise AttributeError("Cannot get num_hidden_layers from model config (tried config and config.text_config)")

        all_log_probs = []
        all_logits = [] if return_full_logits else None
        all_past_key_values = []
        all_hidden_states = [] if return_hidden_states else None

        for start_idx in range(0, batch_size, max_batch):
            end_idx = min(start_idx + max_batch, batch_size)
            chunk_inputs = {
                "input_ids": inputs["input_ids"][start_idx:end_idx],
                "attention_mask": inputs["attention_mask"][start_idx:end_idx],
            }
            chunk_past_kv = _make_dynamic_cache(num_hidden_layers)
            if past_key_values is not None:
                for i in range(cache_compat._cache_num_layers(past_key_values)):
                    k, v = cache_compat._cache_get_layer(past_key_values, i)
                    cache_compat._cache_append_layer(
                        chunk_past_kv, k[start_idx:end_idx], v[start_idx:end_idx]
                    )

            outputs = self.model(
                **chunk_inputs,
                past_key_values=chunk_past_kv,
                use_cache=True,
                output_hidden_states=return_hidden_states,
            )
            if return_full_logits:
                all_logits.append(outputs.logits)
            else:
                last_token_logits = outputs.logits[:, -1, :]
                chunk_log_probs = torch.log_softmax(last_token_logits, dim=-1)
                all_log_probs.append(chunk_log_probs)
            all_past_key_values.append(outputs.past_key_values)
            if return_hidden_states:
                all_hidden_states.append(outputs.hidden_states[-1])

        if return_full_logits:
            logits = torch.cat(all_logits, dim=0)
            return logits, None, None

        log_probs = torch.cat(all_log_probs, dim=0)
        concatenated_past_kv = _make_dynamic_cache(num_hidden_layers)
        num_layers = cache_compat._cache_num_layers(all_past_key_values[0])
        for i in range(num_layers):
            layer_kv = [cache_compat._cache_get_layer(c, i) for c in all_past_key_values]
            cache_compat._cache_append_layer(
                concatenated_past_kv,
                torch.cat([p[0] for p in layer_kv], dim=0),
                torch.cat([p[1] for p in layer_kv], dim=0),
            )

        concatenated_hidden_states = None
        if return_hidden_states:
            last_layer_hidden_states = torch.cat(all_hidden_states, dim=0)
            concatenated_hidden_states = (last_layer_hidden_states,)

        return log_probs, concatenated_past_kv, concatenated_hidden_states

