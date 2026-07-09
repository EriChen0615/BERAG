import time
from typing import Any, Dict, List

import torch


class RegularRAGInferenceEngine:
    """
    Standard (non-BAPE) RAG decoding with a single concatenated evidence prompt.
    Uses transformer KV caching through backend.forward(..., past_key_values=...).
    """

    def __init__(self, backend):
        self.backend = backend
        self.device = backend.device

    @staticmethod
    def _build_generation_stats(
        prefill_ms: float,
        decode_ms: float,
        decode_tokens: int,
        input_tokens: int,
        output_tokens: int,
    ) -> Dict[str, Any]:
        decode_tokens_per_ms = (decode_tokens / decode_ms) if decode_ms > 0 else 0.0
        return {
            "prefill_ms": float(prefill_ms),
            "decode_ms": float(decode_ms),
            "decode_tokens": int(decode_tokens),
            "decode_tokens_per_ms": float(decode_tokens_per_ms),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
        }

    @staticmethod
    def _concat_evidence(passages: List[str]) -> str:
        if len(passages) == 0:
            return "[EVIDENCE] No passage provided.\n"
        return "".join(passages)

    @torch.no_grad()
    def generate(self, input_context, passages, max_new_tokens=512):
        merged_evidence = self._concat_evidence(passages)
        generated_tokens = []
        log_all_tokens_llk = []

        # Prefill pass (full prompt)
        prefill_inputs = self.backend.prepare_input(input_context, generated_tokens, merged_evidence)
        input_tokens = int(prefill_inputs["input_ids"].shape[1])

        prefill_start = time.perf_counter()
        log_probs, past_key_values, _ = self.backend.forward(prefill_inputs, past_key_values=None, return_hidden_states=False)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        decode_ms = 0.0

        token_idx = torch.argmax(log_probs[0]).item()
        log_all_tokens_llk.append(log_probs[0, token_idx].item())
        generated_tokens.append(token_idx)

        while len(generated_tokens) < max_new_tokens:
            if self.backend.is_stop_token(token_idx):
                break

            decode_inputs = self.backend.prepare_input(
                input_context,
                generated_tokens,
                merged_evidence,
                past_key_values=past_key_values,
            )
            decode_start = time.perf_counter()
            log_probs, past_key_values, _ = self.backend.forward(
                decode_inputs,
                past_key_values=past_key_values,
                return_hidden_states=False,
            )
            decode_ms += (time.perf_counter() - decode_start) * 1000.0

            token_idx = torch.argmax(log_probs[0]).item()
            log_all_tokens_llk.append(log_probs[0, token_idx].item())
            generated_tokens.append(token_idx)

        stats = self._build_generation_stats(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            decode_tokens=max(len(generated_tokens) - 1, 0),
            input_tokens=input_tokens,
            output_tokens=len(generated_tokens),
        )
        return generated_tokens, log_all_tokens_llk, stats
