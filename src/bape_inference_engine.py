import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import DynamicCache
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

try:
    from . import cache_compat
except ImportError:
    import cache_compat


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


@dataclass
class BeamState:
    """State tracking for a single beam in beam search."""
    generated_tokens: List[int] = field(default_factory=list)
    log_score: float = 0.0
    log_passage_conditioned_likelihood: torch.Tensor = None  # Shape: [K]
    log_passage_posterior: torch.Tensor = None  # Shape: [K]
    log_last_token_policy_probs: torch.Tensor = None  # Shape: [K], P(y_{i-1} | h_{i-1}, z_k, x)
    posterior_logits_over_steps: List = field(default_factory=list)  # Track posteriors at each step
    batch_start_idx: int = 0  # Start index in the batched forward pass
    batch_end_idx: int = 0  # End index in the batched forward pass
    num_passages: int = 0  # Number of active passages for this beam
    
class BAPEInferenceEngine:
    """
    Inference engine for Bayesian Adaptive Passage Ensemble (BAPE).
    Notations:
        * Z: passage set
        * z_k: the k^th passage retrieved/reranked and provided as context to LLM/VLM
        * y_i: output token at the i^th step
        * h_i: output token (history) up to the (i-1)^th step. I.e., y_{1:i-1}
        * x: other input context (propmts, image, etc.)
        * P_k(y_i | h_i, x) = P(y_i | z_k, h_i, x): the policy focused on the k^th passage
    Names:
        * Passage Posterior: P(z_k|h_i,x,Z)
        * Passage Prior: P(z_k|x,Z) 
        * Policy next token probability: P_k(y_i|h_i,x) = P(y_i|z_k,h_i,x)
        * Total next token probability: P(y_i|h_i,x,Z)
        * Passage-conditioned Likelihood: P(h_i|z_k,x)
    
    B-APE Decoding Objective: 
        argmax_{y_i} \sum_k P(y_i, z_k | h_i, x, Z). I.e., Marginalize over all passages z_k
        Expand to the following:
        P(y_i | h_i, x, Z) = \sum_{k=1}^K P_k(y_i | z_k, h_i, x) P(z_k | h_i, x, Z)

    The *passage posterior* P(z_k | h_i, x, Z) can be computed as via Baye's rule.:
        P(z_k | h_i, x, Z) = P(h_i|z_k,x)P(z_k|x,Z) / \sum_k' P(h_i|z_k',x)P(z_k'|x,Z)
        * Passage Prior P(z_k|x,Z) can be obtained from the retriever/reranker or set to uniform
        * Passage-conditioned Likelihood P(h_i|z_k,x) can be computed recursively from pervious total next token probability P(y_{i-1}|h_{i-1}, z_k, x) and Passage-conditioned Likelihood P(h_{i-1}|z_k, x)
            P(h_i|z_k, x) = P(y_{i-1}|h_{i-1}, z_k, x) P(h_{i-1}|z_k, x)
    """
    def __init__(self, backend, prior_head_path=None, prior_head_config=None, dynamic_k_top_p=None, hidden_state_offset=0, num_beams=0, debug=False):
        """
        * backend: the LLM/VLM backend which has the following APIs:
            `forward(model_input)`: returns the LOGARITHM policy next token probabilities given prompt. 
            `prepare_input(input_data)`: prepares the input_data = (h_i, x, z_k) into prompt acceptable by `step`
            `device`: returns the device
            `is_stop_token(token_idx)`: checks if token_idx is a stop token and hence generation should terminate
        * debug: if True, print detailed generation information
        """
        self.backend = backend # assumes that backend.step(prompt) returns the policy next token probabilities given prompt. 
        self.device = self.backend.device

        self.dynamic_k_top_p = dynamic_k_top_p
        self.hidden_state_offset = hidden_state_offset
        self.prior_head = None
        self.num_beams = num_beams
        self.debug = debug

        prior_head_config = prior_head_config or {}
        if prior_head_path is not None and prior_head_path != '':
            input_dim = _get_config_attr(self.backend.model.config, "hidden_size")
            if input_dim is None:
                raise AttributeError("Cannot get hidden_size from model config (tried config and config.text_config)")
            proj_dim = prior_head_config.get("proj_dim", 1024)
            num_layers = prior_head_config.get("num_layers", 2)
            modeling = prior_head_config.get("modeling", "mlp_head")
            print(f"[BAPE Inference Engine] Prior head modeling: {modeling}")
            if modeling == "mlp_head":
                layers = []
                for i in range(num_layers - 1):
                    layers.append(nn.Linear(input_dim, proj_dim))
                    layers.append(nn.ReLU())
                    input_dim = proj_dim
                layers.append(nn.Linear(input_dim, 1))
                self.prior_head = nn.Sequential(*layers)
            elif modeling == "linear_head":
                layers = []
                for i in range(num_layers - 1):
                    layers.append(nn.Linear(input_dim, proj_dim))
                    input_dim = proj_dim
                layers.append(nn.Linear(input_dim, 1))
                self.prior_head = nn.Sequential(*layers)
            else:
                raise ValueError(f"[BAPE Inference Engine] Invalid prior head modeling: {modeling}")
            print(f"[BAPE Inference Engine] Prior head number of layers: {num_layers}")
            print(f"[BAPE Inference Engine] Prior head projection dimension: {proj_dim}")
            print(f"[BAPE Inference Engine] Prior head parameters: {sum(p.numel() for p in self.prior_head.parameters())}")
            self.prior_head.load_state_dict(torch.load(prior_head_path))
            print(f"[BAPE Inference Engine] Prior head loaded from {prior_head_path}")
            print(f"[BAPE Inference Engine] Prior head hidden state offset: {self.hidden_state_offset}")
            self.prior_head.to(self.device, dtype=torch.bfloat16)
            self.prior_head.eval()
        else:
            print(f"[BAPE Inference Engine] No Prior head")

    def _sample(self, log_next_token_probs):
        # Greedy sampling: select the token with the maximum log probability
        token_idx = torch.argmax(log_next_token_probs).item()
        log_token_llk = log_next_token_probs[token_idx].item()
        return token_idx, log_token_llk

    def _cache_total_batch_size(self, past_key_values) -> Optional[int]:
        if past_key_values is None:
            return None
        if isinstance(past_key_values, list):
            total = 0
            for shard_cache in past_key_values:
                if shard_cache is None:
                    continue
                try:
                    k, _ = cache_compat._cache_get_layer(shard_cache, 0)
                    total += int(k.shape[0])
                except Exception:
                    continue
            return total
        try:
            k, _ = cache_compat._cache_get_layer(past_key_values, 0)
            return int(k.shape[0])
        except Exception:
            return None

    def _validate_cache_batch_size(self, past_key_values, expected_batch_size: int, where: str):
        actual_batch_size = self._cache_total_batch_size(past_key_values)
        if actual_batch_size is None or expected_batch_size is None:
            return
        if actual_batch_size != expected_batch_size:
            msg = (
                f"[BAPE Inference Engine] KV-cache batch mismatch at {where}: "
                f"expected={expected_batch_size}, actual={actual_batch_size}"
            )
            if self.debug:
                raise RuntimeError(msg)
            print(f"WARNING: {msg}")

    @staticmethod
    def _build_generation_stats(
        prefill_ms: float,
        decode_ms: float,
        decode_tokens: int,
        input_tokens: int,
        output_tokens: int,
        prefill_forward_ms: float = 0.0,
        prior_head_ms: float = 0.0,
    ) -> Dict[str, Any]:
        decode_tokens_per_ms = (decode_tokens / decode_ms) if decode_ms > 0 else 0.0
        return {
            "prefill_ms": float(prefill_ms),
            "prefill_forward_ms": float(prefill_forward_ms),
            "prior_head_ms": float(prior_head_ms),
            "decode_ms": float(decode_ms),
            "decode_tokens": int(decode_tokens),
            "decode_tokens_per_ms": float(decode_tokens_per_ms),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
        }

    def _set_incremental_step_inputs(self, batched_inputs, past_key_values, batch_size, token_ids):
        """
        Update batched_inputs for one-token incremental decoding with KV cache.
        Uses full-length attention mask (past_len + 1) and explicit position_ids.
        """
        input_dtype = batched_inputs["input_ids"].dtype
        attn_dtype = batched_inputs["attention_mask"].dtype
        if isinstance(token_ids, int):
            new_token_tensor = torch.full((batch_size, 1), token_ids, device=self.device, dtype=input_dtype)
        elif isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(1)
            new_token_tensor = token_ids.to(device=self.device, dtype=input_dtype)
        else:
            new_token_tensor = torch.tensor(token_ids, device=self.device, dtype=input_dtype).view(batch_size, 1)

        past_length = 0
        if past_key_values is not None:
            cache_probe = past_key_values
            if isinstance(past_key_values, list):
                cache_probe = past_key_values[0] if len(past_key_values) > 0 else None
            try:
                if cache_probe is not None:
                    k, _ = cache_compat._cache_get_layer(cache_probe, 0)
                    past_length = int(k.shape[2])
            except Exception:
                past_length = 0

        new_attention_tensor = torch.ones((batch_size, past_length + 1), device=self.device, dtype=attn_dtype)
        batched_inputs["input_ids"] = new_token_tensor
        batched_inputs["attention_mask"] = new_attention_tensor
        batched_inputs["position_ids"] = torch.full(
            (batch_size, 1), past_length, device=self.device, dtype=torch.long
        )
    
    def _step(self, log_all_policy_next_token_probs, log_previous_all_policy_next_token_probs, 
              log_previous_all_passage_conditioned_likelihood, log_passage_prior):
        """
        Compute the marginalized next token probabilities using BAPE.
        
        Args:
            log_all_policy_next_token_probs: [K, vocab_size] policy probs for each passage
            log_previous_all_policy_next_token_probs: [K] previous token probs for each passage
            log_previous_all_passage_conditioned_likelihood: [K] previous likelihoods
            log_passage_prior: [K] passage prior probabilities
            
        Returns:
            log_next_token_probs: [vocab_size] marginalized token probabilities
            log_passage_posterior: [K] updated passage posteriors
            log_all_passage_conditioned_likelihood: [K] updated likelihoods
        """
        # Update passage-conditioned likelihood: P(h_i|z_k,x) = P(y_{i-1}|h_{i-1},z_k,x) * P(h_{i-1}|z_k,x)
        log_all_passage_conditioned_likelihood = (
            log_previous_all_policy_next_token_probs + log_previous_all_passage_conditioned_likelihood
        )
        
        # Compute passage posterior: P(z_k|h_i,x,Z) via Bayes rule
        passage_posterior_num = log_all_passage_conditioned_likelihood + log_passage_prior
        log_passage_posterior = passage_posterior_num - torch.logsumexp(passage_posterior_num, dim=0)
        
        # Marginalize over passages: P(y_i|h_i,x,Z) = sum_k P(y_i|z_k,h_i,x) * P(z_k|h_i,x,Z)
        log_next_token_probs = log_all_policy_next_token_probs + log_passage_posterior[:, None]
        log_next_token_probs = torch.logsumexp(log_next_token_probs, dim=0)  # shape: [vocab_size]
        
        return log_next_token_probs, log_passage_posterior, log_all_passage_conditioned_likelihood
    
    @torch.no_grad()
    def _compute_passage_prior_with_head(self, hidden_states):
        last_hidden_states = hidden_states[-1] # Shape: [batch_size, seq_len, hidden_size]
        last_hidden_states = last_hidden_states[:, -1-self.hidden_state_offset, :]
        passage_prior_logits = self.prior_head(last_hidden_states).squeeze(-1)
        log_passage_prior = torch.log_softmax(passage_prior_logits, dim=-1)
        return log_passage_prior, passage_prior_logits

    def _distributed_is_ready(self):
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    @staticmethod
    def _compute_shard_range(total_size: int, world_size: int, rank: int) -> Tuple[int, int]:
        base = total_size // world_size
        rem = total_size % world_size
        start = rank * base + min(rank, rem)
        end = start + base + (1 if rank < rem else 0)
        return start, end

    def _dist_all_gather_1d(self, local_tensor: torch.Tensor) -> torch.Tensor:
        if not self._distributed_is_ready():
            return local_tensor
        local_tensor = local_tensor.contiguous()
        world_size = dist.get_world_size()
        device = local_tensor.device
        local_len_t = torch.tensor([int(local_tensor.shape[0])], device=device, dtype=torch.long)
        gathered_lens = [torch.zeros_like(local_len_t) for _ in range(world_size)]
        dist.all_gather(gathered_lens, local_len_t)
        lens = [int(t.item()) for t in gathered_lens]
        max_len = max(lens) if len(lens) > 0 else 0
        if local_tensor.shape[0] < max_len:
            pad = torch.zeros(max_len - local_tensor.shape[0], device=device, dtype=local_tensor.dtype)
            local_padded = torch.cat([local_tensor, pad], dim=0).contiguous()
        else:
            local_padded = local_tensor.contiguous()
        gathered = [torch.zeros_like(local_padded) for _ in range(world_size)]
        dist.all_gather(gathered, local_padded)
        out = []
        for t, ln in zip(gathered, lens):
            if ln > 0:
                out.append(t[:ln])
        if len(out) == 0:
            return local_tensor.new_empty((0,))
        return torch.cat(out, dim=0)

    def _dist_all_gather_rows(self, local_tensor: torch.Tensor) -> torch.Tensor:
        if not self._distributed_is_ready():
            return local_tensor
        local_tensor = local_tensor.contiguous()
        world_size = dist.get_world_size()
        device = local_tensor.device
        local_rows_t = torch.tensor([int(local_tensor.shape[0])], device=device, dtype=torch.long)
        gathered_rows = [torch.zeros_like(local_rows_t) for _ in range(world_size)]
        dist.all_gather(gathered_rows, local_rows_t)
        row_lens = [int(t.item()) for t in gathered_rows]
        max_rows = max(row_lens) if len(row_lens) > 0 else 0
        cols = local_tensor.shape[1]
        if local_tensor.shape[0] < max_rows:
            pad = torch.zeros((max_rows - local_tensor.shape[0], cols), device=device, dtype=local_tensor.dtype)
            local_padded = torch.cat([local_tensor, pad], dim=0).contiguous()
        else:
            local_padded = local_tensor.contiguous()
        gathered = [torch.zeros_like(local_padded) for _ in range(world_size)]
        dist.all_gather(gathered, local_padded)
        out = []
        for t, ln in zip(gathered, row_lens):
            if ln > 0:
                out.append(t[:ln, :])
        if len(out) == 0:
            return local_tensor.new_empty((0, cols))
        return torch.cat(out, dim=0)

    def generate_passage_parallel(
        self,
        input_context,
        passages,
        log_passage_prior=None,
        max_new_tokens=512,
        passage_top_p=None,
        return_stats=False,
    ):
        """
        Greedy BAPE decoding with passage-sharded multi-GPU execution.
        This path is opt-in and keeps the standard generate() unchanged.
        """
        if self.num_beams > 1:
            raise NotImplementedError("Passage-parallel v1 only supports greedy decoding (num_beams=1).")
        if not self._distributed_is_ready():
            return self.generate(
                input_context,
                passages,
                log_passage_prior,
                max_new_tokens=max_new_tokens,
                passage_top_p=passage_top_p,
                return_stats=return_stats,
            )

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        K = len(passages)
        if world_size > K:
            raise ValueError(f"world_size ({world_size}) cannot exceed number of passages K ({K}).")

        shard_start, shard_end = self._compute_shard_range(K, world_size, rank)
        local_passages = passages[shard_start:shard_end]
        local_K = len(local_passages)

        x = input_context
        generated_tokens = []
        log_all_tokens_llk = []
        prior_logits = None
        prior_max_idx = -1
        posterior_max_idx = -1

        local_prev_policy = torch.zeros(local_K, device=self.device)
        local_prev_likelihood = torch.zeros(local_K, device=self.device)

        local_inputs = self.backend.prepare_batched_input(x, generated_tokens, local_passages)
        local_past_key_values = None
        input_tokens = int(local_inputs["input_ids"].shape[1]) if local_K > 0 else 0
        prefill_ms = 0.0
        prefill_forward_ms = 0.0
        prior_head_ms = 0.0
        decode_ms = 0.0

        # First local forward (prompt pass)
        prefill_start = time.perf_counter()
        local_log_policy, local_past_key_values, local_hidden_states = self.backend.forward_sharded(
            local_inputs,
            past_key_values_sharded=local_past_key_values,
            return_hidden_states=(self.prior_head is not None),
        )
        prefill_forward_ms += (time.perf_counter() - prefill_start) * 1000.0

        if self.prior_head is not None:
            prior_head_start = time.perf_counter()
            local_last_hidden = local_hidden_states[-1]
            local_last_hidden = local_last_hidden[:, -1-self.hidden_state_offset, :]
            local_prior_logits = self.prior_head(local_last_hidden).squeeze(-1)
            global_prior_logits = self._dist_all_gather_1d(local_prior_logits)
            global_log_prior = torch.log_softmax(global_prior_logits, dim=0)
            prior_logits = global_prior_logits.detach().cpu().tolist()
            prior_max_idx = int(torch.argmax(global_log_prior).item())
            if "pixel_values" in local_inputs:
                del local_inputs["pixel_values"]
            if "image_grid_thw" in local_inputs:
                del local_inputs["image_grid_thw"]
            if "image_sizes" in local_inputs:
                del local_inputs["image_sizes"]
            prior_head_ms += (time.perf_counter() - prior_head_start) * 1000.0
        else:
            if log_passage_prior is None:
                global_log_prior = torch.log_softmax(torch.zeros(K, device=self.device), dim=0)
            elif isinstance(log_passage_prior, torch.Tensor):
                global_log_prior = log_passage_prior.to(self.device)
            else:
                global_log_prior = torch.tensor(log_passage_prior, device=self.device, dtype=torch.float32)
        prefill_ms = prefill_forward_ms + prior_head_ms

        for _ in range(max_new_tokens):
            global_log_policy = self._dist_all_gather_rows(local_log_policy)
            global_prev_policy = self._dist_all_gather_1d(local_prev_policy)
            global_prev_likelihood = self._dist_all_gather_1d(local_prev_likelihood)

            log_next_token_probs, global_log_posterior, global_likelihood = self._step(
                global_log_policy,
                global_prev_policy,
                global_prev_likelihood,
                global_log_prior,
            )

            # Sample on rank0 and broadcast token id.
            if rank == 0:
                token_idx, log_token_llk = self._sample(log_next_token_probs)
                token_t = torch.tensor([token_idx], device=self.device, dtype=torch.long)
                llk_t = torch.tensor([log_token_llk], device=self.device, dtype=torch.float32)
            else:
                token_t = torch.zeros(1, device=self.device, dtype=torch.long)
                llk_t = torch.zeros(1, device=self.device, dtype=torch.float32)
            dist.broadcast(token_t, src=0)
            dist.broadcast(llk_t, src=0)
            token_idx = int(token_t.item())
            log_token_llk = float(llk_t.item())

            generated_tokens.append(token_idx)
            log_all_tokens_llk.append(log_token_llk)
            posterior_max_idx = int(torch.argmax(global_log_posterior).item())

            local_prev_policy = global_log_policy[shard_start:shard_end, token_idx].contiguous()
            local_prev_likelihood = global_likelihood[shard_start:shard_end].contiguous()

            if self.backend.is_stop_token(token_idx):
                break

            self._set_incremental_step_inputs(local_inputs, local_past_key_values, local_K, token_idx)
            self._validate_cache_batch_size(
                local_past_key_values,
                expected_batch_size=local_K,
                where="generate_passage_parallel.incremental",
            )
            decode_start = time.perf_counter()
            local_log_policy, local_past_key_values, _ = self.backend.forward_sharded(
                local_inputs,
                past_key_values_sharded=local_past_key_values,
                return_hidden_states=False,
            )
            decode_ms += (time.perf_counter() - decode_start) * 1000.0

        stats = self._build_generation_stats(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            decode_tokens=max(len(generated_tokens) - 1, 0),
            input_tokens=input_tokens,
            output_tokens=len(generated_tokens),
            prefill_forward_ms=prefill_forward_ms,
            prior_head_ms=prior_head_ms,
        )
        if return_stats:
            return generated_tokens, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits, stats
        return generated_tokens, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits
    
    def _do_greedy(self, input_context, passages, log_passage_prior=None, max_new_tokens=512, passage_top_p=None, return_stats=False):
        passage_top_p = passage_top_p or self.dynamic_k_top_p
        K = len(passages)
        x = input_context

        # Initialize variables
        log_previous_all_policy_next_token_probs = torch.zeros(K, device=self.device)
        log_previous_all_passage_conditioned_likelihood = torch.zeros(K, device=self.device)

        generated_tokens = []
        log_all_tokens_llk = []
        prior_logits = []
        posterior_logits_over_steps = []

        # Initialize past key values cache
        num_hidden_layers = _get_config_attr(self.backend.model.config, "num_hidden_layers")
        if num_hidden_layers is None:
            raise AttributeError("Cannot get num_hidden_layers from model config (tried config and config.text_config)")
        past_key_values = _make_dynamic_cache(num_hidden_layers)

        # Prepare initial batched inputs
        batched_inputs = self.backend.prepare_batched_input(x, generated_tokens, passages)
        input_tokens = int(batched_inputs["input_ids"].shape[1]) if K > 0 else 0
        prefill_ms = 0.0
        prefill_forward_ms = 0.0
        prior_head_ms = 0.0
        decode_ms = 0.0

        # First forward pass
        prefill_start = time.perf_counter()
        log_all_policy_next_token_probs, past_key_values, hidden_states = self.backend.forward(batched_inputs, past_key_values, return_hidden_states=True)
        prefill_forward_ms += (time.perf_counter() - prefill_start) * 1000.0
        if self.prior_head is not None:
            prior_head_start = time.perf_counter()
            log_passage_prior, passage_prior_logits = self._compute_passage_prior_with_head(hidden_states)
            prior_logits = passage_prior_logits.detach().cpu().tolist()
            if "pixel_values" in batched_inputs:
                del batched_inputs["pixel_values"]  # already in kv cache after first pass
            if "image_grid_thw" in batched_inputs:
                del batched_inputs["image_grid_thw"]
            if "image_sizes" in batched_inputs:
                del batched_inputs["image_sizes"]
            prior_head_ms += (time.perf_counter() - prior_head_start) * 1000.0
        prefill_ms = prefill_forward_ms + prior_head_ms

        while len(generated_tokens) < max_new_tokens:
            log_next_token_probs, log_passage_posterior, log_all_passage_conditioned_likelihood = self._step(
                log_all_policy_next_token_probs, 
                log_previous_all_policy_next_token_probs,
                log_previous_all_passage_conditioned_likelihood,
                log_passage_prior
            )
            token_idx, log_token_llk = self._sample(log_next_token_probs)
            generated_tokens.append(token_idx)
            log_all_tokens_llk.append(log_token_llk)

            # Save computation for next iteration
            log_previous_all_policy_next_token_probs = log_all_policy_next_token_probs[:, token_idx]
            log_previous_all_passage_conditioned_likelihood = log_all_passage_conditioned_likelihood

            # Logging
            posterior_logits_over_steps.append(log_passage_posterior.detach().cpu().tolist())

            if self.backend.is_stop_token(token_idx):
                break

            # Incremental cached decoding: one token input + full-length attention mask.
            # update batched inputs by appending the new token
            # new_token_tensor = torch.full((K, 1), token_idx, device=self.device, dtype=batched_inputs["input_ids"].dtype)
            # new_attention_tensor = torch.ones((K, 1), device=self.device, dtype=batched_inputs["attention_mask"].dtype)
            # batched_inputs["input_ids"] = torch.cat([batched_inputs["input_ids"], new_token_tensor], dim=1)
            # batched_inputs["attention_mask"] = torch.cat([batched_inputs["attention_mask"], new_attention_tensor], dim=1)
            self._set_incremental_step_inputs(batched_inputs, past_key_values, K, token_idx)
            self._validate_cache_batch_size(past_key_values, expected_batch_size=K, where="_do_greedy.incremental")

            decode_start = time.perf_counter()
            log_all_policy_next_token_probs, past_key_values, hidden_states = self.backend.forward(batched_inputs, past_key_values, return_hidden_states=True)
            decode_ms += (time.perf_counter() - decode_start) * 1000.0

        stats = self._build_generation_stats(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            decode_tokens=max(len(generated_tokens) - 1, 0),
            input_tokens=input_tokens,
            output_tokens=len(generated_tokens),
            prefill_forward_ms=prefill_forward_ms,
            prior_head_ms=prior_head_ms,
        )
        if return_stats:
            return generated_tokens, log_all_tokens_llk, posterior_logits_over_steps, prior_logits, stats
        return generated_tokens, log_all_tokens_llk, posterior_logits_over_steps, prior_logits
    
    def _do_beam_search(self, input_context, passages, log_passage_prior=None, max_new_tokens=512, passage_top_p=None, return_n_sequences=1, return_stats=False):
        """
        Beam search with BAPE marginalization. All beams are processed in a single batched forward pass.
        
        Args:
            return_n_sequences: Number of sequences to return (default: 1)
        
        Returns:
            If return_n_sequences == 1:
                generated_tokens: List of token IDs for the best beam
                log_all_tokens_llk: List of log likelihoods for each token
                posterior_logits_over_steps: Posterior probabilities over time for the best beam
                prior_logits: Prior probabilities
            If return_n_sequences > 1:
                all_generated_tokens: List of lists of token IDs for top n beams
                all_log_tokens_llk: List of lists of log likelihoods for top n beams
                all_posterior_logits: List of posterior probabilities over time for top n beams
                prior_logits: Prior probabilities
        """
        passage_top_p = passage_top_p or self.dynamic_k_top_p
        K = len(passages)
        x = input_context
        num_beams = self.num_beams
        
        # Initialize single beam
        initial_beam = BeamState(
            generated_tokens=[],
            log_score=0.0,
            log_passage_conditioned_likelihood=torch.zeros(K, device=self.device),
            log_passage_posterior=torch.zeros(K, device=self.device),
            log_last_token_policy_probs=None,  # No previous token yet
            posterior_logits_over_steps=[],
            batch_start_idx=0,
            batch_end_idx=K,
            num_passages=K
        )
        beams = [initial_beam]
        
        prior_logits = []
        
        # Initialize past key values cache for all beams
        num_hidden_layers = _get_config_attr(self.backend.model.config, "num_hidden_layers")
        if num_hidden_layers is None:
            raise AttributeError("Cannot get num_hidden_layers from model config (tried config and config.text_config)")
        past_key_values = _make_dynamic_cache(num_hidden_layers)
        
        # Prepare initial batched inputs (only 1 beam × K passages = K)
        batched_inputs_list = []
        for beam in beams:
            beam_input = self.backend.prepare_batched_input(x, beam.generated_tokens, passages)
            batched_inputs_list.append(beam_input)
        
        # Concatenate all beam inputs into single batch
        batched_inputs = self._concatenate_batched_inputs(batched_inputs_list)
        input_tokens = int(batched_inputs["input_ids"].shape[1]) if batched_inputs["input_ids"].shape[0] > 0 else 0
        prefill_ms = 0.0
        prefill_forward_ms = 0.0
        prior_head_ms = 0.0
        decode_ms = 0.0
        
        # First forward pass to get initial probs and compute passage prior if needed
        prefill_start = time.perf_counter()
        log_all_policy_next_token_probs, past_key_values, hidden_states = self.backend.forward(
            batched_inputs, past_key_values, return_hidden_states=True
        )
        prefill_forward_ms += (time.perf_counter() - prefill_start) * 1000.0
        
        if self.prior_head is not None:
            prior_head_start = time.perf_counter()
            log_passage_prior, passage_prior_logits = self._compute_passage_prior_with_head(hidden_states)
            prior_logits = passage_prior_logits.detach().cpu().tolist()
            # Remove pixel values after first pass (already cached)
            if "pixel_values" in batched_inputs:
                del batched_inputs["pixel_values"]
            if "image_grid_thw" in batched_inputs:
                del batched_inputs["image_grid_thw"]
            if "image_sizes" in batched_inputs:
                del batched_inputs["image_sizes"]
            prior_head_ms += (time.perf_counter() - prior_head_start) * 1000.0
        prefill_ms = prefill_forward_ms + prior_head_ms
        
        # Initialize passage prior for all beams
        for beam in beams:
            beam.log_passage_posterior = log_passage_prior.clone() - torch.logsumexp(log_passage_prior, dim=0)
        
        # Track finished beams
        finished_beams = []
        early_stop = False  # Track if we stopped because we have enough beams
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"[BAPE Beam Search] Starting beam search with {num_beams} beams, K={K} passages")
            print(f"{'='*80}\n")
        
        # Main generation loop
        for step in range(max_new_tokens):
            if self.debug:
                print(f"\n--- Step {step} ---")
                print(f"Active beams: {len(beams)} | Finished beams: {len(finished_beams)}")
            
            all_candidates = []
            
            # Process each beam to generate candidates
            for beam_idx, beam in enumerate(beams):
                # Extract this beam's policy probs
                beam_log_policy_probs = log_all_policy_next_token_probs[beam.batch_start_idx:beam.batch_end_idx]
                
                # For the first step, use passage prior directly. For subsequent steps, compute BAPE update
                if len(beam.generated_tokens) == 0:
                    # First token generation: use passage prior as posterior
                    log_passage_posterior = log_passage_prior[:beam.num_passages] if beam.num_passages < K else log_passage_prior.clone()
                    log_passage_posterior = log_passage_posterior - torch.logsumexp(log_passage_posterior, dim=0)
                    
                    # Marginalize over passages
                    log_next_token_probs = beam_log_policy_probs + log_passage_posterior[:, None]
                    log_next_token_probs = torch.logsumexp(log_next_token_probs, dim=0)
                    
                    # Passage-conditioned likelihood starts at 1 (log = 0)
                    log_passage_conditioned_likelihood = torch.zeros(beam.num_passages, device=self.device)
                else:
                    # Subsequent tokens: compute BAPE update using saved previous policy probs
                    log_next_token_probs, log_passage_posterior, log_passage_conditioned_likelihood = self._step(
                        beam_log_policy_probs,
                        beam.log_last_token_policy_probs,
                        beam.log_passage_conditioned_likelihood,
                        log_passage_prior[:beam.num_passages] if beam.num_passages < K else log_passage_prior
                    )
                
                # Get top num_beams tokens for this beam
                k = min(num_beams, log_next_token_probs.shape[0])
                top_log_probs, top_indices = torch.topk(log_next_token_probs, k, dim=0)
                
                # Create candidate for each top token
                for token_idx, log_prob in zip(top_indices.tolist(), top_log_probs.tolist()):
                    # Compute log policy probs for this specific token across all passages
                    log_token_policy_probs = beam_log_policy_probs[:, token_idx].clone()
                    
                    # Append current posterior to the history
                    new_posterior_history = beam.posterior_logits_over_steps + [log_passage_posterior.detach().cpu().tolist()]
                    
                    all_candidates.append({
                        'parent_beam_idx': beam_idx,
                        'token': token_idx,
                        'log_score': beam.log_score + log_prob,
                        'log_passage_conditioned_likelihood': log_passage_conditioned_likelihood.clone(),
                        'log_passage_posterior': log_passage_posterior.clone(),
                        'log_last_token_policy_probs': log_token_policy_probs,
                        'posterior_logits_over_steps': new_posterior_history,
                        'generated_tokens': beam.generated_tokens + [token_idx],
                        'num_passages': beam.num_passages
                    })
            
            # Select top num_beams candidates globally
            all_candidates.sort(key=lambda x: x['log_score'], reverse=True)
            selected_candidates = all_candidates[:num_beams]
            
            if self.debug:
                print(f"\nGenerated {len(all_candidates)} total candidates, selected top {len(selected_candidates)}:")
                for i, cand in enumerate(selected_candidates):
                    token_str = self.backend.processor.tokenizer.decode([cand['token']])
                    is_stop = "🛑 STOP" if self.backend.is_stop_token(cand['token']) else ""
                    print(f"  [{i}] Token: {repr(token_str):20s} | Score: {cand['log_score']:.4f} | From beam: {cand['parent_beam_idx']} {is_stop}")
            
            # Separate finished and active candidates
            active_candidates = []
            for cand in selected_candidates:
                if self.backend.is_stop_token(cand['token']):
                    # This beam is finished - add to finished list with length normalization
                    finished_beams.append({
                        'generated_tokens': cand['generated_tokens'],
                        'log_score': cand['log_score'],
                        'normalized_score': cand['log_score'] / len(cand['generated_tokens']) if len(cand['generated_tokens']) > 0 else cand['log_score'],
                        'posterior_logits_over_steps': cand['posterior_logits_over_steps']
                    })
                else:
                    # This beam continues
                    active_candidates.append(cand)
            
            if self.debug:
                print(f"\n→ Finished: {len(selected_candidates) - len(active_candidates)} beams | Continuing: {len(active_candidates)} beams")
            
            # Check termination conditions
            # 1. We have enough finished beams
            if len(finished_beams) >= num_beams:
                if self.debug:
                    print(f"\n✓ Terminating: {len(finished_beams)} beams finished (>= {num_beams} required)")
                early_stop = True
                break
            
            # 2. No active beams left (all finished)
            if len(active_candidates) == 0:
                if self.debug:
                    print(f"\n✓ Terminating: No active beams remaining")
                early_stop = True
                break
            
            # Reorganize KV caches based on beam ancestry (only for active beams)
            past_key_values = self._reorganize_kv_caches_for_beams(
                past_key_values, beams, active_candidates
            )
            
            # Create new beams from active candidates only
            new_beams = []
            batch_offset = 0
            for cand in active_candidates:
                new_beam = BeamState(
                    generated_tokens=cand['generated_tokens'],
                    log_score=cand['log_score'],
                    log_passage_conditioned_likelihood=cand['log_passage_conditioned_likelihood'],
                    log_passage_posterior=cand['log_passage_posterior'],
                    log_last_token_policy_probs=cand['log_last_token_policy_probs'],
                    posterior_logits_over_steps=cand['posterior_logits_over_steps'],
                    batch_start_idx=batch_offset,
                    batch_end_idx=batch_offset + cand['num_passages'],
                    num_passages=cand['num_passages']
                )
                new_beams.append(new_beam)
                batch_offset += cand['num_passages']
            
            beams = new_beams
            
            # Show current state of active beams
            if self.debug:
                print(f"\nActive beams for next step:")
                for i, beam in enumerate(beams):
                    decoded = self.backend.processor.tokenizer.decode(beam.generated_tokens)
                    print(f"  Beam {i}: Score={beam.log_score:.4f} | Tokens={len(beam.generated_tokens):3d} | Text: {repr(decoded[:80])}")
            
            # Prepare batched inputs for next step
            # All beams get the same token appended (their respective tokens)
            total_batch_size = sum(beam.num_passages for beam in beams)
            new_token_list = []
            for beam in beams:
                new_token_list.extend([beam.generated_tokens[-1]] * beam.num_passages)
            
            new_token_tensor = torch.tensor(new_token_list, device=self.device, dtype=torch.long).unsqueeze(1)

            # Subsequent steps: one-token cached decoding with full-length mask/position.
            self._set_incremental_step_inputs(
                batched_inputs, past_key_values, total_batch_size, new_token_tensor
            )
            self._validate_cache_batch_size(
                past_key_values,
                expected_batch_size=total_batch_size,
                where="_do_beam_search.incremental",
            )
            
            # Forward pass for all beams
            decode_start = time.perf_counter()
            log_all_policy_next_token_probs, past_key_values, _ = self.backend.forward(
                batched_inputs, past_key_values
            )
            decode_ms += (time.perf_counter() - decode_start) * 1000.0
        
        # Loop finished - add any remaining active beams to finished_beams ONLY if we hit max_new_tokens
        # Don't add them if we stopped early because we already have enough finished beams
        if not early_stop:
            if self.debug:
                print(f"\n⚠️ Max tokens reached. Adding {len(beams)} remaining active beams as finished.")
            for beam in beams:
                finished_beams.append({
                    'generated_tokens': beam.generated_tokens,
                    'log_score': beam.log_score,
                    'normalized_score': beam.log_score / len(beam.generated_tokens) if len(beam.generated_tokens) > 0 else beam.log_score,
                    'posterior_logits_over_steps': beam.posterior_logits_over_steps
                })
        
        # Select best beam(s) from finished beams (using length-normalized score)
        if len(finished_beams) == 0:
            # Edge case: no beams finished (shouldn't happen, but handle it)
            if self.debug:
                print(f"\n⚠️ WARNING: No beams finished!")
            empty_stats = self._build_generation_stats(0.0, 0.0, 0, input_tokens, 0)
            if return_n_sequences == 1:
                if return_stats:
                    return [], [], [], prior_logits if prior_logits else [], empty_stats
                return [], [], [], prior_logits if prior_logits else []
            else:
                if return_stats:
                    return [[]], [[]], [[]], prior_logits if prior_logits else [], empty_stats
                return [[]], [[]], [[]], prior_logits if prior_logits else []
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"[BAPE Beam Search] Generation complete! Total finished beams: {len(finished_beams)}")
            print(f"\nAll finished beams (sorted by normalized score):")
        
        sorted_finished = sorted(finished_beams, key=lambda b: b['normalized_score'], reverse=True)
        
        # Limit to requested number of sequences
        n_to_return = min(return_n_sequences, len(sorted_finished))
        
        if self.debug:
            for i, beam in enumerate(sorted_finished[:n_to_return]):
                decoded = self.backend.processor.tokenizer.decode(beam['generated_tokens'])
                print(f"  [{i}] Score: {beam['log_score']:.4f} | Norm: {beam['normalized_score']:.4f} | Length: {len(beam['generated_tokens'])}")
                print(f"      Text: {repr(decoded[:100])}")
        
        # Return single sequence (backwards compatible)
        if return_n_sequences == 1:
            best_finished = sorted_finished[0]
            best_beam_tokens = best_finished['generated_tokens']
            log_all_tokens_llk = [best_finished['normalized_score']] * len(best_beam_tokens)
            posterior_logits_over_steps = best_finished['posterior_logits_over_steps']
            
            if self.debug:
                print(f"\n✓ Returning best beam with normalized score: {best_finished['normalized_score']:.4f}")
                print(f"✓ Posterior tracked over {len(posterior_logits_over_steps)} steps")
                print(f"{'='*80}\n")
            
            stats = self._build_generation_stats(
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                decode_tokens=max(len(best_beam_tokens) - 1, 0),
                input_tokens=input_tokens,
                output_tokens=len(best_beam_tokens),
                prefill_forward_ms=prefill_forward_ms,
                prior_head_ms=prior_head_ms,
            )
            if return_stats:
                return best_beam_tokens, log_all_tokens_llk, posterior_logits_over_steps, prior_logits, stats
            return best_beam_tokens, log_all_tokens_llk, posterior_logits_over_steps, prior_logits
        
        # Return multiple sequences
        else:
            all_generated_tokens = []
            all_log_tokens_llk = []
            all_posterior_logits = []
            
            for beam in sorted_finished[:n_to_return]:
                all_generated_tokens.append(beam['generated_tokens'])
                # Store beam-level scores: [log_score, normalized_score]
                # Note: beam search doesn't track per-token likelihoods, only cumulative scores
                all_log_tokens_llk.append(beam['log_score'])
                all_posterior_logits.append(beam['posterior_logits_over_steps'])
            
            if self.debug:
                print(f"\n✓ Returning top {n_to_return} beams")
                print(f"✓ Best normalized score: {sorted_finished[0]['normalized_score']:.4f}")
                print(f"{'='*80}\n")
            
            stats = self._build_generation_stats(
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                decode_tokens=max(len(all_generated_tokens[0]) - 1, 0) if len(all_generated_tokens) > 0 else 0,
                input_tokens=input_tokens,
                output_tokens=len(all_generated_tokens[0]) if len(all_generated_tokens) > 0 else 0,
                prefill_forward_ms=prefill_forward_ms,
                prior_head_ms=prior_head_ms,
            )
            if return_stats:
                return all_generated_tokens, all_log_tokens_llk, all_posterior_logits, prior_logits, stats
            return all_generated_tokens, all_log_tokens_llk, all_posterior_logits, prior_logits
    

    def generate_v2(self, input_context, passages, log_passage_prior=None, max_new_tokens=512, passage_top_p=None, return_n_sequences=1, return_stats=False):
        if self.num_beams > 1:
            return self._do_beam_search(
                input_context,
                passages,
                log_passage_prior,
                max_new_tokens,
                passage_top_p,
                return_n_sequences,
                return_stats=return_stats,
            )
        else:
            return self._do_greedy(
                input_context,
                passages,
                log_passage_prior,
                max_new_tokens,
                passage_top_p,
                return_stats=return_stats,
            )
    
    def generate(
        self,
        input_context,
        passages,
        log_passage_prior=None,
        max_new_tokens=512,
        passage_top_p=None,
        return_stats=False,
        return_posterior_over_steps=False,
    ):
        passage_top_p = passage_top_p or self.dynamic_k_top_p
        
        K = len(passages)
        x = input_context

        new_tokens = 0
        log_previous_all_policy_next_token_probs = torch.zeros(K, device=self.device)
        log_previous_all_passage_conditioned_likelihood = torch.zeros(K, device=self.device)
        
        generated_tokens = [] # i.e, history h_i
        log_all_tokens_llk = []
        log_passage_posterior = None 
        prior_logits = None  # Store prior logits for return
        posterior_over_steps = []
        
        # Initialize past key values cache
        num_hidden_layers = _get_config_attr(self.backend.model.config, "num_hidden_layers")
        if num_hidden_layers is None:
            raise AttributeError("Cannot get num_hidden_layers from model config (tried config and config.text_config)")
        past_key_values = _make_dynamic_cache(num_hidden_layers)
        
        # Prepare initial batched inputs
        batched_inputs = self.backend.prepare_batched_input(x, generated_tokens, passages)
        input_tokens = int(batched_inputs["input_ids"].shape[1]) if K > 0 else 0
        prefill_ms = 0.0
        prefill_forward_ms = 0.0
        prior_head_ms = 0.0
        decode_ms = 0.0
        active_indices, inactive_indices = torch.arange(K, device=self.device), None
        active_original_indices = torch.arange(K, device=self.device)
        previous_K = K

        prior_max_idx = -1
        posterior_max_idx = -1
        
        while new_tokens < max_new_tokens:
            # Compute policy next token probability using batched forward pass
            active_K = active_indices.shape[0]
            if active_K != previous_K: # resize/prune variables.
                previous_K = active_K
                past_key_values = self._keep_past_key_values(past_key_values, active_indices)
                self._validate_cache_batch_size(
                    past_key_values,
                    expected_batch_size=active_K,
                    where="generate.top_p_prune",
                )
                batched_inputs = self._keep_batched_inputs(batched_inputs, active_indices)
                log_previous_all_policy_next_token_probs = log_previous_all_policy_next_token_probs[active_indices].contiguous()
                log_previous_all_passage_conditioned_likelihood = log_previous_all_passage_conditioned_likelihood[active_indices].contiguous()
                log_passage_prior = log_passage_prior[active_indices].contiguous()
                active_original_indices = active_original_indices[active_indices].contiguous()


            if new_tokens == 0 and self.prior_head is not None:
                prefill_forward_start = time.perf_counter()
                log_all_policy_next_token_probs, past_key_values, hidden_states = self.backend.forward(batched_inputs, past_key_values, return_hidden_states=True)
                prefill_forward_ms += (time.perf_counter() - prefill_forward_start) * 1000.0
                prior_head_start = time.perf_counter()
                log_passage_prior, passage_prior_logits = self._compute_passage_prior_with_head(hidden_states)
                prior_max_idx = torch.argmax(log_passage_prior).item()
                # Save prior logits for return
                prior_logits = passage_prior_logits.detach().cpu().tolist()
                if "pixel_values" in batched_inputs:
                    del batched_inputs["pixel_values"]  # already in kv cache after first pass
                if "image_grid_thw" in batched_inputs:
                    del batched_inputs["image_grid_thw"]
                if "image_sizes" in batched_inputs:
                    del batched_inputs["image_sizes"]
                prior_head_ms += (time.perf_counter() - prior_head_start) * 1000.0
                prefill_ms = prefill_forward_ms + prior_head_ms
            else:
                if new_tokens == 0:
                    prefill_forward_start = time.perf_counter()
                else:
                    decode_start = time.perf_counter()
                log_all_policy_next_token_probs, past_key_values, _ = self.backend.forward(batched_inputs, past_key_values)
                elapsed_ms = (time.perf_counter() - (prefill_forward_start if new_tokens == 0 else decode_start)) * 1000.0
                if new_tokens == 0:
                    prefill_forward_ms += elapsed_ms
                    prefill_ms = prefill_forward_ms + prior_head_ms
                else:
                    decode_ms += elapsed_ms
            # log_all_policy_next_token_probs shape: (K, vocab_size)



            log_all_passage_conditioned_likelihood = log_previous_all_policy_next_token_probs + log_previous_all_passage_conditioned_likelihood

            passage_posterior_num = log_all_passage_conditioned_likelihood + log_passage_prior

            log_passage_posterior = passage_posterior_num - torch.logsumexp(passage_posterior_num, dim=0)


            log_next_token_probs = log_all_policy_next_token_probs + log_passage_posterior[:, None]
            log_next_token_probs = torch.logsumexp(log_next_token_probs, dim=0) # shape (vocab_size)

            token_idx, log_token_llk = self._sample(log_next_token_probs)
            generated_tokens.append(token_idx)
            log_all_tokens_llk.append(log_token_llk)
            new_tokens += 1

            # Save computation for next iteration
            log_previous_all_policy_next_token_probs = log_all_policy_next_token_probs[:, token_idx] # shape (active_K, )
            log_previous_all_passage_conditioned_likelihood = log_all_passage_conditioned_likelihood

            # Incremental cached decoding: one token input + full-length attention mask.
            self._set_incremental_step_inputs(
                batched_inputs, past_key_values, active_K, token_idx
            )
            self._validate_cache_batch_size(
                past_key_values,
                expected_batch_size=active_K,
                where="generate.incremental",
            )

            # Note: No need to re-pad since we're just appending one token to all sequences
            # All sequences will have the same length after appending

            posterior_max_idx = int(active_original_indices[torch.argmax(log_passage_posterior)].item())
            if return_posterior_over_steps:
                full_log_passage_posterior = torch.full(
                    (K,),
                    float("-inf"),
                    device=log_passage_posterior.device,
                    dtype=log_passage_posterior.dtype,
                )
                full_log_passage_posterior[active_original_indices] = log_passage_posterior
                posterior_over_steps.append(full_log_passage_posterior.detach().cpu().tolist())

            if self.backend.is_stop_token(token_idx):
                break

            if passage_top_p is not None and passage_top_p > 0 and passage_top_p < 1.0:
                all_passage_posterior = log_passage_posterior.exp()
                active_indices = self._top_p_filter(all_passage_posterior, passage_top_p)
                # print(f"[BAPE Inference Engine] Current Time Step: {new_tokens}")
                # print(f"[BAPE Inference Engine] Active passages: {active_indices}")
                # print(f"[BAPE Inference Engine] Passage Posterior Probabilities: {all_passage_posterior}")
                # print(f"[BAPE Inference Engine] Log All Passage Conditioned Likelihood: {log_all_passage_conditioned_likelihood}")
                # print(f"[BAPE Inference Engine] Generated tokens: {self.backend.processor.tokenizer.decode(generated_tokens)}")

        
        if isinstance(log_passage_prior, torch.Tensor):
            log_passage_prior = log_passage_prior.detach().cpu().tolist()
        if isinstance(log_passage_posterior, torch.Tensor):
            log_passage_posterior = log_passage_posterior.detach().cpu().tolist()

        stats = self._build_generation_stats(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            decode_tokens=max(len(generated_tokens) - 1, 0),
            input_tokens=input_tokens,
            output_tokens=len(generated_tokens),
            prefill_forward_ms=prefill_forward_ms,
            prior_head_ms=prior_head_ms,
        )
        if return_posterior_over_steps:
            if return_stats:
                return generated_tokens, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits, posterior_over_steps, stats
            return generated_tokens, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits, posterior_over_steps
        if return_stats:
            return generated_tokens, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits, stats
        return generated_tokens, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits
    
    def _top_p_filter(self, passage_posterior_probs, passage_top_p):
        """
        Apply Top-P filtering to select active passages.
        
        Args:
            log_passage_posterior: Log posterior probabilities for passages (K,)
            passage_top_p: Threshold for cumulative probability (e.g., 0.9)
            
        Returns:
            Tuple of (active_indices, inactive_indices)
        """
        K = passage_posterior_probs.shape[0]
        passage_probs = passage_posterior_probs
        
        # Sort passages by probability (descending)
        sorted_indices = torch.argsort(passage_probs, descending=True)
        sorted_probs = passage_probs[sorted_indices]
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        
        # Find minimal set where cumulative probability >= passage_top_p
        # This is the CORRECT Top-P definition
        keep_mask = cumulative_probs < passage_top_p

        if keep_mask.all():
            return torch.arange(K, device=self.device)

        first_exceed_idx = (~keep_mask).nonzero(as_tuple=True)[0][0].item()
        num_to_keep = first_exceed_idx + 1
        
        # Handle edge case: if no passage reaches the threshold, keep at least the top one
        active_indices = sorted_indices[:num_to_keep]
        return active_indices
    
    def _keep_batched_inputs(self, batched_inputs, active_indices):
        """
        Keep only batched inputs for active passages.
        Properly handles vision inputs and sample metadata for mixed modality batches.
        """
        pruned_inputs = {}
        batch_size = batched_inputs["input_ids"].shape[0]
        active_indices_list = active_indices.cpu().tolist()
        sample_meta = batched_inputs.get("__sample_meta__", None)
        pruned_sample_meta = None
        if isinstance(sample_meta, list) and len(sample_meta) == batch_size:
            pruned_sample_meta = [sample_meta[i] for i in active_indices_list]
        
        # Check if we have vision inputs
        vision_keys = ["pixel_values", "image_grid_thw", "image_sizes"]
        has_any_vision = any(k in batched_inputs for k in vision_keys)
        if has_any_vision and pruned_sample_meta is not None:
            for k in vision_keys:
                if k not in batched_inputs:
                    continue
                value = batched_inputs[k]
                if not isinstance(value, torch.Tensor):
                    continue
                # Repack vision tensors according to per-sample counts in metadata.
                old_counts = [int(m.get("vision_counts", {}).get(k, 0)) for m in sample_meta]
                assert sum(old_counts) == int(value.shape[0]), f"Vision metadata mismatch for {k}"
                offsets = []
                cursor = 0
                for c in old_counts:
                    offsets.append((cursor, cursor + c))
                    cursor += c
                kept_chunks = []
                for idx in active_indices_list:
                    s, e = offsets[idx]
                    if e > s:
                        kept_chunks.append(value[s:e])
                if kept_chunks:
                    pruned_inputs[k] = torch.cat(kept_chunks, dim=0).contiguous()
                    expected_rows = sum(int(m.get("vision_counts", {}).get(k, 0)) for m in pruned_sample_meta)
                    assert pruned_inputs[k].shape[0] == expected_rows, f"Pruned vision rows mismatch for {k}"
        
        # Handle all other inputs
        for key, value in batched_inputs.items():
            if key in ["pixel_values", "image_grid_thw", "image_sizes", "__sample_meta__"]:
                # Already handled above
                continue
            elif isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
                # Batched tensor - prune it
                pruned_inputs[key] = value[active_indices].contiguous()
            else:
                # Non-batched or non-tensor - keep as-is
                pruned_inputs[key] = value
        if pruned_sample_meta is not None:
            pruned_inputs["__sample_meta__"] = pruned_sample_meta
        
        return pruned_inputs
    
    def _keep_past_key_values(self, past_key_values, active_indices):
        """
        Keep only the past key values for active passages.
        Optimized for speed using tuple comprehension and minimal checks.
        """
        if past_key_values is None:
            return None
        for i in range(cache_compat._cache_num_layers(past_key_values)):
            k, v = cache_compat._cache_get_layer(past_key_values, i)
            cache_compat._cache_set_layer(
                past_key_values, i, k[active_indices], v[active_indices]
            )
        return past_key_values
    
    def _concatenate_batched_inputs(self, batched_inputs_list):
        """
        Concatenate multiple batched inputs into a single batch.
        Handles vision inputs (pixel_values, image_grid_thw) correctly.
        
        Args:
            batched_inputs_list: List of batched input dictionaries
            
        Returns:
            Single batched input dictionary with all beams concatenated
        """
        if len(batched_inputs_list) == 1:
            return batched_inputs_list[0]
        
        concatenated = {}
        
        # Get all keys from first input
        keys = batched_inputs_list[0].keys()
        
        for key in keys:
            if key == "__sample_meta__":
                merged_meta = []
                for inp in batched_inputs_list:
                    if isinstance(inp.get("__sample_meta__", None), list):
                        merged_meta.extend(inp["__sample_meta__"])
                concatenated[key] = merged_meta
                continue
            values = [inp[key] for inp in batched_inputs_list]
            
            if key == "pixel_values":
                # Concatenate pixel values along batch dimension
                concatenated[key] = torch.cat(values, dim=0)
            elif key == "image_grid_thw":
                # Concatenate image grid info
                concatenated[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], torch.Tensor):
                # Regular tensor concatenation along batch dimension
                concatenated[key] = torch.cat(values, dim=0)
            else:
                # Non-tensor values (should be same across all beams)
                concatenated[key] = values[0]
        
        return concatenated
    
    def _reorganize_kv_caches_for_beams(self, past_key_values, old_beams, selected_candidates):
        """
        Reorganize KV caches based on beam ancestry.
        
        Args:
            past_key_values: Current KV cache (DynamicCache)
            old_beams: List of old BeamState objects
            selected_candidates: List of selected candidate dictionaries with 'parent_beam_idx'
            
        Returns:
            Reorganized KV cache
        """
        if past_key_values is None:
            return None
        
        # Build index mapping: new position -> old position
        new_to_old_indices = []
        
        for candidate in selected_candidates:
            parent_idx = candidate['parent_beam_idx']
            parent_beam = old_beams[parent_idx]
            
            # Copy all K passages from parent beam
            parent_indices = list(range(parent_beam.batch_start_idx, parent_beam.batch_end_idx))
            new_to_old_indices.extend(parent_indices)
        
        # Convert to tensor for efficient indexing
        indices_tensor = torch.tensor(new_to_old_indices, device=self.device, dtype=torch.long)
        
        # Reorganize KV cache
        for i in range(cache_compat._cache_num_layers(past_key_values)):
            k, v = cache_compat._cache_get_layer(past_key_values, i)
            cache_compat._cache_set_layer(
                past_key_values,
                i,
                k[indices_tensor].contiguous(),
                v[indices_tensor].contiguous(),
            )

        return past_key_values