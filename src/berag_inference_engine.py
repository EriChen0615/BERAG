import torch
import torch.nn as nn
from transformers import DynamicCache
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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


@dataclass
class BERAGSGBeamState:
    """State for one beam in BERAG-SG segment-level beam search."""
    h: List[int] = field(default_factory=list)
    q: Optional[torch.Tensor] = None  # log posterior [K]
    alpha: float = 0.0
    terminated: bool = False
    last_segment_tokens: List[int] = field(default_factory=list)


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
    def __init__(
        self,
        backend,
        prior_head_path=None,
        prior_head_config=None,
        dynamic_k_top_p=None,
        hidden_state_offset=0,
        num_beams=0,
        debug=True,
        segment_length: int = 4,
        berag_sg_beam_size: int = 2,
        segment_gen_batch_size: int = 4,
        max_composite_size: Optional[int] = 1,
        log_dir: Optional[str] = None,
        berag_sg_top_p: Optional[float] = 0.9,
        berag_sg_temperature: float = 0.5,
        berag_sg_beam_prune: str = "diverse_beam_search",
        berag_sg_beam_score_mode: str = "marginal",
    ):
        """
        * backend: the LLM/VLM backend which has the following APIs:
            `forward(model_input)`: returns the LOGARITHM policy next token probabilities given prompt. 
            `prepare_input(input_data)`: prepares the input_data = (h_i, x, z_k) into prompt acceptable by `step`
            `device`: returns the device
            `is_stop_token(token_idx)`: checks if token_idx is a stop token and hence generation should terminate
        * debug: if True, print detailed generation information
        * segment_length, berag_sg_beam_size, segment_gen_batch_size, max_composite_size, log_dir, berag_sg_top_p, berag_sg_temperature, berag_sg_beam_prune, berag_sg_beam_score_mode: for BERAG-SG generate_berag_sg.
        """
        self.backend = backend  # assumes that backend.step(prompt) returns the policy next token probabilities given prompt. 
        self.device = self.backend.device

        self.dynamic_k_top_p = dynamic_k_top_p
        self.hidden_state_offset = hidden_state_offset
        self.prior_head = None
        self.num_beams = num_beams
        self.debug = debug
        self.segment_length = segment_length
        self.berag_sg_beam_size = berag_sg_beam_size
        self.segment_gen_batch_size = segment_gen_batch_size
        self.max_composite_size = max_composite_size
        self.log_dir = log_dir
        self.berag_sg_top_p = berag_sg_top_p
        self.berag_sg_temperature = berag_sg_temperature
        self.berag_sg_beam_prune = berag_sg_beam_prune
        self.berag_sg_beam_score_mode = berag_sg_beam_score_mode

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

    @torch.no_grad()
    def _generate_segment_under_chunk(
        self,
        x: dict,
        h: List[int],
        z_k: Union[str, dict],
        segment_length: int,
        temperature: float = 0.5,
    ) -> Tuple[List[int], float]:
        """
        Generate a segment of up to segment_length tokens conditioned on (h, x, z_k).
        Uses multinomial sampling with temperature (default 0.5); temperature <= 0 means greedy.
        Stops early on EOS. Returns (segment_tokens, log_likelihood_under_k).
        """
        segment_tokens: List[int] = []
        log_lik = 0.0
        context = list(h)
        pkv = None
        for _ in range(segment_length):
            inputs = self.backend.prepare_input(x, context, z_k, past_key_values=pkv)
            log_probs, pkv, _ = self.backend.forward(inputs, pkv)
            log_p_next = log_probs[0]  # shape (vocab_size,)
            if temperature is not None and temperature > 0:
                probs = torch.softmax(log_p_next / temperature, dim=0)
                token_idx = torch.multinomial(probs, num_samples=1).item()
            else:
                token_idx = torch.argmax(log_p_next).item()
            log_p = log_p_next[token_idx].item()
            log_lik += log_p
            segment_tokens.append(token_idx)
            context.append(token_idx)
            if self.backend.is_stop_token(token_idx):
                break
        return segment_tokens, log_lik

    @torch.no_grad()
    def _generate_segment_under_chunks_batched(
        self,
        x: dict,
        h: List[int],
        passages: List[Union[str, dict]],
        chunk_indices: List[int],
        segment_length: int,
        temperature: float = 0.5,
        batch_size: int = 4,
    ) -> List[Tuple[List[int], float]]:
        """
        Generate segments for multiple chunks in batches. Returns list of (segment_tokens, log_lik)
        for each chunk in chunk_indices.
        """
        prepare_incremental = getattr(
            self.backend, "prepare_batched_incremental_input", None
        )
        if prepare_incremental is None:
            return [
                self._generate_segment_under_chunk(x, h, passages[k], segment_length, temperature)
                for k in chunk_indices
            ]

        results: List[Tuple[List[int], float]] = []
        for start in range(0, len(chunk_indices), batch_size):
            end = min(start + batch_size, len(chunk_indices))
            indices_batch = chunk_indices[start:end]
            passages_batch = [passages[k] for k in indices_batch]
            batch_sz = len(passages_batch)

            batched_inputs = self.backend.prepare_batched_input(x, list(h), passages_batch)
            pkv = None
            done = [False] * batch_sz
            segment_tokens_list: List[List[int]] = [[] for _ in range(batch_sz)]
            log_lik_list = [0.0] * batch_sz

            for _ in range(segment_length):
                if all(done):
                    break
                log_probs, pkv, _ = self.backend.forward(batched_inputs, pkv)
                new_token_ids = []
                for i in range(batch_sz):
                    if done[i]:
                        new_token_ids.append(segment_tokens_list[i][-1])
                        continue
                    log_p_next = log_probs[i]
                    if temperature is not None and temperature > 0:
                        probs = torch.softmax(log_p_next / temperature, dim=0)
                        token_idx = torch.multinomial(probs, num_samples=1).item()
                    else:
                        token_idx = torch.argmax(log_p_next).item()
                    log_p = log_p_next[token_idx].item()
                    log_lik_list[i] += log_p
                    segment_tokens_list[i].append(token_idx)
                    new_token_ids.append(token_idx)
                    if self.backend.is_stop_token(token_idx):
                        done[i] = True

                batched_inputs = prepare_incremental(new_token_ids, pkv)

            results.extend(
                (segment_tokens_list[i], log_lik_list[i]) for i in range(batch_sz)
            )
        return results

    @torch.no_grad()
    def _compute_segment_log_likelihood_all_chunks(
        self,
        x: dict,
        h: List[int],
        segment_tokens: List[int],
        passages: List[Union[str, dict]],
    ) -> torch.Tensor:
        """
        log P_theta(segment | h, x, z_{k'}) for each chunk k'.
        Returns 1D tensor of shape (len(passages),).
        Single forward pass: run model on full sequence [prompt|z_k|h|segment] and extract
        log P(t_i | prefix, t_1..t_{i-1}) from logits at the appropriate positions.
        """
        K = len(passages)
        if K == 0:
            return torch.tensor([], device=self.device, dtype=torch.float32)
        if not segment_tokens:
            return torch.zeros(K, device=self.device, dtype=torch.float32)

        full_context = list(h) + list(segment_tokens)
        batched_inputs = self.backend.prepare_batched_input(x, full_context, passages)
        logits, _, _ = self.backend.forward(batched_inputs, None, return_full_logits=True)

        max_len = logits.shape[1]
        m = len(segment_tokens)
        # With left padding: logits at position (max_len - m - 1 + i) predict segment_tokens[i]
        start_idx = max_len - m - 1
        end_idx = max_len - 1
        logits_at_positions = logits[:, start_idx:end_idx, :]  # [K, m, vocab]
        log_probs = torch.log_softmax(logits_at_positions.float(), dim=-1)

        segment_tensor = torch.tensor(segment_tokens, device=self.device, dtype=torch.long)
        log_lik = log_probs[:, torch.arange(m, device=self.device), segment_tensor].sum(dim=1)
        return log_lik.to(torch.float32)

    def _berag_sg_top_p_chunk_indices(self, log_q: torch.Tensor, top_p: float) -> List[int]:
        """
        Nucleus (TopP) over chunks: return the smallest set of chunk indices (by descending prob)
        whose cumulative probability >= top_p. Chunks below the nucleus are treated as P=0 (skipped).
        """
        K = log_q.numel()
        if K == 0:
            return []
        probs = torch.softmax(log_q, dim=0)
        sorted_probs, sort_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum >= top_p
        if mask.any():
            j = (mask.nonzero(as_tuple=True)[0][0].item()) + 1
        else:
            j = K
        j = max(1, min(j, K))
        active = sort_idx[:j].cpu().tolist()
        return active

    def _berag_sg_beam_search_one_round(
        self,
        x: dict,
        passages: List[Union[str, dict]],
        chunk_indices: List[Tuple[int, ...]],
        log_q0: torch.Tensor,
        B: int,
        m: int,
        max_new_tokens: int,
        round_idx: int = 0,
        log_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        results_stem: Optional[str] = None,
        top_p: Optional[float] = None,
        temperature: float = 0.5,
        beam_prune: str = "diverse_beam_search",
        beam_score_mode: str = "marginal",
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        One round of BERAG-SG beam search. Returns (beams, beam_scores_alpha, beam_weighted_posterior, all_finished_beams).
        beams: list of dicts with h, q, alpha, terminated, last_segment_tokens (final B survivors).
        all_finished_beams: all beams that ever had terminated=True in this round (including those pruned out).
        """
        K = len(passages)
        device = self.device
        if self.debug:
            top_p_str = f", top_p={top_p}" if top_p is not None else ""
            print(f"[BERAG-SG round {round_idx}] One-round beam search: K={K}, B={B}, m={m}, max_new_tokens={max_new_tokens}{top_p_str}")
        # Initial beam: single (h=[], q=q0, alpha=0); expand to up to B beams after first step
        log_q0 = log_q0.to(device)
        beams: List[Dict[str, Any]] = [
            {
                "h": [],
                "q": log_q0.clone(),
                "alpha": 0.0,
                "terminated": False,
                "last_segment_tokens": [],
            }
        ]
        step = 0
        step_beams_for_log: List[List[Dict[str, Any]]] = []
        all_finished_beams: List[Dict[str, Any]] = []
        seen_h: set = set()
        early_stop_extra_rounds_remaining = 0

        while True:
            total_len = max(len(b["h"]) for b in beams)
            if total_len >= max_new_tokens:
                if self.debug:
                    print(f"[BERAG-SG round {round_idx}] Step {step}: stopping (total_len={total_len} >= max_new_tokens={max_new_tokens})")
                break
            children: List[Dict[str, Any]] = []
            active_beams = [b for b in beams if not b["terminated"]]
            for beam_idx, beam in enumerate(beams):
                if beam["terminated"]:
                    continue
                h_j = beam["h"]
                q_j = beam["q"]
                alpha_j = beam["alpha"]
                chunk_indices_to_expand = (
                    self._berag_sg_top_p_chunk_indices(q_j, top_p) if top_p is not None else list(range(K))
                )
                if self.debug:
                    print(f"[BERAG-SG round {round_idx}] Step {step}: beam {beam_idx} expanding {len(chunk_indices_to_expand)}/{K} chunks (TopP nucleus, batch_size={self.segment_gen_batch_size})...", flush=True)
                batch_results = self._generate_segment_under_chunks_batched(
                    x, h_j, passages, chunk_indices_to_expand, m,
                    temperature=temperature, batch_size=self.segment_gen_batch_size,
                )
                for i, k in enumerate(chunk_indices_to_expand):
                    segment_tokens, _ = batch_results[i]
                    if not segment_tokens:
                        continue
                    if self.debug:
                        print(f"    [round {round_idx} step {step}] beam {beam_idx} chunk {k}/{K} (gen segment)...", flush=False)
                        print(f"    segment_tokens: {tokenizer.decode(segment_tokens, skip_special_tokens=True)}", flush=False)
                    ell = self._compute_segment_log_likelihood_all_chunks(x, h_j, segment_tokens, passages)
                    log_marginal = torch.logsumexp(q_j + ell, dim=0)
                    if beam_score_mode == "marginal":
                        alpha_increment = log_marginal.item()
                    elif beam_score_mode == "proposal_chunk":
                        alpha_increment = ell[k].item()
                    else:
                        raise ValueError(f"Unsupported BERAG-SG beam score mode: {beam_score_mode}")
                    alpha_prime = alpha_j + alpha_increment
                    log_q_prime = q_j + ell - torch.logsumexp(q_j + ell, dim=0)
                    terminated = self.backend.is_stop_token(segment_tokens[-1])
                    children.append({
                        "h": h_j + segment_tokens,
                        "q": log_q_prime.clone(),
                        "alpha": alpha_prime,
                        "terminated": terminated,
                        "last_segment_tokens": segment_tokens,
                        "chunk_idx": k,
                    })
                    if self.debug:
                        print(f"    [round {round_idx} step {step}] beam {beam_idx} chunk {k}/{K} done (alpha'={alpha_prime:.3f}, len_h'={len(h_j)+len(segment_tokens)})", flush=False)
                if self.debug and top_p is not None and len(chunk_indices_to_expand) < K:
                    print(f"    [round {round_idx} step {step}] beam {beam_idx}: skipped {K - len(chunk_indices_to_expand)} low-prob chunks (TopP={top_p})", flush=False)
            if not children:
                if self.debug:
                    print(f"[BERAG-SG round {round_idx}] Step {step}: no children (all beams terminated), stopping")
                break
            # Accumulate early-finished beams before pruning (so pruned-out finished beams are still reported)
            for c in children:
                if c["terminated"]:
                    h_key = tuple(c["h"])
                    if h_key not in seen_h:
                        seen_h.add(h_key)
                        all_finished_beams.append(c)
            if beam_prune == "diverse_beam_search":
                # Diverse beam: best-from-each-group by conditioning chunk, then round-robin until B
                groups = defaultdict(list)
                for c in children:
                    k = c.get("chunk_idx", -1)
                    groups[k].append(c)
                for k in groups:
                    groups[k].sort(key=lambda c: c["alpha"], reverse=True)
                group_keys = sorted(
                    groups.keys(),
                    key=lambda k: groups[k][0]["alpha"] if groups[k] else -float("inf"),
                    reverse=True,
                )
                top_b_list: List[Dict[str, Any]] = []
                idx = 0
                while len(top_b_list) < B:
                    added = False
                    for gk in group_keys:
                        if len(top_b_list) >= B:
                            break
                        if idx < len(groups[gk]):
                            top_b_list.append(groups[gk][idx])
                            added = True
                    if not added:
                        break
                    idx += 1
                beams = top_b_list
            else:
                children.sort(key=lambda c: c["alpha"], reverse=True)
                beams = children[:B]
            step += 1
            if self.debug:
                best_alpha = beams[0]["alpha"]
                n_term = sum(1 for b in beams if b["terminated"])
                print(f"[BERAG-SG round {round_idx}] Step {step}: {len(children)} children -> top B={B}, best_alpha={best_alpha:.4f}, len_h={len(beams[0]['h'])}, terminated={n_term}/{B}", flush=True)
                tokenizer = getattr(self.backend, "tokenizer", None)
                for bi, b in enumerate(beams):
                    h = b["h"]
                    seg = b.get("last_segment_tokens", [])
                    full_text = ""
                    seg_text = ""
                    if tokenizer:
                        try:
                            full_text = tokenizer.decode(h, skip_special_tokens=True)
                            if seg:
                                seg_text = tokenizer.decode(seg, skip_special_tokens=True)
                        except Exception:
                            pass
                    print(f"    Surviving beam {bi}: alpha={b['alpha']:.4f} len_h={len(h)} terminated={b['terminated']}", flush=True)
                    print(f"      last_segment: {seg_text!r}", flush=True)
                    max_text_len = 800
                    if len(full_text) > max_text_len:
                        print(f"      text:\n{full_text[:max_text_len]}... (truncated, total {len(full_text)} chars)", flush=True)
                    else:
                        print(f"      text:\n{full_text}", flush=True)
            step_beams_for_log.append(self._berag_sg_beam_snapshot(beams, chunk_indices))

            terminated_beams = [b for b in beams if b["terminated"]]
            unfinished_beams = [b for b in beams if not b["terminated"]]
            just_triggered_early_stop = False
            if terminated_beams and unfinished_beams and early_stop_extra_rounds_remaining == 0:
                best_finished_alpha = max(b["alpha"] for b in terminated_beams)
                best_unfinished_alpha = max(b["alpha"] for b in unfinished_beams)
                if best_finished_alpha > best_unfinished_alpha:
                    early_stop_extra_rounds_remaining = 1
                    just_triggered_early_stop = True
                    if self.debug:
                        print(
                            f"[BERAG-SG round {round_idx}] Early-stop trigger: best finished beam ({best_finished_alpha:.4f}) "
                            f"outranks all unfinished beams ({best_unfinished_alpha:.4f}); allowing one more expansion round",
                            flush=True,
                        )

            if early_stop_extra_rounds_remaining > 0 and not just_triggered_early_stop:
                early_stop_extra_rounds_remaining -= 1
                finished_beams = [b for b in beams if b["terminated"]]
                if finished_beams:
                    beams = finished_beams
                if self.debug:
                    print(
                        f"[BERAG-SG round {round_idx}] Early stop: dropping {len(unfinished_beams)} unfinished surviving beams "
                        f"after the final extra expansion round",
                        flush=True,
                    )
                break

            if all(b["terminated"] for b in beams):
                if self.debug:
                    print(f"[BERAG-SG round {round_idx}] All beams terminated after step {step}")
                break

        step_beams_for_log.append(self._berag_sg_beam_snapshot(beams, chunk_indices))
        if log_dir or results_dir:
            self._write_berag_sg_beam_log(round_idx, step_beams_for_log, chunk_indices, log_dir, results_dir, results_stem)

        alphas = torch.tensor([b["alpha"] for b in beams], device=device, dtype=torch.float32)
        log_omega = alphas - torch.logsumexp(alphas, dim=0)
        omega = torch.exp(log_omega)
        beam_weighted_posterior = sum(omega[b_idx].item() * torch.exp(beams[b_idx]["q"]) for b_idx in range(len(beams)))
        if self.debug:
            k_star_r = int(torch.argmax(beam_weighted_posterior).item())
            print(f"[BERAG-SG round {round_idx}] Done: {step} steps, best_alpha={alphas.max().item():.4f}, k_star={k_star_r}, all_finished={len(all_finished_beams)}")
        return beams, alphas, beam_weighted_posterior, all_finished_beams

    def _berag_sg_beam_snapshot(
        self, beams: List[Dict[str, Any]], chunk_indices: List[Tuple[int, ...]]
    ) -> List[Dict[str, Any]]:
        """Serializable snapshot of beams for logging."""
        out = []
        for i, b in enumerate(beams):
            q = b["q"]
            q_list = q.cpu().tolist() if isinstance(q, torch.Tensor) else list(q)
            out.append({
                "beam_idx": i,
                "alpha": b["alpha"],
                "q": q_list,
                "len_h": len(b["h"]),
                "last_segment_tokens": b.get("last_segment_tokens", []),
                "terminated": b["terminated"],
                "chunk_indices": list(chunk_indices),
            })
        return out

    def _write_berag_sg_beam_log(
        self,
        round_idx: int,
        step_beams_for_log: List[List[Dict[str, Any]]],
        chunk_indices: List[Tuple[int, ...]],
        log_dir: Optional[str],
        results_dir: Optional[str],
        results_stem: Optional[str],
    ) -> None:
        """Write per-step surviving beams to JSONL."""
        import json
        import os
        path = None
        if results_dir and results_stem:
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{results_stem}_berag_sg_beams.jsonl")
            if round_idx == 0:
                with open(path, "w", encoding="utf-8"):
                    pass
        elif log_dir:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, "berag_sg_beams.jsonl")
        if path is None:
            return
        tokenizer = getattr(self.backend, "processor", None) and getattr(self.backend.processor, "tokenizer", None)
        with open(path, "a", encoding="utf-8") as f:
            for step, beams_snap in enumerate(step_beams_for_log):
                for rec in beams_snap:
                    rec_copy = dict(rec)
                    rec_copy["round"] = round_idx
                    rec_copy["step"] = step
                    if tokenizer and rec_copy.get("last_segment_tokens"):
                        try:
                            rec_copy["last_segment_text"] = tokenizer.decode(rec_copy["last_segment_tokens"], skip_special_tokens=True)
                        except Exception:
                            rec_copy["last_segment_text"] = ""
                    f.write(json.dumps(rec_copy) + "\n")

    def _write_berag_sg_posterior_log(
        self,
        round_idx: int,
        chunk_indices: List[Tuple[int, ...]],
        posterior: torch.Tensor,
        k_star: int,
        best_score: float,
        log_dir: Optional[str],
        results_dir: Optional[str],
        results_stem: Optional[str],
    ) -> None:
        """Write final posterior for this round to JSON (append to JSONL or single file)."""
        import json
        import os
        path = None
        if results_dir and results_stem:
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{results_stem}_berag_sg_posterior.jsonl")
        elif log_dir:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, "berag_sg_posterior.jsonl")
        if path is None:
            return
        post_list = posterior.cpu().tolist() if isinstance(posterior, torch.Tensor) else list(posterior)
        rec = {
            "round": round_idx,
            "chunk_indices": [list(ci) for ci in chunk_indices],
            "posterior": post_list,
            "k_star": k_star,
            "best_score": best_score,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _write_berag_sg_composite_states_log(
        self,
        round_idx: int,
        chunk_indices: List[Tuple[int, ...]],
        log_dir: Optional[str],
        results_dir: Optional[str],
        results_stem: Optional[str],
    ) -> None:
        """Write composite state descriptors (chunk index tuples) for this round."""
        import json
        import os
        path = None
        if results_dir and results_stem:
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{results_stem}_berag_sg_composite_states.jsonl")
        elif log_dir:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, "berag_sg_composite_states.jsonl")
        if path is None:
            return
        rec = {"round": round_idx, "chunk_indices": [list(ci) for ci in chunk_indices]}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _form_composite_chunks(
        self,
        passages: List[Union[str, dict]],
        chunk_indices: List[Tuple[int, ...]],
        k_star: int,
        max_composite_size: Optional[int] = None,
    ) -> Tuple[List[Union[str, dict]], List[Tuple[int, ...]]]:
        """
        Form composite chunks z'_k = z_{k_star} ⊕ z_k for k != k_star.
        chunk_indices[i] gives the tuple of original chunk indices for passages[i].
        Returns (new_passages, new_chunk_indices). Composites with len(indices) > max_composite_size are skipped.
        """
        K = len(passages)
        new_passages: List[Union[str, dict]] = []
        new_chunk_indices: List[Tuple[int, ...]] = []
        z_star = passages[k_star]
        indices_star = set(chunk_indices[k_star])
        for k in range(K):
            if k == k_star:
                continue
            union = tuple(sorted(indices_star | set(chunk_indices[k])))
            if max_composite_size is not None and len(union) > max_composite_size:
                continue
            z_k = passages[k]
            if isinstance(z_star, str) and isinstance(z_k, str):
                new_passages.append(z_star + "\n" + z_k)
                new_chunk_indices.append(union)
            elif isinstance(z_star, dict) and isinstance(z_k, dict):
                text_s = z_star.get("text", "")
                text_k = z_k.get("text", "")
                new_text = (text_s + "\n" + text_k) if text_s and text_k else (text_s or text_k)
                images = list(z_star.get("images", [])) + list(z_k.get("images", []))
                new_passages.append({"text": new_text, "images": images} if images else {"text": new_text})
                new_chunk_indices.append(union)
            else:
                new_passages.append((z_star if isinstance(z_star, str) else z_star.get("text", "")) + "\n" + (z_k if isinstance(z_k, str) else z_k.get("text", "")))
                new_chunk_indices.append(union)
        return new_passages, new_chunk_indices

    def generate_berag_sg(
        self,
        input_context: dict,
        passages: List[Union[str, dict]],
        log_passage_prior: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        segment_length: Optional[int] = None,
        berag_sg_beam_size: Optional[int] = None,
        max_composite_size: Optional[int] = None,
        log_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        results_stem: Optional[str] = None,
        berag_sg_top_p: Optional[float] = None,
        berag_sg_temperature: Optional[float] = None,
        berag_sg_beam_prune: Optional[str] = None,
        berag_sg_beam_score_mode: Optional[str] = None,
    ) -> Tuple[List[int], torch.Tensor, List[Tuple[int, ...]]]:
        """
        BERAG-SG: segment-level beam search with marginalized chunk scoring and multi-round composite states.
        Returns (generated_tokens_best, final_posterior, final_chunk_indices).
        """
        m = segment_length if segment_length is not None else self.segment_length
        B = berag_sg_beam_size if berag_sg_beam_size is not None else self.berag_sg_beam_size
        max_comp = max_composite_size if max_composite_size is not None else self.max_composite_size
        log_dir_use = log_dir if log_dir is not None else self.log_dir
        top_p = berag_sg_top_p if berag_sg_top_p is not None else self.berag_sg_top_p
        temperature = berag_sg_temperature if berag_sg_temperature is not None else self.berag_sg_temperature
        beam_prune = berag_sg_beam_prune if berag_sg_beam_prune is not None else self.berag_sg_beam_prune
        beam_score_mode = (
            berag_sg_beam_score_mode
            if berag_sg_beam_score_mode is not None
            else self.berag_sg_beam_score_mode
        )
        self._last_completed_beams = []

        x = input_context
        K = len(passages)
        device = self.device

        if self.debug:
            top_p_str = f", top_p={top_p}" if top_p is not None else ""
            print(f"[BERAG-SG] generate_berag_sg: K={K}, m={m}, B={B}, max_composite_size={max_comp}, max_new_tokens={max_new_tokens}, temperature={temperature}{top_p_str}, beam_score_mode={beam_score_mode}")
            # Confirm engine sees correct instruction and passages
            instruction = x.get("text", "")
            print(f"[BERAG-SG] instruction len={len(instruction)}")
            print(f"[BERAG-SG] instruction has <<<EVIDENCE>>>: {'<<<EVIDENCE>>>' in instruction}")
            print(f"[BERAG-SG] instruction has <|im_end|>: {'<|im_end|>' in instruction}")
            print(f"[BERAG-SG] instruction has <|im_start|>assistant: {'<|im_start|>assistant' in instruction}")
            print(f"[BERAG-SG] instruction (first 400 chars):\n{instruction[:400]}{'...' if len(instruction) > 400 else ''}")
            print(f"[BERAG-SG] instruction (last 300 chars):\n...{instruction[-300:]}")
            if passages:
                p0 = passages[0] if isinstance(passages[0], str) else str(passages[0])
                print(f"[BERAG-SG] passage[0] len={len(p0)}, preview: {p0[:150]}{'...' if len(p0) > 150 else ''}")
                # Show full prompt with first passage substituted (what model actually sees)
                full_prompt = instruction.replace("<<<EVIDENCE>>>", p0)
                print(f"[BERAG-SG] full prompt with passage[0] (last 400 chars, where model generates):\n...{full_prompt[-400:]}")

        if K == 0:
            return [], torch.tensor([], device=device), []

        if log_passage_prior is not None:
            log_q0 = log_passage_prior.to(device)
            if log_q0.dim() == 0 or log_q0.shape[0] != K:
                log_q0 = torch.log(torch.ones(K, device=device) / K)
            if self.debug:
                print(f"[BERAG-SG] Prior: user-provided log_passage_prior")
        elif self.prior_head is not None:
            if self.debug:
                print(f"[BERAG-SG] Prior: computing from prior head (batched forward over K passages)...")
            batched_inputs = self.backend.prepare_batched_input(x, [], passages)
            _, _, hidden_states = self.backend.forward(batched_inputs, None, return_hidden_states=True)
            log_q0, _ = self._compute_passage_prior_with_head(hidden_states)
        else:
            log_q0 = torch.log(torch.ones(K, device=device) / K)
            if self.debug:
                print(f"[BERAG-SG] Prior: uniform 1/K")

        chunk_indices: List[Tuple[int, ...]] = [(k,) for k in range(K)]
        current_passages = passages
        best_score = -float("inf")
        best_h: List[int] = []
        final_posterior: Optional[torch.Tensor] = None
        final_chunk_indices: List[Tuple[int, ...]] = chunk_indices
        last_beams: Optional[List[Dict[str, Any]]] = None
        last_all_finished: Optional[List[Dict[str, Any]]] = None
        last_chunk_indices: Optional[List[Tuple[int, ...]]] = None

        round_idx = 0
        while True:
            if self.debug:
                print(f"[BERAG-SG] ========== Round {round_idx} (K={len(current_passages)}) ==========")
            beams, alphas, beam_weighted_posterior, all_finished_beams = self._berag_sg_beam_search_one_round(
                x,
                current_passages,
                chunk_indices,
                log_q0,
                B,
                m,
                max_new_tokens,
                round_idx=round_idx,
                log_dir=log_dir_use,
                results_dir=results_dir,
                results_stem=results_stem,
                top_p=top_p,
                temperature=temperature,
                beam_prune=beam_prune,
                beam_score_mode=beam_score_mode,
            )
            round_best = alphas.max().item()
            k_star = int(torch.argmax(beam_weighted_posterior).item())
            last_beams = beams
            last_all_finished = all_finished_beams
            last_chunk_indices = list(chunk_indices)
            if log_dir_use or results_dir:
                self._write_berag_sg_posterior_log(
                    round_idx, chunk_indices, beam_weighted_posterior, k_star, round_best,
                    log_dir_use, results_dir, results_stem,
                )
                self._write_berag_sg_composite_states_log(
                    round_idx, chunk_indices, log_dir_use, results_dir, results_stem,
                )
            prev_best = best_score
            if round_best > best_score:
                best_score = round_best
                best_h = list(beams[0]["h"])
                final_posterior = beam_weighted_posterior.clone()
                final_chunk_indices = list(chunk_indices)
                if self.debug:
                    print(f"[BERAG-SG] Round {round_idx}: new best_score={best_score:.4f}, len(best_h)={len(best_h)}")

            if max_comp is not None and max_comp <= 1:
                if self.debug:
                    print(f"[BERAG-SG] Stopping: max_composite_size={max_comp} <= 1")
                break
            if len(current_passages) <= 1:
                if self.debug:
                    print(f"[BERAG-SG] Stopping: only {len(current_passages)} passage(s) left")
                break
            new_passages, new_chunk_indices = self._form_composite_chunks(
                current_passages, chunk_indices, k_star, max_comp,
            )
            if self.debug:
                print(f"[BERAG-SG] Round {round_idx}: k_star={k_star}, formed {len(new_passages)} composite chunks (indices: {[list(ci) for ci in new_chunk_indices]})")
            if not new_passages:
                if self.debug:
                    print(f"[BERAG-SG] Stopping: no new composite passages")
                break
            if round_idx >= 1 and round_best <= prev_best:
                if self.debug:
                    print(f"[BERAG-SG] Stopping: round_best={round_best:.4f} <= prev_best={prev_best:.4f} (no improvement)")
                break
            K_next = len(new_passages)
            log_q0 = torch.log(torch.ones(K_next, device=device) / K_next)
            current_passages = new_passages
            chunk_indices = new_chunk_indices
            round_idx += 1
            max_len_ci = max(len(ci) for ci in chunk_indices)
            if max_comp is not None and max_len_ci > max_comp:
                if self.debug:
                    print(f"[BERAG-SG] Stopping: max composite size reached ({max_len_ci} > {max_comp})")
                break

        if final_posterior is None:
            final_posterior = beam_weighted_posterior
        # Store all final beams + early-finished beams + posteriors for Test Summary / printing
        self._last_completed_beams = []
        if last_chunk_indices is not None:
            tokenizer = getattr(self.backend, "tokenizer", None)
            seen_h: set = set()

            def _append_beam(b: Dict[str, Any], beam_idx: int, label: str) -> None:
                q = b["q"]
                if isinstance(q, torch.Tensor):
                    probs = torch.softmax(q, dim=0).cpu().tolist()
                else:
                    probs = list(q)
                state_reprs = [list(ci) for ci in last_chunk_indices]
                generated_text = ""
                if tokenizer:
                    try:
                        generated_text = tokenizer.decode(b["h"], skip_special_tokens=True)
                    except Exception:
                        pass
                self._last_completed_beams.append({
                    "beam_idx": beam_idx,
                    "alpha": float(b["alpha"]),
                    "finished": b["terminated"],
                    "posterior": {"states": state_reprs, "probs": probs},
                    "generated_text": generated_text,
                    "source": label,
                })
                seen_h.add(tuple(b["h"]))

            if last_beams is not None:
                for i, b in enumerate(last_beams):
                    _append_beam(b, i, "survivor")
            # Add early-finished beams that were pruned out (not already in last_beams)
            if last_all_finished is not None:
                idx = len(self._last_completed_beams)
                for b in last_all_finished:
                    if tuple(b["h"]) not in seen_h:
                        _append_beam(b, idx, "early_finished")
                        idx += 1
        if self.debug:
            print(f"[BERAG-SG] Finished: best_score={best_score:.4f}, len(best_h)={len(best_h)}, rounds={round_idx + 1}")
        return best_h, final_posterior, final_chunk_indices

    def generate_berag_iterative_rerank(
        self,
        input_context: dict,
        passages: List[Union[str, dict]],
        max_new_tokens: int = 128,
        rerank_rounds: int = 1,
        tau: float = 1.0,
        lambda_: float = 0.5,
        opt_steps: int = 30,
        opt_lr: float = 0.05,
        berag_sg_temperature: Optional[float] = None,
        prior: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], torch.Tensor, List[Dict[str, Any]]]:
        """
        `berag_iterative_rerank`: BERAG without training via latent chunk-selection inference.

        This implementation follows the "full-response candidates" variant:
        - Generate one complete response hypothesis `y_k` per evidence chunk `c_k`.
        - Approximate the BERAG latent-chunk objective by restricting expectations to the
          discrete candidate set {y_k}.
        - Numerically maximize the KL-regularized objective w.r.t. the simplex distribution m.
        - Optionally repeat the procedure for `rerank_rounds` (regenerating candidates each round).
        - Decode/copy out the MAP chunk's hypothesis y_{k*} as the final return value.

        Notes on batching/KV cache:
        - Candidate generation uses the engine's batched KV-cache generation helper
          (`_generate_segment_under_chunks_batched`) with `h=[]` and `segment_length=max_new_tokens`,
          which yields full responses in a single segment.
        - Candidate scoring (cross/no-evidence log-likelihoods) uses batched forward passes with
          `return_full_logits=True` inside `_compute_segment_log_likelihood_all_chunks`.
        """
        x = input_context
        K = len(passages)
        device = self.device

        self._last_completed_beams = []

        if K == 0:
            empty_m = torch.empty(0, device=device, dtype=torch.float32)
            return [], empty_m, []

        # Candidate generation temperature: default to engine setting.
        temperature = (
            berag_sg_temperature if berag_sg_temperature is not None else self.berag_sg_temperature
        )

        # Build a "no-evidence" passage for p0(y)=P(y|x) by empty evidence replacement.
        # Supports both text-only passages and dict-based multimodal passages.
        first_passage = passages[0]
        if isinstance(first_passage, dict):
            # HF backends look for `zk.get("text", "")` and `zk.get("images", ...)`.
            no_evidence_passage: Union[str, dict] = {"text": "", "images": []}
        else:
            no_evidence_passage = ""

        passages_with_no_evidence: List[Union[str, dict]] = list(passages) + [no_evidence_passage]
        K_tot = K + 1
        evidence_chunk_indices = list(range(K))

        # Prior pi in simplex for KL(m||pi); updated each rerank round.
        if prior is None:
            pi = torch.full((K,), 1.0 / K, device=device, dtype=torch.float32)
        else:
            pi = prior.to(device=device, dtype=torch.float32)
            if pi.numel() != K:
                pi = torch.full((K,), 1.0 / K, device=device, dtype=torch.float32)
            else:
                pi = pi.clamp_min(0)
                pi = pi / (pi.sum() + 1e-12)

        # Helper tokenizer for candidate inspection logs.
        tokenizer = getattr(self.backend, "tokenizer", None)
        if tokenizer is None:
            processor = getattr(self.backend, "processor", None)
            tokenizer = getattr(processor, "tokenizer", None)

        last_round_candidates: List[List[int]] = []
        last_round_m_star: torch.Tensor = pi

        # Re-generate full candidates each rerank round (per user requirement).
        for round_idx in range(rerank_rounds):
            if self.debug:
                print(
                    f"[berag_iterative_rerank] round {round_idx+1}/{rerank_rounds}: "
                    f"K={K}, max_new_tokens={max_new_tokens}, tau={tau}, lambda_={lambda_}, "
                    f"opt_steps={opt_steps}, opt_lr={opt_lr}"
                )

            # 1) Generate K candidate full responses y_k ~ P(.|x,c_k).
            gen_results = self._generate_segment_under_chunks_batched(
                x=x,
                h=[],
                passages=passages,
                chunk_indices=evidence_chunk_indices,
                segment_length=max_new_tokens,
                temperature=temperature,
                batch_size=min(self.segment_gen_batch_size, K),
            )

            candidate_tokens: List[List[int]] = [seg_tokens for seg_tokens, _ in gen_results]
            candidate_self_logliks: List[float] = [float(log_lik) for _, log_lik in gen_results]
            last_round_candidates = candidate_tokens

            if self.debug:
                cand_lens = [len(toks) for toks in candidate_tokens]
                topk_len = sorted(cand_lens, reverse=True)[: min(5, len(cand_lens))]
                print(
                    f"[berag_iterative_rerank] round {round_idx+1}: "
                    f"generated {K} candidates; "
                    f"len(min/mean/max)={min(cand_lens)}/{sum(cand_lens)/max(1,len(cand_lens)):.2f}/{max(cand_lens)}; "
                    f"top-lens={topk_len}"
                )
                if candidate_self_logliks:
                    ll_min = min(candidate_self_logliks)
                    ll_max = max(candidate_self_logliks)
                    ll_mean = sum(candidate_self_logliks) / max(1, len(candidate_self_logliks))
                    print(
                        f"[berag_iterative_rerank] round {round_idx+1}: "
                        f"self-loglik(s_k) min/mean/max={ll_min:.3f}/{ll_mean:.3f}/{ll_max:.3f}"
                    )

            # 2) Precompute cross/no-evidence log-likelihoods on the discrete candidate set.
            #    log_a[j,k] = log P(y_k | x, c_j) for evidence chunks j in [0..K-1]
            #    b[k]       = log P(y_k | x)      (no-evidence) in row j=K
            log_a = torch.zeros((K, K), device=device, dtype=torch.float32)
            b = torch.zeros((K,), device=device, dtype=torch.float32)

            for k in range(K):
                y_k = candidate_tokens[k]
                log_lik_vec = self._compute_segment_log_likelihood_all_chunks(
                    x=x,
                    h=[],
                    segment_tokens=y_k,
                    passages=passages_with_no_evidence,
                )  # shape: (K+1,)
                log_a[:, k] = log_lik_vec[:K]
                b[k] = log_lik_vec[K]

            if self.debug:
                diag_a = torch.diag(log_a)
                print(
                    f"[berag_iterative_rerank] round {round_idx+1}: "
                    f"log_a diag(min/mean/max)={diag_a.min().item():.3f}/"
                    f"{diag_a.mean().item():.3f}/{diag_a.max().item():.3f}; "
                    f"b(min/mean/max)={b.min().item():.3f}/{b.mean().item():.3f}/{b.max().item():.3f}"
                )
                # Quick sanity: compare diag(log_a) vs candidate_self_logliks if available.
                if candidate_self_logliks and len(candidate_self_logliks) == K:
                    s_k = torch.tensor(candidate_self_logliks, device=device, dtype=torch.float32)
                    diff = (diag_a - s_k).abs()
                    print(
                        f"[berag_iterative_rerank] round {round_idx+1}: "
                        f"|diag(log_a)-s_k| mean/max={diff.mean().item():.6f}/{diff.max().item():.6f}"
                    )

            # 3) Numerically maximize r_hat(m) over the simplex (softmax parameterization).
            #    Initialize logits from current prior pi (m starts at pi).
            m_logits = torch.log(pi + 1e-12).clone().detach().requires_grad_(True)
            log_pi = torch.log(pi + 1e-12).detach()

            optimizer = torch.optim.Adam([m_logits], lr=opt_lr)
            eps = 1e-12

            m_star = pi
            for step in range(opt_steps):
                m_prob = torch.softmax(m_logits, dim=0)  # m in simplex
                log_m = torch.log(m_prob + eps)

                # log_u[k] = log sum_j exp(log m_j + a_{j,k})
                # log_a has shape [K,K] with a_{j,k} = log P(y_k|c_j)
                log_u = torch.logsumexp(log_m[:, None] + log_a, dim=0)  # [K]
                log_Z = torch.logsumexp(log_u, dim=0)  # scalar

                # Candidate mixture weights over the discrete set: w_k = exp(log_u - log_Z)
                log_q = log_u - log_Z  # log w_k (since w_k normalizes u across candidates)
                w = torch.exp(log_q)  # [K]

                # Responsibilities over chunks for each candidate: P(Z=j|y_k)
                # log_r[j,k] = log m_j + a_{j,k} - log_u[k]
                log_r = log_m[:, None] + log_a - log_u[None, :]
                r = torch.exp(log_r)  # [K,K]

                # Entropy H(P(Z|y_k)) per candidate.
                # entropy_k[k] = -sum_j r[j,k] * log r[j,k]
                entropy_k = -(r * log_r).sum(dim=0)  # [K]

                # KL(m||pi) = sum_j m_j (log m_j - log pi_j)
                kl = (m_prob * (log_m - log_pi)).sum()

                # Expected info term:
                # sum_k w_k * (log q_m(y_k) - log p0(y_k)) where log q_m(y_k) ~= log w_k in this
                # discrete approximation.
                expected_info = (w * (log_q - b)).sum()
                expected_entropy = (w * entropy_k).sum()

                r_hat = expected_info - tau * expected_entropy - lambda_ * kl
                loss = -r_hat

                if torch.isnan(loss):
                    # If we hit numerical instability, stop early and keep current pi.
                    if self.debug:
                        print(f"[berag_iterative_rerank] step {step}: NaN loss; aborting optimization.")
                    break

                if self.debug:
                    topm_vals, topm_idx = torch.topk(m_prob, k=min(5, K), dim=0)
                    w_top_vals, w_top_idx = torch.topk(w, k=min(5, K), dim=0)
                    # Scalar summaries only; avoid printing full vectors for large K.
                    ent_m = -(m_prob * log_m).sum().item()
                    print(
                        f"[berag_iterative_rerank] opt step {step}: "
                        f"r_hat={r_hat.item():.4f}, loss={loss.item():.4f}, "
                        f"expected_info={expected_info.item():.4f}, "
                        f"expected_entropy={expected_entropy.item():.4f}, "
                        f"kl={kl.item():.4f}, H(m)={ent_m:.4f}; "
                        f"m_top5={list(zip(topm_idx.tolist(), [round(v.item(),4) for v in topm_vals]))}; "
                        f"w_top5={list(zip(w_top_idx.tolist(), [round(v.item(),4) for v in w_top_vals]))}"
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            m_star = torch.softmax(m_logits, dim=0).detach()
            last_round_m_star = m_star
            pi = m_star  # rerank: prior becomes m*

            if self.debug:
                log_m_star = torch.log(last_round_m_star + 1e-12)
                ent_m_star = -(last_round_m_star * log_m_star).sum().item()
                topm_vals, topm_idx = torch.topk(last_round_m_star, k=min(5, K), dim=0)
                print(
                    f"[berag_iterative_rerank] round {round_idx+1} done: "
                    f"m_star entropy={ent_m_star:.4f}; "
                    f"m_star_top5={list(zip(topm_idx.tolist(), [round(v.item(),4) for v in topm_vals]))}"
                )

        # 4) MAP chunk selection and final response output.
        k_star = int(torch.argmax(last_round_m_star).item())
        best_tokens = last_round_candidates[k_star] if last_round_candidates else []

        # Populate inspection candidates (one per chunk) similar to beam-search-style logging.
        # We store the posterior responsibilities P_{m*}(Z|y_k) for each candidate y_k.
        for k in range(K):
            y_k = last_round_candidates[k] if last_round_candidates else []
            finished = bool(y_k) and self.backend.is_stop_token(y_k[-1])
            generated_text = ""
            if tokenizer is not None and y_k:
                try:
                    generated_text = tokenizer.decode(y_k, skip_special_tokens=True)
                except Exception:
                    generated_text = ""

            # Compute responsibilities for this candidate:
            # r[j,k] = m*_j * exp(a_{j,k}) / sum_l m*_l * exp(a_{l,k})
            # (log-space for stability).
            eps = 1e-12
            log_m_star = torch.log(last_round_m_star + eps)  # [K]
            log_u_k = torch.logsumexp(log_m_star + log_a[:, k], dim=0)  # scalar
            log_r_jk = log_m_star + log_a[:, k] - log_u_k  # [K]
            posterior_probs = torch.exp(log_r_jk).detach().cpu().tolist()

            if self.debug and k in torch.topk(last_round_m_star, k=min(3, K), dim=0).indices.tolist():
                post_t = torch.tensor(posterior_probs, device=device, dtype=torch.float32)
                post_log_t = torch.log(post_t + 1e-12)
                post_ent = (-(post_t * post_log_t).sum()).item()
                post_max = float(post_t.max().item())
                print(
                    f"[berag_iterative_rerank] inspection: candidate k={k} "
                    f"alpha(m*)={last_round_m_star[k].item():.4f}, "
                    f"posterior max={post_max:.4f}, posterior entropy={post_ent:.4f}, "
                    f"posterior argmax={int(torch.argmax(post_t).item())}"
                )

            self._last_completed_beams.append({
                "beam_idx": k,
                "alpha": float(last_round_m_star.detach().cpu().tolist()[k]),
                "finished": finished,
                "posterior": {"states": [[j] for j in range(K)], "probs": posterior_probs},
                "generated_text": generated_text,
                "source": "berag_iterative_rerank",
            })

        if self.debug:
            print(
                f"[berag_iterative_rerank] Done: k_star={k_star}, "
                f"m_star_max={float(last_round_m_star.max().item()):.4f}, "
                f"best_tokens_len={len(best_tokens)}"
            )

        return best_tokens, last_round_m_star, self._last_completed_beams

    def _do_greedy(self, input_context, passages, log_passage_prior=None, max_new_tokens=512, passage_top_p=None):
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

        # First forward pass
        log_all_policy_next_token_probs, past_key_values, hidden_states = self.backend.forward(batched_inputs, past_key_values, return_hidden_states=True)
        if self.prior_head is not None:
            log_passage_prior = self._compute_passage_prior_with_head(hidden_states)
            prior_logits = log_passage_prior.detach().cpu().tolist()
            if "pixel_values" in batched_inputs:
                del batched_inputs["pixel_values"]  # already in kv cache after first pass
            if "image_grid_thw" in batched_inputs:
                del batched_inputs["image_grid_thw"]

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

            # update batched inputs by appending the new token
            new_token_tensor = torch.full((K, 1), token_idx, device=self.device, dtype=batched_inputs["input_ids"].dtype)
            new_attention_tensor = torch.ones((K, 1), device=self.device, dtype=batched_inputs["attention_mask"].dtype)
            batched_inputs["input_ids"] = torch.cat([batched_inputs["input_ids"], new_token_tensor], dim=1)
            batched_inputs["attention_mask"] = torch.cat([batched_inputs["attention_mask"], new_attention_tensor], dim=1)

            log_all_policy_next_token_probs, past_key_values, hidden_states = self.backend.forward(batched_inputs, past_key_values, return_hidden_states=True)

        return generated_tokens, log_all_tokens_llk, posterior_logits_over_steps, prior_logits
    
    def _do_beam_search(self, input_context, passages, log_passage_prior=None, max_new_tokens=512, passage_top_p=None, return_n_sequences=1):
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
                all_beam_scores: List of log scores (one per beam)
                all_normalized_scores: List of length-normalized scores (one per beam)
                all_posterior_logits: List of posterior-over-steps (one list per beam)
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
        
        # First forward pass to get initial probs and compute passage prior if needed
        log_all_policy_next_token_probs, past_key_values, hidden_states = self.backend.forward(
            batched_inputs, past_key_values, return_hidden_states=True
        )
        
        if self.prior_head is not None:
            log_passage_prior, passage_prior_logits = self._compute_passage_prior_with_head(hidden_states)
            prior_logits = passage_prior_logits.detach().cpu().tolist()
            # Remove pixel values after first pass (already cached)
            if "pixel_values" in batched_inputs:
                del batched_inputs["pixel_values"]
            if "image_grid_thw" in batched_inputs:
                del batched_inputs["image_grid_thw"]
        
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
            new_attention_tensor = torch.ones_like(new_token_tensor)
            
            # Subsequent steps: just use new tokens (KV cache has history)
            batched_inputs["input_ids"] = new_token_tensor
            batched_inputs["attention_mask"] = new_attention_tensor
            
            # Forward pass for all beams
            log_all_policy_next_token_probs, past_key_values, _ = self.backend.forward(
                batched_inputs, past_key_values
            )
        
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
            if return_n_sequences == 1:
                return [], [], [], prior_logits if prior_logits else []
            else:
                return [[]], [], [], [], prior_logits if prior_logits else []
        
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
            
            return best_beam_tokens, log_all_tokens_llk, posterior_logits_over_steps, prior_logits
        
        # Return multiple sequences: all candidates with beam scores and posteriors
        else:
            all_generated_tokens = []
            all_beam_scores = []  # log_score per beam
            all_normalized_scores = []  # length-normalized score per beam
            all_posterior_logits = []
            
            for beam in sorted_finished[:n_to_return]:
                all_generated_tokens.append(beam['generated_tokens'])
                all_beam_scores.append(float(beam['log_score']))
                all_normalized_scores.append(float(beam['normalized_score']))
                all_posterior_logits.append(beam['posterior_logits_over_steps'])
            
            if self.debug:
                print(f"\n✓ Returning top {n_to_return} beams")
                print(f"✓ Best normalized score: {sorted_finished[0]['normalized_score']:.4f}")
                print(f"{'='*80}\n")
            
            return all_generated_tokens, all_beam_scores, all_normalized_scores, all_posterior_logits, prior_logits
    

    def generate_v2(self, input_context, passages, log_passage_prior=None, max_new_tokens=512, passage_top_p=None, return_n_sequences=1):
        if self.num_beams > 1:
            return self._do_beam_search(input_context, passages, log_passage_prior, max_new_tokens, passage_top_p, return_n_sequences)
        else:
            return self._do_greedy(input_context, passages, log_passage_prior, max_new_tokens, passage_top_p)
    
    def generate(self, input_context, passages, log_passage_prior=None, max_new_tokens=512, passage_top_p=None):
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
        
        # Initialize past key values cache
        num_hidden_layers = _get_config_attr(self.backend.model.config, "num_hidden_layers")
        if num_hidden_layers is None:
            raise AttributeError("Cannot get num_hidden_layers from model config (tried config and config.text_config)")
        past_key_values = _make_dynamic_cache(num_hidden_layers)
        
        # Prepare initial batched inputs
        batched_inputs = self.backend.prepare_batched_input(x, generated_tokens, passages)
        active_indices, inactive_indices = torch.arange(K, device=self.device), None
        previous_K = K
        current_idx_to_original_idx_map = {i: i for i in range(K)}

        prior_max_idx = -1
        posterior_max_idx = -1

        # Uniform log prior over passages when none provided and no prior_head
        if log_passage_prior is None:
            log_passage_prior = torch.log(torch.ones(K, device=self.device, dtype=torch.float32) / K)
        
        while new_tokens < max_new_tokens:
            # Compute policy next token probability using batched forward pass
            active_K = active_indices.shape[0]
            if active_K != previous_K: # resize/prune variables.
                previous_K = active_K
                past_key_values = self._keep_past_key_values(past_key_values, active_indices)
                batched_inputs = self._keep_batched_inputs(batched_inputs, active_indices)
                log_previous_all_policy_next_token_probs = log_previous_all_policy_next_token_probs[active_indices].contiguous()
                log_previous_all_passage_conditioned_likelihood = log_previous_all_passage_conditioned_likelihood[active_indices].contiguous()
                log_passage_prior = log_passage_prior[active_indices].contiguous()


            if new_tokens == 0 and self.prior_head is not None:
                log_all_policy_next_token_probs, past_key_values, hidden_states = self.backend.forward(batched_inputs, past_key_values, return_hidden_states=True)
                log_passage_prior, passage_prior_logits = self._compute_passage_prior_with_head(hidden_states)
                prior_max_idx = torch.argmax(log_passage_prior).item()
                # Save prior logits for return
                prior_logits = passage_prior_logits.detach().cpu().tolist()
                if "pixel_values" in batched_inputs:
                    del batched_inputs["pixel_values"]  # already in kv cache after first pass
                if "image_grid_thw" in batched_inputs:
                    del batched_inputs["image_grid_thw"]
            else:
                log_all_policy_next_token_probs, past_key_values, _ = self.backend.forward(batched_inputs, past_key_values)
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

            # Update batched inputs by appending the new token
            # Create new token tensor for all passages
            new_token_tensor = torch.full((active_K, 1), token_idx, device=self.device, dtype=batched_inputs["input_ids"].dtype)
            new_attention_tensor = torch.ones((active_K, 1), device=self.device, dtype=batched_inputs["attention_mask"].dtype)
            
            # Append to batched inputs
            # batched_inputs["input_ids"] = torch.cat([batched_inputs["input_ids"], new_token_tensor], dim=1)
            # batched_inputs["attention_mask"] = torch.cat([batched_inputs["attention_mask"], new_attention_tensor], dim=1)
            batched_inputs["input_ids"] = new_token_tensor
            batched_inputs["attention_mask"] = new_attention_tensor

            posterior_max_idx = current_idx_to_original_idx_map[torch.argmax(log_passage_posterior).item()]
            
            # Note: No need to re-pad since we're just appending one token to all sequences
            # All sequences will have the same length after appending
            
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
        Properly handles vision inputs (pixel_values, image_grid_thw).
        """
        pruned_inputs = {}
        batch_size = batched_inputs["input_ids"].shape[0]
        
        # Check if we have vision inputs
        has_pixel_values = "pixel_values" in batched_inputs
        has_image_grid_thw = "image_grid_thw" in batched_inputs
        
        if has_pixel_values and has_image_grid_thw:
            # Vision-language model: need to prune pixel_values based on image_grid_thw
            image_grid_thw = batched_inputs["image_grid_thw"]  # Shape: [batch_size, 3]
            pixel_values = batched_inputs["pixel_values"]  # Shape: [total_patches, hidden_dim]
            
            # Calculate number of patches per image: t * h * w for each image
            patches_per_image = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).cpu().tolist()
            
            # Calculate which patches to keep based on active_indices
            active_indices_list = active_indices.cpu().tolist()
            patches_to_keep = []
            cumulative_patches = 0
            for idx in range(batch_size):
                num_patches = int(patches_per_image[idx])
                if idx in active_indices_list:
                    patches_to_keep.extend(range(cumulative_patches, cumulative_patches + num_patches))
                cumulative_patches += num_patches
            
            # Keep only patches for active images
            patches_to_keep = torch.tensor(patches_to_keep, device=pixel_values.device, dtype=torch.long)
            pruned_inputs["pixel_values"] = pixel_values[patches_to_keep].contiguous()
            pruned_inputs["image_grid_thw"] = image_grid_thw[active_indices].contiguous()
        
        # Handle all other inputs
        for key, value in batched_inputs.items():
            if key in ["pixel_values", "image_grid_thw"]:
                # Already handled above
                continue
            elif isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
                # Batched tensor - prune it
                pruned_inputs[key] = value[active_indices].contiguous()
            else:
                # Non-batched or non-tensor - keep as-is
                pruned_inputs[key] = value
        
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
