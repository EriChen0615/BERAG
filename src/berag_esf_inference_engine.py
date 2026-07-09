"""
Evidence-State Filtering (ESF) for segment-level BERAG.

Evidence-state space S = 2^[K]. Belief filtering: predict -> decode -> update per segment.
Uses hf_backend (Qwen2.5-VL) for LLM forward passes.
"""

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    from . import cache_compat
except ImportError:
    import cache_compat

try:
    from transformers import DynamicCache
except ImportError:
    DynamicCache = None


def _get_config_attr(config, attr):
    """Get attribute from config, with fallback to text_config for VL configs."""
    val = getattr(config, attr, None)
    if val is not None:
        return val
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return getattr(text_config, attr, None)
    return None


def _make_dynamic_cache(num_hidden_layers=None):
    """Create a DynamicCache compatible with transformers API."""
    if DynamicCache is None:
        raise ImportError("transformers.DynamicCache not available")
    try:
        if num_hidden_layers is not None:
            return DynamicCache(num_hidden_layers=num_hidden_layers)
    except TypeError:
        pass
    return DynamicCache()


def _passage_text(passage, k: int) -> str:
    """Extract raw text from passage at index k. String or dict with 'text' key."""
    if isinstance(passage, str):
        return passage
    if isinstance(passage, dict):
        return passage.get("text", "")
    return str(passage)


def pack_passages_for_state(
    passages: List[Union[str, dict]], state: frozenset
) -> Union[str, dict]:
    """
    Pack passages for evidence state s by simple concatenation.
    Returns packed string (or dict for multimodal) for Z_s = {z_k : k in s}.
    """
    if not state:
        return ""
    texts = [_passage_text(passages[k], k) for k in sorted(state)]
    packed = "\n".join(texts)
    if isinstance(passages[0], dict) and "images" in passages[0]:
        images = []
        for k in sorted(state):
            p = passages[k]
            if isinstance(p, dict) and p.get("images"):
                images.extend(p["images"])
        return {"text": packed, "images": images} if images else packed
    return packed


def _initial_support(K: int) -> List[frozenset]:
    """Return K initial states: one singleton per document (no empty set)."""
    return [frozenset({k}) for k in range(K)]


def _get_neighborhood(
    s: frozenset, K: int, max_state_size: int
) -> List[frozenset]:
    """Neighborhood N(s): stay, add one, drop one (under size cap)."""
    neighbors = [s]
    for j in range(K):
        if j not in s:
            s_new = s | frozenset({j})
            if len(s_new) <= max_state_size:
                neighbors.append(s_new)
    for j in s:
        s_new = s - frozenset({j})
        neighbors.append(s_new)
    return list(neighbors)


def _symmetric_difference_size(s: frozenset, s_prime: frozenset) -> int:
    """|s △ s'| = |s \ s'| + |s' \ s|"""
    return len(s ^ s_prime)


def _to_tensor(
    log_b_dict: Dict[frozenset, float],
    state_list: List[frozenset],
    device: torch.device,
) -> torch.Tensor:
    """Build 1D tensor of log beliefs in state_list order; use -inf for missing states."""
    return torch.tensor(
        [log_b_dict.get(s, -float("inf")) for s in state_list],
        device=device,
        dtype=torch.float32,
    )


def _to_dict(
    log_b_tensor: torch.Tensor,
    state_list: List[frozenset],
) -> Dict[frozenset, float]:
    """Build dict from tensor and state_list (for segment_beliefs and API expecting dicts)."""
    return {s: log_b_tensor[i].item() for i, s in enumerate(state_list)}


class BERAGESFInferenceEngine:
    """
    Evidence-State Filtering inference engine for segment-level BERAG.
    Uses HFQwen2VLBackend for Qwen2.5-VL models.
    """

    def __init__(
        self,
        backend,
        segment_size: int = 1,
        state_explore_TopK: int = 10,
        decode_mode: str = "segment_beam",
        beam_width: int = 4,
        log_dir: Optional[str] = None,
        debug: bool = False,
        transition_kernel: str = "identity",
        beta: float = 1.0,
        lambda_edit: float = 0.1,
        lambda_size: float = 0.0,
        max_state_size: Optional[int] = None,
        state_explore_mode: str = "TopP_capped",
        top_p: float = 0.95,
        threshold_single: Optional[float] = None,
        threshold_combined: float = 0.5,
        mass_threshold_epsilon: float = 0.0,
    ):
        self.backend = backend
        self.device = backend.device
        self.segment_size = segment_size
        self.state_explore_TopK = state_explore_TopK
        self.decode_mode = decode_mode
        self.beam_width = beam_width
        self.log_dir = log_dir
        self.debug = debug
        self.transition_kernel = transition_kernel
        self.beta = beta
        self.lambda_edit = lambda_edit
        self.lambda_size = lambda_size
        self.max_state_size = max_state_size
        self.state_explore_mode = state_explore_mode
        self.top_p = top_p
        self.threshold_single = threshold_single
        self.threshold_combined = threshold_combined
        self.mass_threshold_epsilon = mass_threshold_epsilon

        if transition_kernel not in ("identity", "log_linear", "multiplicative_threshold", "mass_threshold_kernel"):
            raise ValueError(
                f"transition_kernel must be 'identity', 'log_linear', 'multiplicative_threshold', or 'mass_threshold_kernel', got {transition_kernel}"
            )
        if decode_mode not in ("mixture", "map", "segment_beam"):
            raise ValueError(
                f"decode_mode must be 'mixture', 'map', or 'segment_beam', got {decode_mode}"
            )
        if decode_mode == "segment_beam" and beam_width < 1:
            raise ValueError(
                f"beam_width must be >= 1 when decode_mode is 'segment_beam', got {beam_width}"
            )
        if state_explore_mode not in ("TopK", "TopP", "TopP_capped"):
            raise ValueError(
                f"state_explore_mode must be 'TopK', 'TopP', or 'TopP_capped', got {state_explore_mode}"
            )

        num_hidden_layers = _get_config_attr(
            self.backend.model.config, "num_hidden_layers"
        )
        if num_hidden_layers is None:
            raise AttributeError(
                "Cannot get num_hidden_layers from model config"
            )
        self._num_hidden_layers = num_hidden_layers

    def _prune_support_tensor(
        self,
        state_list: List[frozenset],
        log_belief: torch.Tensor,
    ) -> Tuple[List[frozenset], torch.Tensor]:
        """Prune to top-B states and renormalize. Returns (state_list, log_belief tensor)."""
        n = min(self.state_explore_TopK, log_belief.numel())
        if n >= log_belief.numel():
            log_z = torch.logsumexp(log_belief, dim=0)
            return state_list, log_belief - log_z
        top_vals, top_idx = torch.topk(log_belief, n, largest=True)
        log_z = torch.logsumexp(top_vals, dim=0)
        top_vals = top_vals - log_z
        new_state_list = [state_list[i] for i in top_idx.cpu().tolist()]
        return new_state_list, top_vals

    def _prune_support(
        self, log_belief: Dict[frozenset, float]
    ) -> Dict[frozenset, float]:
        """Prune to top-B states and renormalize (dict API)."""
        state_list = list(log_belief.keys())
        log_b_tensor = _to_tensor(log_belief, state_list, self.device)
        state_list_new, log_b_new = self._prune_support_tensor(
            state_list, log_b_tensor
        )
        return _to_dict(log_b_new, state_list_new)

    def _predict_identity(
        self, log_b_t: Dict[frozenset, float]
    ) -> Dict[frozenset, float]:
        """Identity: b̄_{t+1} = b_t."""
        return dict(log_b_t)

    def _transition_log_prob(
        self,
        s: frozenset,
        s_prime: frozenset,
        log_U: float,
    ) -> float:
        """log P(s'|s) = β U - λ_edit |s△s'| - λ_size |s'| - log Z(s)."""
        edit_cost = _symmetric_difference_size(s, s_prime)
        size_cost = len(s_prime)
        return (
            self.beta * log_U
            - self.lambda_edit * edit_cost
            - self.lambda_size * size_cost
        )

    def _predict_log_linear_tensor(
        self,
        state_list: List[frozenset],
        log_b_t: torch.Tensor,
        log_segment_likelihoods: Optional[torch.Tensor],
        K: int,
    ) -> Tuple[List[frozenset], torch.Tensor]:
        """
        b̄_{t+1}(s') = sum_s T(s'|s) b_t(s). Vectorized via transition matrix.
        Returns (state_list_next, log_bar_b tensor).
        """
        max_size = self.max_state_size if self.max_state_size is not None else K
        all_s_prime = set()
        for s in state_list:
            all_s_prime.update(_get_neighborhood(s, K, max_size))
        state_list_next = sorted(all_s_prime, key=lambda s: (len(s), sorted(s)))

        n_next = len(state_list_next)
        n_curr = len(state_list)
        if n_next == 0:
            return [], torch.tensor([], device=self.device, dtype=torch.float32)

        log_T = torch.full(
            (n_next, n_curr),
            -float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        s_prime_to_idx = {state_list_next[i]: i for i in range(n_next)}
        state_to_idx = {state_list[i]: i for i in range(n_curr)}

        if log_segment_likelihoods is not None:
            log_seg_next = torch.full(
                (n_next,), -float("inf"), device=self.device, dtype=torch.float32
            )
            for i, s in enumerate(state_list_next):
                if s in state_to_idx:
                    log_seg_next[i] = log_segment_likelihoods[state_to_idx[s]]
        else:
            log_seg_next = None

        for i in range(n_curr):
            s_i = state_list[i]
            neighbors_i = _get_neighborhood(s_i, K, max_size)
            log_probs_neighbors = []
            for s_prime in neighbors_i:
                if s_prime == s_i:
                    log_U = 0.0
                elif log_seg_next is not None:
                    j = s_prime_to_idx[s_prime]
                    log_U = (log_seg_next[j] - log_segment_likelihoods[i]).item()
                else:
                    log_U = 0.0
                lp = self._transition_log_prob(s_i, s_prime, log_U)
                log_probs_neighbors.append((s_prime, lp))
            log_Z_i = torch.logsumexp(
                torch.tensor(
                    [lp for _, lp in log_probs_neighbors],
                    device=self.device,
                    dtype=torch.float32,
                ),
                dim=0,
            ).item()
            for s_prime, lp in log_probs_neighbors:
                j = s_prime_to_idx[s_prime]
                log_T[j, i] = lp - log_Z_i

        log_bar_b_next = torch.logsumexp(log_T + log_b_t[None, :], dim=1)
        state_list_new, log_bar_b_new = self._prune_support_tensor(
            state_list_next, log_bar_b_next
        )
        return state_list_new, log_bar_b_new

    def _predict_multiplicative_threshold_tensor(
        self,
        state_list: List[frozenset],
        log_b_tensor: torch.Tensor,
        K: int,
    ) -> Tuple[List[frozenset], torch.Tensor]:
        """
        b̄_{t+1}(s') = sum_s T(s'|s) b(s). T(s|s)=1 if no valid pairs; else T(s'|s)=1/M(s)
        for stay and each of M(s)-1 composite targets. Valid pair: both b(s)>t_s, b(s)+b(s_j)>t_w.
        """
        t_s = (
            self.threshold_single
            if self.threshold_single is not None
            else (2.0 / K)
        )
        t_w = self.threshold_combined
        max_size = self.max_state_size if self.max_state_size is not None else K
        n_curr = len(state_list)
        if n_curr == 0:
            return [], torch.tensor([], device=self.device, dtype=torch.float32)

        probs = torch.softmax(log_b_tensor, dim=0)

        # Valid partners from each state i: j != i with probs[i]>t_s, probs[j]>t_s, probs[i]+probs[j]>t_w, len(s_i|s_j)<=max_size
        valid_partners: List[List[int]] = []
        all_composites: set = set()
        for i in range(n_curr):
            partners_i: List[int] = []
            for j in range(n_curr):
                if i == j:
                    continue
                if probs[i].item() <= t_s or probs[j].item() <= t_s:
                    continue
                if (probs[i] + probs[j]).item() <= t_w:
                    continue
                s_union = state_list[i] | state_list[j]
                if len(s_union) > max_size:
                    continue
                partners_i.append(j)
                all_composites.add(s_union)
            valid_partners.append(partners_i)

        M_list = [1 + len(partners_i) for partners_i in valid_partners]

        state_list_next = sorted(
            set(state_list) | all_composites,
            key=lambda s: (len(s), sorted(s)),
        )
        n_next = len(state_list_next)
        s_prime_to_idx = {state_list_next[i]: i for i in range(n_next)}

        log_T = torch.full(
            (n_next, n_curr),
            -float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        for i in range(n_curr):
            M_i = M_list[i]
            log_p = math.log(1.0 / M_i)
            # Self
            row_self = s_prime_to_idx[state_list[i]]
            log_T[row_self, i] = log_p
            # Composites
            for j in valid_partners[i]:
                s_prime = state_list[i] | state_list[j]
                row = s_prime_to_idx[s_prime]
                log_T[row, i] = log_p

        log_bar_b_next = torch.logsumexp(log_T + log_b_tensor[None, :], dim=1)
        state_list_new, log_bar_b_new = self._prune_support_tensor(
            state_list_next, log_bar_b_next
        )
        return state_list_new, log_bar_b_new

    def _predict_mass_threshold_tensor(
        self,
        state_list: List[frozenset],
        log_b_tensor: torch.Tensor,
        K: int,
    ) -> Tuple[List[frozenset], torch.Tensor]:
        """
        mass_threshold_kernel: combine states when both have probability > 1/K + epsilon.
        T(s'|s) = mass_at_destination(s') / Z(s), where Z(s) = sum of masses at all possible
        destinations from s (self + composites). Self has mass b(s); composite s∪s_j has mass b(s)+b(s_j).
        So composite transition probability is always > self-transition when b(s_j) > 0.
        """
        t = (1.0 / K) + self.mass_threshold_epsilon
        max_size = self.max_state_size if self.max_state_size is not None else K
        n_curr = len(state_list)
        if n_curr == 0:
            return [], torch.tensor([], device=self.device, dtype=torch.float32)

        probs = torch.softmax(log_b_tensor, dim=0)

        # Valid partners from state i: j != i, probs[i] > t, probs[j] > t, len(s_i | s_j) <= max_size
        valid_partners: List[List[int]] = []
        all_composites: set = set()
        for i in range(n_curr):
            partners_i: List[int] = []
            for j in range(n_curr):
                if i == j:
                    continue
                if probs[i].item() <= t or probs[j].item() <= t:
                    continue
                s_union = state_list[i] | state_list[j]
                if len(s_union) > max_size:
                    continue
                partners_i.append(j)
                all_composites.add(s_union)
            valid_partners.append(partners_i)

        # Destinations from each state i: self (mass probs[i]) + composites (mass probs[i]+probs[j])
        # Z_i = probs[i] + sum_{j in partners_i} (probs[i] + probs[j])
        state_list_next = sorted(
            set(state_list) | all_composites,
            key=lambda s: (len(s), sorted(s)),
        )
        n_next = len(state_list_next)
        s_prime_to_idx = {state_list_next[ix]: ix for ix in range(n_next)}

        log_T = torch.full(
            (n_next, n_curr),
            -float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        for i in range(n_curr):
            mass_self = probs[i].item()
            mass_composites = [
                (probs[i] + probs[j]).item() for j in valid_partners[i]
            ]
            Z_i = mass_self + sum(mass_composites)
            if Z_i <= 0:
                Z_i = 1.0
            # Self-transition
            row_self = s_prime_to_idx[state_list[i]]
            log_T[row_self, i] = math.log(mass_self / Z_i)
            # Composites
            for jj, j in enumerate(valid_partners[i]):
                s_prime = state_list[i] | state_list[j]
                row = s_prime_to_idx[s_prime]
                log_T[row, i] = math.log(mass_composites[jj] / Z_i)

        log_bar_b_next = torch.logsumexp(log_T + log_b_tensor[None, :], dim=1)
        state_list_new, log_bar_b_new = self._prune_support_tensor(
            state_list_next, log_bar_b_next
        )
        return state_list_new, log_bar_b_new

    def _predict_log_linear(
        self,
        log_b_t: Dict[frozenset, float],
        log_segment_likelihoods: Optional[Dict[frozenset, float]],
        K: int,
    ) -> Dict[frozenset, float]:
        """Dict API: b̄_{t+1}(s') = sum_s T(s'|s) b_t(s)."""
        state_list = list(log_b_t.keys())
        log_b_tensor = _to_tensor(log_b_t, state_list, self.device)
        log_seg_tensor = (
            _to_tensor(log_segment_likelihoods, state_list, self.device)
            if log_segment_likelihoods is not None
            else None
        )
        state_list_new, log_bar_b_tensor = self._predict_log_linear_tensor(
            state_list, log_b_tensor, log_seg_tensor, K
        )
        return _to_dict(log_bar_b_tensor, state_list_new)

    def _predict(
        self,
        state_list: List[frozenset],
        log_b_tensor: torch.Tensor,
        log_segment_likelihoods_prev: Optional[Dict[frozenset, float]],
        K: int,
        segment_idx: Optional[int] = None,
    ) -> Tuple[List[frozenset], torch.Tensor]:
        """Predict: b̄_{t+1}. Returns (state_list_bar, log_bar_b tensor).
        Transition kernel is applied only after the first segment (segment_idx >= 1);
        for segment_idx == 0 we use identity."""
        if segment_idx is not None and segment_idx == 0:
            return state_list, log_b_tensor
        if self.transition_kernel == "identity":
            return state_list, log_b_tensor
        if self.transition_kernel == "multiplicative_threshold":
            return self._predict_multiplicative_threshold_tensor(
                state_list, log_b_tensor, K
            )
        if self.transition_kernel == "mass_threshold_kernel":
            return self._predict_mass_threshold_tensor(
                state_list, log_b_tensor, K
            )
        log_seg_tensor = (
            _to_tensor(log_segment_likelihoods_prev, state_list, self.device)
            if log_segment_likelihoods_prev is not None
            else None
        )
        return self._predict_log_linear_tensor(
            state_list, log_b_tensor, log_seg_tensor, K
        )

    def _update_tensor(
        self,
        state_list: List[frozenset],
        log_bar_b: torch.Tensor,
        log_segment_lik: torch.Tensor,
    ) -> Tuple[List[frozenset], torch.Tensor]:
        """b_{t+1}(s) ∝ P_θ(y_t|h_t,x,s) b̄_{t+1}(s). Returns (state_list, log_b tensor)."""
        log_b = log_segment_lik + log_bar_b
        log_b = log_b - torch.logsumexp(log_b, dim=0)
        return self._prune_support_tensor(state_list, log_b)

    def _update(
        self,
        log_bar_b: Dict[frozenset, float],
        log_segment_likelihoods: Dict[frozenset, float],
    ) -> Dict[frozenset, float]:
        """b_{t+1}(s) = P_θ(y_t|h_t,x,s) * b̄_{t+1}(s) / Z (dict API)."""
        state_list = list(log_bar_b.keys())
        log_bar_b_t = _to_tensor(log_bar_b, state_list, self.device)
        log_seg_t = _to_tensor(
            log_segment_likelihoods, state_list, self.device
        )
        state_list_new, log_b_t = self._update_tensor(
            state_list, log_bar_b_t, log_seg_t
        )
        return _to_dict(log_b_t, state_list_new)

    @torch.no_grad()
    def _compute_segment_log_likelihood(
        self,
        x: dict,
        passages: List,
        h_t: List[int],
        segment_tokens: List[int],
        state: frozenset,
        past_key_values=None,
    ) -> Tuple[float, object]:
        """
        log P_θ(y_t | h_t, x, s) = sum_j log P_θ(y_{t,j} | h_t, x, s, y_{t,<j}).
        Returns (log_likelihood, past_key_values_for_next).
        """
        packed = pack_passages_for_state(passages, state)
        log_lik = 0.0
        context = list(h_t)
        pkv = past_key_values

        for j, token in enumerate(segment_tokens):
            inputs = self.backend.prepare_input(x, context, packed, past_key_values=pkv)
            log_probs, pkv, _ = self.backend.forward(inputs, pkv)
            log_p = log_probs[0, token].item()
            log_lik += log_p
            context.append(token)

        return log_lik, pkv

    @torch.no_grad()
    def _compute_segment_log_likelihood_batched(
        self,
        x: dict,
        passages: List,
        h_t: List[int],
        segment_tokens: List[int],
        state_list: List[frozenset],
    ) -> torch.Tensor:
        """
        log P_θ(y_t | h_t, x, s_i) for all s_i in state_list.
        Returns 1D tensor of shape (len(state_list),).
        """
        n_states = len(state_list)
        if n_states == 0:
            return torch.tensor([], device=self.device)
        log_lik = torch.zeros(n_states, device=self.device, dtype=torch.float32)
        context = list(h_t)
        for j, token in enumerate(segment_tokens):
            packed_list = [
                pack_passages_for_state(passages, state_list[i])
                for i in range(n_states)
            ]
            batched_inputs = self.backend.prepare_batched_input(
                x, context, packed_list
            )
            log_probs, _, _ = self.backend.forward(batched_inputs, None)
            log_lik = log_lik + log_probs[:, token]
            context.append(token)
        return log_lik

    def _topk_states_from_belief(
        self,
        state_list_bar: List[frozenset],
        log_bar_b: torch.Tensor,
        k: int,
    ) -> Tuple[List[frozenset], torch.Tensor]:
        """
        TopK(b̄): return the top-k states by predictive belief (descending).
        Returns (top_k_states, their_log_beliefs) for use as sparse support S_t^(b).
        """
        if k <= 0 or len(state_list_bar) == 0:
            return [], torch.tensor([], device=self.device, dtype=torch.float32)
        n = min(k, log_bar_b.numel())
        top_vals, top_idx = torch.topk(log_bar_b, n, largest=True)
        top_states = [state_list_bar[i] for i in top_idx.cpu().tolist()]
        return top_states, top_vals

    def _select_states_from_belief(
        self,
        state_list_bar: List[frozenset],
        log_bar_b: torch.Tensor,
        mode: str,
        k: int,
        top_p: float,
    ) -> Tuple[List[frozenset], torch.Tensor]:
        """
        Select states for expansion: TopK, TopP (nucleus), or TopP_capped (nucleus capped at k).
        Returns (selected_states, their_log_beliefs) for use as sparse support S_t^(b).
        """
        if k <= 0 or len(state_list_bar) == 0:
            return [], torch.tensor([], device=self.device, dtype=torch.float32)
        if mode == "TopK":
            return self._topk_states_from_belief(state_list_bar, log_bar_b, k)
        # TopP or TopP_capped: nucleus by cumulative probability
        probs = torch.softmax(log_bar_b, dim=0)
        sorted_probs, sort_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        # TopP: consider all states; TopP_capped: consider only top-k by prob
        n_cap = min(k, log_bar_b.numel())
        n = n_cap if mode == "TopP_capped" else log_bar_b.numel()
        mask = cumsum[:n] >= top_p
        if mask.any():
            j = (mask.nonzero(as_tuple=True)[0][0].item()) + 1
        else:
            j = n
        if mode == "TopP_capped":
            j = min(j, n_cap)
        selected_idx = sort_idx[:j]
        selected_idx, _ = torch.sort(selected_idx)
        top_states = [state_list_bar[i] for i in selected_idx.cpu().tolist()]
        top_vals = log_bar_b[selected_idx]
        return top_states, top_vals

    def _sample_states_from_belief(
        self,
        state_list_bar: List[frozenset],
        log_bar_b: torch.Tensor,
        n: int,
    ) -> Tuple[List[frozenset], torch.Tensor]:
        """
        Sample n states from the distribution b̄ (probs from log_bar_b).
        Returns (sampled_state_list, log_bar_b_sampled) where sampled_state_list
        has length n (possibly with duplicates) and log_bar_b_sampled gives
        log b̄(s) for each sampled state in order.
        """
        if n <= 0 or len(state_list_bar) == 0:
            return [], torch.tensor([], device=self.device, dtype=torch.float32)
        log_probs = log_bar_b - torch.logsumexp(log_bar_b, dim=0)
        probs = torch.exp(log_probs)
        indices = torch.multinomial(probs, num_samples=min(n, len(state_list_bar)), replacement=True)
        sampled_states = [state_list_bar[i] for i in indices.cpu().tolist()]
        log_bar_b_sampled = log_probs[indices]
        return sampled_states, log_bar_b_sampled

    @torch.no_grad()
    def _generate_segment_greedy(
        self,
        x: dict,
        passages: List,
        h_t: List[int],
        state: frozenset,
        segment_size: int,
        past_key_values=None,
    ) -> Tuple[List[int], float, Any]:
        """
        Generate a segment of up to segment_size tokens greedily conditioned on (h_t, x, s).
        Returns (segment_tokens, log_likelihood, past_key_values_for_next).
        Stops early if a stop token is generated.
        """
        packed = pack_passages_for_state(passages, state)
        segment_tokens: List[int] = []
        log_lik = 0.0
        context = list(h_t)
        pkv = past_key_values

        for _ in range(segment_size):
            inputs = self.backend.prepare_input(x, context, packed, past_key_values=pkv)
            log_probs, pkv, _ = self.backend.forward(inputs, pkv)
            token_idx = torch.argmax(log_probs[0]).item()
            log_p = log_probs[0, token_idx].item()
            log_lik += log_p
            segment_tokens.append(token_idx)
            context.append(token_idx)
            if self.backend.is_stop_token(token_idx):
                break

        return segment_tokens, log_lik, pkv

    @torch.no_grad()
    def _decode_mixture(
        self,
        x: dict,
        passages: List,
        generated_tokens: List[int],
        log_bar_b: Dict[frozenset, float],
        past_key_values_per_state: Optional[Dict[frozenset, object]] = None,
        past_key_values_batched: Optional[object] = None,
    ) -> Tuple[
        int, float, Optional[Dict[frozenset, object]], Optional[object]
    ]:
        """
        Mixture: P(y|h,x) = sum_s P_θ(y|h,x,s) b̄(s).
        When past_key_values_batched is provided (and support unchanged), use
        batched cache and skip building per-state caches; returns (..., None, new_batched_pkv).
        """
        states = list(log_bar_b.keys())
        if not states:
            raise ValueError("Empty support in mixture decode")

        packed_list = [
            pack_passages_for_state(passages, s) for s in states
        ]
        batched_inputs = self.backend.prepare_batched_input(
            x, generated_tokens, packed_list
        )

        use_batched_cache = past_key_values_batched is not None
        if use_batched_cache:
            pkv = past_key_values_batched
            log_probs, pkv, _ = self.backend.forward(batched_inputs, pkv)
        elif past_key_values_per_state is not None:
            num_batch = len(states)
            num_layers = cache_compat._cache_num_layers(
                list(past_key_values_per_state.values())[0]
            )
            pkv_list = list(past_key_values_per_state.values())
            if len(pkv_list) != num_batch:
                raise ValueError("Past KV cache count mismatch")
            pkv = _make_dynamic_cache(self._num_hidden_layers)
            for i in range(num_layers):
                k_all = torch.cat(
                    [cache_compat._cache_get_layer(c, i)[0] for c in pkv_list],
                    dim=0,
                )
                v_all = torch.cat(
                    [cache_compat._cache_get_layer(c, i)[1] for c in pkv_list],
                    dim=0,
                )
                cache_compat._cache_append_layer(pkv, k_all, v_all)
            log_probs, pkv, _ = self.backend.forward(batched_inputs, pkv)
        else:
            log_probs, pkv, _ = self.backend.forward(batched_inputs, None)

        log_belief = torch.tensor(
            [log_bar_b[s] for s in states], device=self.device
        )
        log_mixture = log_probs + log_belief[:, None]
        log_mixture = torch.logsumexp(log_mixture, dim=0)
        token_idx = torch.argmax(log_mixture).item()
        log_token_llk = log_mixture[token_idx].item()

        num_batch = len(states)
        new_token = torch.full(
            (num_batch, 1),
            token_idx,
            device=self.device,
            dtype=batched_inputs["input_ids"].dtype,
        )
        new_attn = torch.ones_like(new_token)
        batched_inputs["input_ids"] = new_token
        batched_inputs["attention_mask"] = new_attn
        if "pixel_values" in batched_inputs:
            del batched_inputs["pixel_values"]
        if "image_grid_thw" in batched_inputs:
            del batched_inputs["image_grid_thw"]
        _, pkv, _ = self.backend.forward(batched_inputs, pkv)

        if use_batched_cache:
            return token_idx, log_token_llk, None, pkv
        pkv_per_state = {}
        for i, s in enumerate(states):
            pkv_s = _make_dynamic_cache(self._num_hidden_layers)
            for j in range(cache_compat._cache_num_layers(pkv)):
                k, v = cache_compat._cache_get_layer(pkv, j)
                cache_compat._cache_append_layer(
                    pkv_s, k[i : i + 1], v[i : i + 1]
                )
            pkv_per_state[s] = pkv_s
        return token_idx, log_token_llk, pkv_per_state, pkv

    @torch.no_grad()
    def _decode_given_state(
        self,
        x: dict,
        passages: List,
        generated_tokens: List[int],
        state: frozenset,
        past_key_values=None,
    ) -> Tuple[int, float, object]:
        """Decode one token conditioned on a fixed state s (state stays same for the whole segment)."""
        packed = pack_passages_for_state(passages, state)
        inputs = self.backend.prepare_input(x, generated_tokens, packed, past_key_values=past_key_values)
        log_probs, pkv, _ = self.backend.forward(inputs, past_key_values)
        token_idx = torch.argmax(log_probs[0]).item()
        log_token_llk = log_probs[0, token_idx].item()
        return token_idx, log_token_llk, pkv

    def _sample_one_state_from_belief(
        self, log_bar_b: Dict[frozenset, float]
    ) -> frozenset:
        """Sample one state from the belief distribution b̄."""
        state_list = list(log_bar_b.keys())
        if not state_list:
            raise ValueError("Empty support when sampling state")
        log_b_tensor = torch.tensor(
            [log_bar_b[s] for s in state_list],
            device=self.device,
            dtype=torch.float32,
        )
        probs = torch.softmax(log_b_tensor, dim=0)
        idx = torch.multinomial(probs, num_samples=1).item()
        return state_list[idx]

    def _debug_print_beams(
        self,
        segment_idx: int,
        beams: List[Dict[str, Any]],
        passages: Optional[List] = None,
        needle: Optional[str] = None,
        gold_chunks_per_needle: Optional[List[int]] = None,
    ) -> None:
        """Print progress at each beam update: segment index, each beam's segment tokens, score, belief (state indices and weights), and ground-truth chunks (by exact needle match or gold_chunks_per_needle)."""
        # Ground-truth chunk indices: prefer explicit gold_chunks_per_needle (one chunk index per needle), else derive from single needle
        if gold_chunks_per_needle is not None:
            print(f"[ESF segment {segment_idx}] ground_truth_chunks_per_needle (gold): {gold_chunks_per_needle}")
        elif needle is not None and passages is not None and needle.strip():
            gt_chunks = [i for i, p in enumerate(passages) if needle.strip() in (p or "")]
            print(f"[ESF segment {segment_idx}] ground_truth_chunks (needle exact match): {gt_chunks}")
        print(f"[ESF segment {segment_idx}] beams (score, segment_tokens, segment_text):")
        for i, b in enumerate(beams):
            seg = b.get("last_segment_tokens", [])
            try:
                seg_text = self.backend.tokenizer.decode(seg, skip_special_tokens=True)
            except Exception:
                seg_text = "<decode failed>"
            seg_state = b.get("segment_state")
            seg_state_str = list(seg_state) if seg_state is not None else None
            print(f"  beam {i}: alpha={b['alpha']:.4f}  len_h={len(b['h_t'])}  segment_conditioned_on={seg_state_str}  segment_tokens={seg[:20]}{'...' if len(seg) > 20 else ''}  segment_text={repr(seg_text[:80])}{'...' if len(seg_text) > 80 else ''}")
            # Belief b̄: state indices and corresponding belief weights
            state_list_bar_pred = b.get("state_list_bar_pred")
            log_bar_b_pred = b.get("log_bar_b_pred")
            if state_list_bar_pred is not None and log_bar_b_pred is not None and len(state_list_bar_pred) > 0:
                log_vals = [float(x) if hasattr(x, "item") else float(x) for x in log_bar_b_pred]
                log_b_arr = torch.tensor(log_vals, device=self.device)
                probs = torch.softmax(log_b_arr, dim=0).cpu().tolist()
                map_idx = int(torch.argmax(log_b_arr).item())
                map_prob = probs[map_idx]
                state_indices = [list(s) for s in state_list_bar_pred]
                print(f"       b̄: state_indices={state_indices}  weights={[f'{p:.3f}' for p in probs]}  argmax_idx={map_idx} (state={state_indices[map_idx]})  max_weight={map_prob:.3f}")
        print()

    def _write_beam_log(
        self,
        segment_idx: int,
        beams: List[Dict[str, Any]],
        log_dir: str,
        run_id: Optional[str] = None,
        results_dir: Optional[str] = None,
        results_stem: Optional[str] = None,
        gold_chunks_per_needle: Optional[List[int]] = None,
    ) -> None:
        """Write per-segment beam state summaries to log_dir (JSONL), or under results_dir with name {results_stem}_results.beam_log.jsonl (same as result file with .beam_log suffix) when given."""
        if results_dir and results_stem:
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{results_stem}_results.beam_log.jsonl")
            # First segment of run: truncate file so we don't append to a previous run
            if segment_idx == 0:
                if self.debug:
                    print(f"[beam log] saving to {path}")
                with open(path, "w", encoding="utf-8"):
                    pass
        elif log_dir:
            os.makedirs(log_dir, exist_ok=True)
            suffix = f"_{run_id}" if run_id else ""
            path = os.path.join(log_dir, f"beam_log{suffix}.jsonl")
            if segment_idx == 0 and self.debug:
                print(f"[beam log] saving to {path}")
        else:
            return
        entries = []
        for i, b in enumerate(beams):
            entry = {
                "segment": segment_idx,
                "beam_idx": i,
                "alpha": b["alpha"],
                "len_h_t": len(b["h_t"]),
                "state_list": [list(s) for s in b["state_list"]],
                "finished": b["finished"],
            }
            # State the current segment was conditioned on
            seg_state = b.get("segment_state")
            if seg_state is not None:
                entry["segment_state"] = list(seg_state)
            # Generated segment text and full text so far (for reconstruction)
            seg_tokens = b.get("last_segment_tokens", [])
            h_t = b.get("h_t", [])
            try:
                entry["segment_text"] = self.backend.tokenizer.decode(seg_tokens, skip_special_tokens=True)
            except Exception:
                entry["segment_text"] = ""
            try:
                entry["generated_text"] = self.backend.tokenizer.decode(h_t, skip_special_tokens=True)
            except Exception:
                entry["generated_text"] = ""
            # Predictive belief b̄ (per-state log prob) for visualization
            state_list_bar_pred = b.get("state_list_bar_pred")
            log_bar_b_pred = b.get("log_bar_b_pred")
            if state_list_bar_pred is not None and log_bar_b_pred is not None:
                entry["state_list_bar"] = [list(s) for s in state_list_bar_pred]
                # log_bar_b_pred: list of floats in same order as state_list_bar
                if isinstance(log_bar_b_pred, (list, tuple)):
                    entry["log_bar_b"] = [float(x) for x in log_bar_b_pred]
                else:
                    entry["log_bar_b"] = [
                        log_bar_b_pred[j].item() if hasattr(log_bar_b_pred[j], "item") else float(log_bar_b_pred[j])
                        for j in range(len(state_list_bar_pred))
                    ]
            if gold_chunks_per_needle is not None:
                entry["gold_chunks_per_needle"] = gold_chunks_per_needle
            entries.append(entry)
        with open(path, "a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def _generate_segment_beam(
        self,
        x: dict,
        passages: List,
        max_new_tokens: int,
        log_dir_override: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[int], List[float], List[Dict[frozenset, float]]]:
        """
        Global beam search over segment hypotheses. Each beam carries (h_t, b_t, alpha).
        Returns (generated_tokens_best, log_token_llks_best, segment_beliefs_best).
        """
        K = len(passages)
        if K == 0:
            return [], [], []

        log_dir = log_dir_override if log_dir_override is not None else self.log_dir
        state_list_0 = _initial_support(K)
        log_b_0 = torch.full(
            (len(state_list_0),),
            math.log(1.0 / len(state_list_0)),
            device=self.device,
            dtype=torch.float32,
        )

        beams: List[Dict[str, Any]] = [
            {
                "h_t": [],
                "state_list": state_list_0,
                "log_b": log_b_0,
                "alpha": 0.0,
                "log_segment_likelihoods_prev": None,
                "finished": False,
                "segment_beliefs_list": [_to_dict(log_b_0, state_list_0)],
                "segment_log_liks": [],
            }
        ]

        segment_idx = 0
        while True:
            # Check exit: all finished or max length
            if all(b["finished"] for b in beams):
                break
            best_len = max(len(b["h_t"]) for b in beams)
            if best_len >= max_new_tokens:
                break

            children: List[Dict[str, Any]] = []

            for beam in beams:
                if beam["finished"]:
                    continue
                h_t = beam["h_t"]
                state_list = beam["state_list"]
                log_b = beam["log_b"]
                alpha = beam["alpha"]
                log_seg_prev = beam["log_segment_likelihoods_prev"]
                seg_beliefs = beam["segment_beliefs_list"]
                seg_log_liks = beam["segment_log_liks"]

                # Predict and store (state_list_bar, log_bar_b)
                state_list_bar, log_bar_b = self._predict(
                    state_list, log_b, log_seg_prev, K, segment_idx=segment_idx
                )

                # S_t^(b) = state selection (TopK / TopP / TopP_capped) by predictive belief b̄
                k_explore = min(K, self.state_explore_TopK)
                topk_states, log_bar_b_topk = self._select_states_from_belief(
                    state_list_bar,
                    log_bar_b,
                    self.state_explore_mode,
                    k_explore,
                    self.top_p,
                )
                if not topk_states:
                    continue

                # Build list of log b̄(s) for top-k states (handle 0-dim / 1-dim tensor)
                if log_bar_b_topk.numel() == 1:
                    log_bar_b_list = [log_bar_b_topk.item()] * len(topk_states)
                else:
                    log_bar_b_list = log_bar_b_topk.cpu().tolist()

                # Generate segment per state (greedy) and score; record ℓ_t(y;s)
                for s, log_b_s in zip(topk_states, log_bar_b_list):
                    log_b_s_val = float(log_b_s)
                    segment_tokens, log_lik_y, _ = self._generate_segment_greedy(
                        x, passages, h_t, s, self.segment_size
                    )
                    if not segment_tokens:
                        continue
                    score = alpha + log_b_s_val + log_lik_y

                    # Segment log-likelihoods for all states in state_list_bar (for update)
                    log_segment_lik = self._compute_segment_log_likelihood_batched(
                        x, passages, h_t, segment_tokens, state_list_bar
                    )
                    state_list_new, log_b_new = self._update_tensor(
                        state_list_bar, log_bar_b, log_segment_lik
                    )
                    log_seg_prev_dict = _to_dict(log_segment_lik, state_list_bar)
                    finished = self.backend.is_stop_token(segment_tokens[-1])
                    child_belief = _to_dict(log_b_new, state_list_new)

                    # Store predictive belief b̄ for this child (for logging/visualization)
                    log_bar_b_list_serial = log_bar_b.cpu().tolist()
                    if not isinstance(log_bar_b_list_serial, list):
                        log_bar_b_list_serial = [log_bar_b_list_serial] * len(state_list_bar)

                    children.append(
                        {
                            "h_t": h_t + segment_tokens,
                            "state_list": state_list_new,
                            "log_b": log_b_new,
                            "alpha": score,
                            "log_segment_likelihoods_prev": log_seg_prev_dict,
                            "finished": finished,
                            "segment_beliefs_list": seg_beliefs + [child_belief],
                            "segment_log_liks": seg_log_liks + [log_lik_y],
                            "state_list_bar_pred": state_list_bar,
                            "log_bar_b_pred": log_bar_b_list_serial,
                            "last_segment_tokens": segment_tokens,
                            "segment_state": s,
                        }
                    )

            if not children:
                break

            # Diverse beam prune: best-from-each-group by conditioning chunks, then round-robin until beam_width
            groups = defaultdict(list)
            for c in children:
                k = len(c.get("segment_state") or frozenset())
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
            while len(top_b_list) < self.beam_width:
                added = False
                for k in group_keys:
                    if len(top_b_list) >= self.beam_width:
                        break
                    if idx < len(groups[k]):
                        top_b_list.append(groups[k][idx])
                        added = True
                if not added:
                    break
                idx += 1
            beams = top_b_list

            # Print and log all beams after every update (including final beams before early stop)
            if self.debug:
                self._debug_print_beams(
                    segment_idx, beams, passages=passages, needle=x.get("needle"),
                    gold_chunks_per_needle=x.get("gold_chunks_per_needle"),
                )

            self._write_beam_log(
                segment_idx,
                beams,
                log_dir or "",
                run_id=run_id,
                results_dir=x.get("results_dir"),
                results_stem=x.get("results_stem"),
                gold_chunks_per_needle=x.get("gold_chunks_per_needle"),
            )

            # Early stopping: exit if a completed beam has score >= best overall
            finished_beams = [b for b in beams if b["finished"]]
            if finished_beams:
                best_finished_score = max(b["alpha"] for b in finished_beams)
                best_overall_score = max(b["alpha"] for b in beams)
                if best_finished_score >= best_overall_score:
                    break

            segment_idx += 1

        # Best beam by alpha
        best = max(beams, key=lambda b: b["alpha"])
        generated_tokens = best["h_t"]
        segment_beliefs = best["segment_beliefs_list"]

        # Store all final beams (scores + posteriors) for Test Summary / logging
        tokenizer = getattr(self.backend, "tokenizer", None)
        self._last_completed_beams = []
        for i, b in enumerate(beams):
            state_list = b["state_list"]
            log_b = b["log_b"]
            if isinstance(log_b, torch.Tensor):
                probs = torch.softmax(log_b, dim=0).cpu().tolist()
            else:
                probs = list(log_b)
            state_reprs = [list(sorted(s)) for s in state_list]
            generated_text = ""
            if tokenizer:
                try:
                    generated_text = tokenizer.decode(b["h_t"], skip_special_tokens=True)
                except Exception:
                    pass
            self._last_completed_beams.append({
                "beam_idx": i,
                "alpha": float(b["alpha"]),
                "finished": b["finished"],
                "posterior": {"states": state_reprs, "probs": probs},
                "generated_text": generated_text,
            })

        # Per-token log probs: spread each segment log-lik over its tokens (infer segment lengths from segment_size)
        log_token_llks = []
        remaining = len(generated_tokens)
        for seg_log in best["segment_log_liks"]:
            L = min(self.segment_size, remaining) if remaining > 0 else 0
            if L <= 0:
                break
            log_per_token = seg_log / L
            log_token_llks.extend([log_per_token] * L)
            remaining -= L
        if len(log_token_llks) < len(generated_tokens):
            log_token_llks.extend([0.0] * (len(generated_tokens) - len(log_token_llks)))
        elif len(log_token_llks) > len(generated_tokens):
            log_token_llks = log_token_llks[: len(generated_tokens)]
        return generated_tokens, log_token_llks, segment_beliefs

    def generate(
        self,
        x: dict,
        passages: List,
        max_new_tokens: int = 512,
        log_dir_override: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[int], List[float], List[Dict[frozenset, float]]]:
        """
        Generate with ESF. Returns (generated_tokens, log_token_llks, segment_beliefs).
        When decode_mode is 'segment_beam', runs global beam search over segments.
        """
        K = len(passages)
        if K == 0:
            self._last_completed_beams = []
            return [], [], []

        self._last_completed_beams = []
        if self.decode_mode == "segment_beam":
            return self._generate_segment_beam(
                x, passages, max_new_tokens,
                log_dir_override=log_dir_override,
                run_id=run_id,
            )

        max_state_size = (
            self.max_state_size if self.max_state_size is not None else K
        )

        # b_0: initial belief over K states (one singleton per document)
        state_list = _initial_support(K)
        log_b = torch.full(
            (len(state_list),),
            math.log(1.0 / len(state_list)),
            device=self.device,
            dtype=torch.float32,
        )

        generated_tokens = []
        log_all_tokens_llk = []
        segment_beliefs = [_to_dict(log_b, state_list)]

        log_segment_likelihoods_prev: Optional[Dict[frozenset, float]] = None
        past_key_values_per_state: Optional[Dict[frozenset, object]] = None
        past_key_values_batched: Optional[object] = None
        last_decode_states: Optional[frozenset] = None
        past_key_values_map = None

        while len(generated_tokens) < max_new_tokens:
            # Predict: b̄_{t+1}(s') = sum_s T(s'|s) b_t(s). Kernel only after first segment.
            segment_idx_loop = len(segment_beliefs) - 1
            state_list_bar, log_bar_b = self._predict(
                state_list, log_b, log_segment_likelihoods_prev, K,
                segment_idx=segment_idx_loop,
            )
            log_bar_b_dict = _to_dict(log_bar_b, state_list_bar)

            # Sample one state at the beginning of each segment; use it for the whole segment
            if self.decode_mode != "mixture":
                segment_state = self._sample_one_state_from_belief(log_bar_b_dict)

            segment_tokens = []
            for _ in range(self.segment_size):
                if self.decode_mode == "mixture":
                    current_states = frozenset(state_list_bar)
                    use_batched = (
                        last_decode_states == current_states
                        and past_key_values_batched is not None
                    )
                    # Use per-state cache only when support is unchanged (avoids KV cache count mismatch)
                    if use_batched:
                        pkv_per_state_arg = None
                        pkv_batched_arg = past_key_values_batched
                    else:
                        pkv_batched_arg = None
                        if past_key_values_per_state is not None and set(
                            past_key_values_per_state.keys()
                        ) == current_states:
                            pkv_per_state_arg = past_key_values_per_state
                        else:
                            pkv_per_state_arg = None
                    token_idx, log_llk, pkv_per_state, batched_pkv = (
                        self._decode_mixture(
                            x,
                            passages,
                            generated_tokens + segment_tokens,
                            log_bar_b_dict,
                            past_key_values_per_state=pkv_per_state_arg,
                            past_key_values_batched=pkv_batched_arg,
                        )
                    )
                    last_decode_states = current_states
                    past_key_values_batched = batched_pkv
                    if pkv_per_state is not None:
                        past_key_values_per_state = pkv_per_state
                else:
                    # Decode one token conditioned on the same segment_state for the whole segment
                    token_idx, log_llk, pkv = self._decode_given_state(
                        x,
                        passages,
                        generated_tokens + segment_tokens,
                        segment_state,
                        past_key_values_map,
                    )
                    past_key_values_map = pkv

                segment_tokens.append(token_idx)
                generated_tokens.append(token_idx)
                log_all_tokens_llk.append(log_llk)

                if self.backend.is_stop_token(token_idx):
                    break

            if not segment_tokens:
                break

            # log P_θ(y_t | h_t, x, s) for each s in support
            h_t = generated_tokens[:-len(segment_tokens)]
            log_segment_lik = self._compute_segment_log_likelihood_batched(
                x, passages, h_t, segment_tokens, state_list_bar
            )
            log_segment_likelihoods_prev = _to_dict(
                log_segment_lik, state_list_bar
            )

            # Update: b_{t+1}(s) ∝ P_θ(y_t|h_t,x,s) b̄_{t+1}(s)
            state_list, log_b = self._update_tensor(
                state_list_bar, log_bar_b, log_segment_lik
            )
            segment_beliefs.append(_to_dict(log_b, state_list))

            if self.backend.is_stop_token(segment_tokens[-1]):
                break

        return generated_tokens, log_all_tokens_llk, segment_beliefs
