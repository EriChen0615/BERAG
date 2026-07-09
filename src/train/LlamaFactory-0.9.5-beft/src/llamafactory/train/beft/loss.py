# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ...extras.constants import IGNORE_INDEX


def get_answer_token_logps(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    label_pad_token_id: int = IGNORE_INDEX,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    r"""Extract answer-token log-probabilities from causal LM logits.

    Args:
        logits: Float tensor of shape [K, L, V].
        labels: Long tensor of shape [K, L]. Non-answer tokens must be masked
            with `label_pad_token_id`. BEFT expects all K rows to supervise the
            same answer tokens.

    Returns:
        token_logps: Float tensor of shape [K, T].
        answer_lengths: Long tensor of shape [K].
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} is incompatible with labels {tuple(labels.shape)}.")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].clone()
    loss_mask = shifted_labels != label_pad_token_id

    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    all_token_logps = torch.gather(
        shifted_logits.log_softmax(dim=-1),
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    answer_lengths = loss_mask.sum(dim=-1)
    if not torch.all(answer_lengths == answer_lengths[0]):
        raise ValueError(f"BEFT expects equal answer lengths across passages, got {answer_lengths.tolist()}.")

    token_logps = torch.stack(
        [row_logps[row_mask] for row_logps, row_mask in zip(all_token_logps, loss_mask, strict=True)],
        dim=0,
    )
    return token_logps, answer_lengths


def compute_beft_loss(
    token_logps: "torch.Tensor",
    prior_logits: "torch.Tensor | None" = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    r"""Compute the BEFT marginalized next-token loss.

    `token_logps[k, i]` is log p_theta(y_i | x, z_k, y_<i). BEFT first
    computes the passage posterior before y_i from the answer prefix and then
    marginalizes the current token over passages. If `prior_logits` is omitted,
    zeros are used as a uniform log-prior up to an additive constant; the
    constant cancels in the posterior normalization.
    """
    if token_logps.ndim != 2:
        raise ValueError(f"token_logps must have shape [K, T], got {tuple(token_logps.shape)}.")

    num_passages = token_logps.size(0)
    if prior_logits is None:
        prior_logprobs = token_logps.new_zeros(num_passages)
    else:
        prior_logprobs = torch.log_softmax(prior_logits.view(num_passages), dim=0)

    prefix_logps = torch.cumsum(token_logps, dim=1) - token_logps
    posterior_scores = prefix_logps + prior_logprobs[:, None]
    posterior_logprobs = posterior_scores - torch.logsumexp(posterior_scores, dim=0, keepdim=True)

    token_marginal_logprobs = torch.logsumexp(token_logps + posterior_logprobs, dim=0)
    loss = -token_marginal_logprobs.sum()
    return loss, posterior_logprobs, prior_logprobs


def get_hidden_state_before_first_label(
    hidden_states: "torch.Tensor",
    labels: "torch.Tensor",
    hidden_state_offset: int = 0,
    label_pad_token_id: int = IGNORE_INDEX,
) -> "torch.Tensor":
    r"""Return the hidden state immediately before the first supervised answer token."""
    label_mask = labels != label_pad_token_id
    if not torch.all(label_mask.any(dim=1)):
        raise ValueError("Every BEFT passage row must contain at least one supervised answer token.")

    first_label_indices = label_mask.to(torch.long).argmax(dim=1)
    hidden_indices = first_label_indices - hidden_state_offset - 1
    if torch.any(hidden_indices < 0):
        raise ValueError("`beft_hidden_state_offset` places the prior hidden state before the sequence start.")

    batch_indices = torch.arange(labels.size(0), device=labels.device)
    return hidden_states[batch_indices, hidden_indices, :]
