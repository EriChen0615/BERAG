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

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from typing_extensions import override

from ...extras import logging
from ..sft.trainer import CustomSeq2SeqTrainer
from .loss import compute_beft_loss, get_answer_token_logps, get_hidden_state_before_first_label


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


def _infer_hidden_size(model: "torch.nn.Module") -> int:
    config = getattr(model, "config", None)
    for candidate_config in (config, getattr(config, "text_config", None), getattr(config, "llm_config", None)):
        hidden_size = getattr(candidate_config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)

    raise ValueError("Cannot infer hidden size for BEFT prior head from model config.")


def _build_prior_head(finetuning_args: "FinetuningArguments", hidden_size: int) -> "nn.Module | None":
    if finetuning_args.beft_prior_modeling == "none":
        return None

    if finetuning_args.beft_prior_modeling == "linear_head":
        return nn.Linear(hidden_size, 1)

    if finetuning_args.beft_prior_modeling == "mlp_head":
        num_layers = max(1, finetuning_args.beft_prior_head_num_layers)
        proj_dim = finetuning_args.beft_prior_head_proj_dim or hidden_size
        layers = []
        input_dim = hidden_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, proj_dim))
            layers.append(nn.ReLU())
            input_dim = proj_dim

        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    raise ValueError(f"Unknown BEFT prior modeling type: {finetuning_args.beft_prior_modeling}.")


class CustomSeq2SeqBEFTTrainer(CustomSeq2SeqTrainer):
    r"""Seq2Seq trainer with BEFT marginalized next-token loss."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(finetuning_args=finetuning_args, processor=processor, **kwargs)
        self._beft_metrics: dict[str, list[float]] = defaultdict(list)
        self.prior_loss_fn = nn.BCEWithLogitsLoss()
        self.prior_head = _build_prior_head(finetuning_args, _infer_hidden_size(self.model))
        if self.prior_head is not None:
            self._load_prior_head()
            self._move_prior_head_to_model_device()
            self.prior_head = self.accelerator.prepare_model(self.prior_head)

    def _load_prior_head(self) -> None:
        if self.prior_head is None or self.finetuning_args.beft_prior_head_path is None:
            return

        state_dict = torch.load(self.finetuning_args.beft_prior_head_path, map_location="cpu")
        self.prior_head.load_state_dict(state_dict)
        logger.info_rank0(f"Loaded BEFT prior head from {self.finetuning_args.beft_prior_head_path}.")

    def _move_prior_head_to_model_device(self) -> None:
        if self.prior_head is None:
            return

        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return

        self.prior_head.to(device=first_param.device)
        if first_param.is_floating_point():
            self.prior_head.to(dtype=first_param.dtype)

    def _maybe_add_prior_head_to_optimizer(self) -> None:
        if self.prior_head is None or self.optimizer is None:
            return

        prior_head_params = [param for param in self.prior_head.parameters() if param.requires_grad]
        if len(prior_head_params) == 0:
            return

        if not hasattr(self.optimizer, "add_param_group"):
            logger.warning_rank0("Optimizer does not support adding BEFT prior head parameters.")
            return

        optimizer_param_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group.get("params", [])
        }
        prior_head_param_ids = {id(param) for param in prior_head_params}
        if prior_head_param_ids.issubset(optimizer_param_ids):
            return

        prior_head_lr = self.finetuning_args.beft_prior_head_lr or self.args.learning_rate
        self.optimizer.add_param_group(
            {
                "params": prior_head_params,
                "lr": prior_head_lr,
                "weight_decay": self.args.weight_decay,
            }
        )
        logger.info_rank0(f"Added BEFT prior head parameters to optimizer with lr={prior_head_lr}.")

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        optimizer = super().create_optimizer(*args, **kwargs)
        self._maybe_add_prior_head_to_optimizer()
        return optimizer

    def _record_metric(self, key: str, value: "torch.Tensor | float") -> None:
        if torch.is_tensor(value):
            value = value.detach().float().cpu().item()

        self._beft_metrics[key].append(float(value))

    def _record_beft_metrics(
        self,
        beft_loss: "torch.Tensor",
        prior_loss: "torch.Tensor",
        total_loss: "torch.Tensor",
        posterior_logprobs: "torch.Tensor",
        prior_logprobs: "torch.Tensor",
        is_gt_passage: "torch.Tensor | None",
    ) -> None:
        with torch.no_grad():
            num_passages = posterior_logprobs.size(0)
            if is_gt_passage is None:
                gt_mask = torch.zeros(num_passages, dtype=torch.bool, device=posterior_logprobs.device)
                gt_mask[0] = True
            else:
                gt_mask = is_gt_passage.to(device=posterior_logprobs.device).bool().view(num_passages)

            num_gt_docs = int(gt_mask.sum().item())
            normalized_prior_logprobs = torch.log_softmax(prior_logprobs, dim=0)
            prior_entropy = -(normalized_prior_logprobs.exp() * normalized_prior_logprobs).sum()
            posterior_entropy = -(posterior_logprobs.exp() * posterior_logprobs).sum(dim=0).mean()

            prior_acc = posterior_acc_last = posterior_acc_mean = posterior_logprobs.new_tensor(0.0)
            if num_gt_docs > 0:
                topk_indices = torch.topk(normalized_prior_logprobs, k=num_gt_docs, dim=0).indices
                prior_pred_mask = torch.zeros_like(gt_mask)
                prior_pred_mask[topk_indices] = True
                prior_acc = (prior_pred_mask == gt_mask).all().float()

                posterior_argmax = posterior_logprobs.argmax(dim=0)
                posterior_hits = gt_mask[posterior_argmax].float()
                posterior_acc_last = posterior_hits[-1]
                posterior_acc_mean = posterior_hits.mean()

            self._record_metric("beft_loss", beft_loss)
            self._record_metric("prior_loss", prior_loss)
            self._record_metric("total_loss", total_loss)
            self._record_metric("prior_acc", prior_acc)
            self._record_metric("posterior_acc_last", posterior_acc_last)
            self._record_metric("posterior_acc_mean", posterior_acc_mean)
            self._record_metric("prior_entropy", prior_entropy)
            self._record_metric("posterior_entropy_mean", posterior_entropy)
            self._record_metric("num_gt_docs", float(num_gt_docs))

    @override
    def compute_loss(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional["torch.Tensor"] = None,
        **kwargs,
    ):
        is_gt_passage = inputs.pop("is_gt_passage", None)
        labels = inputs["labels"]
        return_hidden_states = self.prior_head is not None
        outputs = model(
            **inputs,
            return_dict=True,
            use_cache=False,
            output_hidden_states=return_hidden_states,
        )
        token_logps, _ = get_answer_token_logps(outputs.logits, labels)

        prior_logits = None
        if self.prior_head is not None:
            hidden_at_pre_label = get_hidden_state_before_first_label(
                outputs.hidden_states[-1],
                labels,
                hidden_state_offset=self.finetuning_args.beft_hidden_state_offset,
            )
            prior_logits = self.prior_head(hidden_at_pre_label)

        beft_loss, posterior_logprobs, prior_logprobs = compute_beft_loss(token_logps, prior_logits)
        prior_loss = beft_loss.new_zeros(())
        if self.finetuning_args.beft_use_prior_head_loss and prior_logits is not None and is_gt_passage is not None:
            prior_targets = is_gt_passage.to(device=prior_logits.device, dtype=prior_logits.dtype).view_as(
                prior_logits
            )
            if torch.any(prior_targets > 0):
                prior_loss = (
                    self.prior_loss_fn(prior_logits, prior_targets) * self.finetuning_args.beft_prior_loss_factor
                )

        total_loss = beft_loss + prior_loss
        self._record_beft_metrics(beft_loss, prior_loss, total_loss, posterior_logprobs, prior_logprobs, is_gt_passage)
        return (total_loss, outputs) if return_outputs else total_loss

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        metrics = {}
        for key, values in self._beft_metrics.items():
            if len(values) > 0:
                metrics[key] = sum(values) / len(values)

        self._beft_metrics.clear()
        super().log({**logs, **metrics}, *args, **kwargs)

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if self.prior_head is None or not self.is_world_process_zero():
            return

        output_dir = output_dir or self.args.output_dir
        prior_head_path = os.path.join(output_dir, "prior_head.pt")
        prior_head = self.accelerator.unwrap_model(self.prior_head)
        torch.save(prior_head.state_dict(), prior_head_path)
        logger.info_rank0(f"Saved BEFT prior head to {prior_head_path}.")


__all__ = ["CustomSeq2SeqBEFTTrainer"]
