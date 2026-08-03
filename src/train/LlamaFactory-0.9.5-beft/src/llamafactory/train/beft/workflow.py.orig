# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from ...data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import CustomSeq2SeqBEFTTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


@dataclass
class BeftDataCollator(MultiModalDataCollatorForSeq2Seq):
    r"""Expand BEFT examples with K passages each into ordinary multimodal rows."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        expanded_features = []
        is_gt_passage = []
        batch_idx = []
        for instance_idx, feature in enumerate(features):
            num_passages = len(feature["all_input_ids"])
            gt_passage_idx = feature.get("gt_passage_idx") or []
            gt_passage_idx = gt_passage_idx if isinstance(gt_passage_idx, list) else [gt_passage_idx]
            gt_passage_idx_set = {int(idx) for idx in gt_passage_idx if int(idx) != -1}

            all_passage_images = feature.get("all_passage_images") or [[] for _ in range(num_passages)]
            for passage_idx, (input_ids, attention_mask, labels) in enumerate(
                zip(feature["all_input_ids"], feature["all_attention_mask"], feature["all_labels"], strict=True)
            ):
                passage_images = all_passage_images[passage_idx] if passage_idx < len(all_passage_images) else []
                expanded_features.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "images": passage_images,
                        "videos": feature.get("videos") or [],
                        "audios": feature.get("audios") or [],
                    }
                )
                is_gt_passage.append(int(passage_idx in gt_passage_idx_set))
                batch_idx.append(instance_idx)

        if len(expanded_features) == 0:
            raise ValueError("BEFT requires at least one passage row in a device batch.")

        batch = super().__call__(expanded_features)
        batch["is_gt_passage"] = torch.tensor(is_gt_passage, dtype=torch.long)
        batch["batch_idx"] = torch.tensor(batch_idx, dtype=torch.long)
        return batch


def run_beft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="beft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    data_collator = BeftDataCollator(
        template=template,
        model=model,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    trainer = CustomSeq2SeqBEFTTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        model_args=model_args,
        **dataset_module,
        **tokenizer_module,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
