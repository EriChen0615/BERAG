# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)

EVIDENCE_PLACEHOLDER = "<<<EVIDENCE>>>"


def _normalize_gt_passage_idx(gt_passage_idx: Any, num_passages: int) -> list[int]:
    if gt_passage_idx is None:
        return []

    raw_indices = gt_passage_idx if isinstance(gt_passage_idx, list) else [gt_passage_idx]
    normalized_indices = []
    for raw_idx in raw_indices:
        idx = int(raw_idx)
        if idx == -1:
            continue

        if 0 <= idx < num_passages:
            normalized_indices.append(idx)
        else:
            logger.warning_rank0(f"Dropped invalid gt_passage_idx {idx} for {num_passages} passages.")

    return normalized_indices


def _get_passage_text(passage: Any, passage_idx: int) -> str:
    if isinstance(passage, dict):
        if "text" not in passage:
            raise ValueError(f"Passage {passage_idx} is missing the `text` field.")

        passage_text = passage["text"]
    else:
        passage_text = passage

    return "" if passage_text is None else str(passage_text)


def _get_passage_images(passage: Any) -> list["ImageInput"]:
    if not isinstance(passage, dict):
        return []

    passage_images = passage.get("images")
    if passage_images is None:
        return []

    return list(passage_images) if isinstance(passage_images, list) else [passage_images]


@dataclass
class BeftDatasetProcessor(DatasetProcessor):
    def _expand_prompt_with_passage(self, prompt: list[dict[str, str]], passage_text: str) -> list[dict[str, str]]:
        expanded_prompt = deepcopy(prompt)
        prompt_content = expanded_prompt[-1]["content"]
        if EVIDENCE_PLACEHOLDER not in prompt_content:
            raise ValueError(f"BEFT prompt must contain `{EVIDENCE_PLACEHOLDER}`.")

        expanded_prompt[-1]["content"] = prompt_content.replace(EVIDENCE_PLACEHOLDER, passage_text)
        return expanded_prompt

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        discarding_history_cot = self.data_args.mask_history and not self.template.preserve_thinking
        encoded_pairs = self.template.encode_multiturn(
            self.tokenizer, messages, system, tools, discarding_history_cot
        )
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            passages = examples["_passages"][i]
            if not passages:
                logger.warning_rank0(f"Dropped BEFT example without passages: {examples['_prompt'][i]}.")
                continue

            all_input_ids, all_attention_mask, all_labels, all_passage_images = [], [], [], []
            for passage_idx, passage in enumerate(passages):
                passage_text = _get_passage_text(passage, passage_idx)
                expanded_prompt = self._expand_prompt_with_passage(examples["_prompt"][i], passage_text)
                passage_images = list(examples["_images"][i] or []) + _get_passage_images(passage)
                input_ids, labels = self._encode_data_example(
                    prompt=expanded_prompt,
                    response=examples["_response"][i],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=passage_images,
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                )

                all_input_ids.append(input_ids)
                all_attention_mask.append([1] * len(input_ids))
                all_labels.append(labels)
                all_passage_images.append(passage_images)

            model_inputs["all_input_ids"].append(all_input_ids)
            model_inputs["all_attention_mask"].append(all_attention_mask)
            model_inputs["all_labels"].append(all_labels)
            model_inputs["all_passage_images"].append(all_passage_images)
            model_inputs["gt_passage_idx"].append(
                _normalize_gt_passage_idx(examples["_gt_passage_idx"][i], len(passages))
            )
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["all_labels"][0]))
        print("input_ids:\n{}".format(example["all_input_ids"][0]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["all_input_ids"][0], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["all_labels"][0]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
        print(f"gt_passage_idx:\n{example['gt_passage_idx']}")
