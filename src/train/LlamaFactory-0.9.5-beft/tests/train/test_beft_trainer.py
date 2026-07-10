# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
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

import pytest
import torch

from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.beft.loss import compute_beft_loss, get_answer_token_logps
from llamafactory.train.beft.trainer import compute_grouped_beft_loss
from llamafactory.train.beft.workflow import BeftDataCollator


def _labels(answer_positions: list[list[int]], seq_len: int = 6) -> torch.Tensor:
    labels = torch.full((len(answer_positions), seq_len), IGNORE_INDEX, dtype=torch.long)
    for row_idx, positions in enumerate(answer_positions):
        for offset, position in enumerate(positions):
            labels[row_idx, position] = (row_idx + offset) % 7

    return labels


def _expected_grouped_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_idx: torch.Tensor,
    prior_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []
    for group_idx in torch.unique(batch_idx, sorted=True):
        row_indices = torch.nonzero(batch_idx == group_idx, as_tuple=False).flatten()
        token_logps, _ = get_answer_token_logps(logits[row_indices], labels[row_indices])
        group_prior_logits = prior_logits[row_indices] if prior_logits is not None else None
        losses.append(compute_beft_loss(token_logps, group_prior_logits)[0])

    return torch.stack(losses).mean()


def test_grouped_beft_loss_keeps_instances_separate():
    torch.manual_seed(0)
    logits = torch.randn(4, 6, 7)
    labels = _labels([[3], [3], [3], [3]])
    batch_idx = torch.tensor([0, 0, 1, 1])
    prior_logits = torch.tensor([[1.5], [-0.5], [-1.0], [2.0]])

    grouped_loss = compute_grouped_beft_loss(logits, labels, batch_idx=batch_idx, prior_logits=prior_logits)
    expected_loss = _expected_grouped_loss(logits, labels, batch_idx, prior_logits)

    token_logps, _ = get_answer_token_logps(logits, labels)
    ungrouped_loss = compute_beft_loss(token_logps, prior_logits)[0]

    assert torch.allclose(grouped_loss, expected_loss)
    assert not torch.allclose(grouped_loss, ungrouped_loss)


def test_grouped_beft_loss_allows_different_answer_lengths_across_instances():
    torch.manual_seed(1)
    logits = torch.randn(5, 6, 7)
    labels = _labels([[3], [3], [2, 3], [2, 3], [2, 3]])
    batch_idx = torch.tensor([0, 0, 1, 1, 1])

    grouped_loss = compute_grouped_beft_loss(logits, labels, batch_idx=batch_idx)
    expected_loss = _expected_grouped_loss(logits, labels, batch_idx)

    assert torch.allclose(grouped_loss, expected_loss)


def test_grouped_beft_loss_rejects_different_answer_lengths_within_instance():
    torch.manual_seed(2)
    logits = torch.randn(2, 6, 7)
    labels = _labels([[3], [2, 3]])
    batch_idx = torch.tensor([0, 0])

    with pytest.raises(ValueError, match="equal answer lengths"):
        compute_grouped_beft_loss(logits, labels, batch_idx=batch_idx)


def test_grouped_beft_prior_targets_follow_flattened_rows():
    torch.manual_seed(3)
    logits = torch.randn(4, 6, 7)
    labels = _labels([[3], [3], [3], [3]])
    batch_idx = torch.tensor([0, 0, 1, 1])
    prior_logits = torch.tensor([[3.0], [-3.0], [-3.0], [3.0]])
    is_gt_passage = torch.tensor([1, 0, 0, 1])
    prior_loss_fn = torch.nn.BCEWithLogitsLoss()
    prior_loss_factor = 0.5

    grouped_loss = compute_grouped_beft_loss(
        logits,
        labels,
        batch_idx=batch_idx,
        prior_logits=prior_logits,
        is_gt_passage=is_gt_passage,
        prior_loss_fn=prior_loss_fn,
        use_prior_head_loss=True,
        prior_loss_factor=prior_loss_factor,
    )

    expected_losses = []
    for group_idx in torch.unique(batch_idx, sorted=True):
        row_indices = torch.nonzero(batch_idx == group_idx, as_tuple=False).flatten()
        token_logps, _ = get_answer_token_logps(logits[row_indices], labels[row_indices])
        beft_loss = compute_beft_loss(token_logps, prior_logits[row_indices])[0]
        targets = is_gt_passage[row_indices].to(dtype=prior_logits.dtype).view_as(prior_logits[row_indices])
        prior_loss = prior_loss_fn(prior_logits[row_indices], targets) * prior_loss_factor
        expected_losses.append(beft_loss + prior_loss)

    assert torch.allclose(grouped_loss, torch.stack(expected_losses).mean())


def test_beft_data_collator_flattens_instances_with_batch_idx(monkeypatch):
    captured_features = []

    def fake_mm_call(self, features):
        captured_features.extend(features)
        return {
            "input_ids": torch.tensor([feature["input_ids"] for feature in features]),
            "attention_mask": torch.tensor([feature["attention_mask"] for feature in features]),
            "labels": torch.tensor([feature["labels"] for feature in features]),
        }

    monkeypatch.setattr(MultiModalDataCollatorForSeq2Seq, "__call__", fake_mm_call)
    collator = BeftDataCollator.__new__(BeftDataCollator)
    features = [
        {
            "all_input_ids": [[1, 2], [3, 4]],
            "all_attention_mask": [[1, 1], [1, 1]],
            "all_labels": [[IGNORE_INDEX, 2], [IGNORE_INDEX, 4]],
            "all_passage_images": [["img-a"], []],
            "gt_passage_idx": [1],
            "videos": ["video-a"],
            "audios": [],
        },
        {
            "all_input_ids": [[5, 6], [7, 8], [9, 10]],
            "all_attention_mask": [[1, 1], [1, 1], [1, 1]],
            "all_labels": [[IGNORE_INDEX, 6], [IGNORE_INDEX, 8], [IGNORE_INDEX, 10]],
            "all_passage_images": [[], ["img-b"], []],
            "gt_passage_idx": [0, 2],
            "videos": [],
            "audios": ["audio-b"],
        },
    ]

    batch = BeftDataCollator.__call__(collator, features)

    assert batch["batch_idx"].tolist() == [0, 0, 1, 1, 1]
    assert batch["is_gt_passage"].tolist() == [0, 1, 1, 0, 1]
    assert batch["input_ids"].shape[0] == 5
    assert [feature["images"] for feature in captured_features] == [["img-a"], [], [], ["img-b"], []]
    assert [feature["videos"] for feature in captured_features] == [["video-a"], ["video-a"], [], [], []]
    assert [feature["audios"] for feature in captured_features] == [[], [], ["audio-b"], ["audio-b"], ["audio-b"]]
