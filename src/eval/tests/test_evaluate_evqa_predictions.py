from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_evqa_predictions import add_monitoring_metrics, build_evaluated_instances


def test_monitoring_metrics_count_k_zero_gold_as_miss():
    rows = [
        {
            "question_id": "q1",
            "gt_passage_id": "p2",
            "retrieved_passage_ids": ["p1", "p2"],
            "retrieval_topk": 2,
            "berag_prior_top_passage_id": "p1",
            "berag_posterior_top_passage_id": "p2",
            "berag_prior_sorted_passage_ids": ["p1", "p2"],
            "berag_posterior_sorted_passage_ids": ["p2", "p1"],
            "gt_passage_in_zidx": 1,
            "z_dominant_idx": 1,
        },
        {
            "question_id": "q0",
            "gt_passage_id": "p9",
            "retrieved_passage_ids": ["empty_document"],
            "retrieval_topk": 0,
            "berag_prior_top_passage_id": "empty_document",
            "berag_posterior_top_passage_id": "empty_document",
            "berag_prior_sorted_passage_ids": ["empty_document"],
            "berag_posterior_sorted_passage_ids": ["empty_document"],
            "gt_passage_in_zidx": -1,
            "z_dominant_idx": 0,
        },
    ]

    report = {}
    add_monitoring_metrics(report, rows, retrieval_topk=None)

    assert report["prior_hit_rate"] == 0.0
    assert report["prior_passage_hit_rate"] == 0.0
    assert report["posterior_hit_rate"] == 0.5
    assert report["posterior_passage_hit_rate"] == 0.5
    assert report["retrieval_hit_rate"] == 0.5
    assert report["prior_recall_at_k"] == {1: 0.0, 2: 0.5}
    assert report["posterior_recall_at_k"] == {1: 0.5, 2: 0.5}


def test_evaluated_instances_preserve_berag_telemetry():
    rows = [
        {
            "question_id": "q1",
            "question": "What color?",
            "gold_answer": "blue",
            "image_path": "image.jpg",
            "response": "[ANSWER] blue",
            "score": 1.0,
            "gt_passage_id": "p1",
            "gt_passage_in_zidx": 1,
            "retrieved_passage_ids": ["p0", "p1"],
            "berag_log_prior": [-0.3, -1.3],
            "berag_log_posterior": [-2.0, -0.1],
            "berag_prior_max_idx": 0,
            "berag_posterior_max_idx": 1,
            "berag_prior_sorted_indices": [0, 1],
            "berag_posterior_sorted_indices": [1, 0],
            "berag_prior_sorted_passage_ids": ["p0", "p1"],
            "berag_posterior_sorted_passage_ids": ["p1", "p0"],
            "berag_prior_top_passage_id": "p0",
            "berag_posterior_top_passage_id": "p1",
            "prior_hit": False,
            "posterior_hit": True,
        }
    ]

    compact = build_evaluated_instances(rows)[0]

    assert compact["question_id"] == "q1"
    assert compact["evqa_score"] == 1.0
    assert compact["berag_log_prior"] == [-0.3, -1.3]
    assert compact["berag_log_posterior"] == [-2.0, -0.1]
    assert compact["berag_posterior_top_passage_id"] == "p1"
    assert compact["prior_hit"] is False
    assert compact["posterior_hit"] is True
