from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_infoseek_predictions import add_recall_metrics


def test_recall_aggregates_section_probabilities_at_page_level():
    row = {
        "question_id": "q1",
        "gt_passage_id": "page-a",
        "explode_page_sections": True,
        "retrieved_page_ids": ["page-b", "page-a"],
        "retrieved_passage_ids": [
            "page-b::section::0",
            "page-a::section::0",
            "page-a::section::1",
        ],
        "berag_branch_parent_page_ids": ["page-a", "page-a", "page-b"],
        "berag_log_prior": [math.log(0.3), math.log(0.3), math.log(0.4)],
        "berag_log_posterior": [math.log(0.1), math.log(0.1), math.log(0.8)],
        "berag_prior_sorted_passage_ids": [
            "page-b::section::0",
            "page-a::section::0",
            "page-a::section::1",
        ],
        "berag_posterior_sorted_passage_ids": [
            "page-b::section::0",
            "page-a::section::0",
            "page-a::section::1",
        ],
    }

    report = {}
    add_recall_metrics(report, [row])

    assert row["berag_prior_sorted_page_ids"] == ["page-a", "page-b"]
    assert row["berag_posterior_sorted_page_ids"] == ["page-b", "page-a"]
    assert row["berag_prior_log_probability_by_page"]["page-a"] == pytest.approx(
        math.log(0.6)
    )
    assert row["prior_page_hit"] is True
    assert row["posterior_page_hit"] is False
    assert report["retrieval_recall_at_k"] == {"1": 0.0, "2": 1.0}
    assert report["prior_recall_at_k"] == {"1": 1.0, "2": 1.0}
    assert report["posterior_recall_at_k"] == {"1": 0.0, "2": 1.0}
    assert report["berag_recall_level"] == "page"
    assert (
        report["berag_page_probability_aggregation"]
        == "sum_section_probabilities"
    )


def test_page_aggregation_rejects_misaligned_branch_metadata():
    row = {
        "question_id": "q1",
        "gt_passage_id": "page-a",
        "berag_branch_parent_page_ids": ["page-a"],
        "berag_log_prior": [math.log(0.5), math.log(0.5)],
    }

    with pytest.raises(ValueError, match="not aligned"):
        add_recall_metrics({}, [row])
