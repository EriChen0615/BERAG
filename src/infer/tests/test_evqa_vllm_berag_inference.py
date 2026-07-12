from argparse import Namespace
import sys
import types

import torch

from beft_prior_head import build_beft_prior_head
from infer.beft_prior import BeftPriorHead
from infer.evqa_vllm_berag_inference import (
    DEFAULT_PROMPT_TEMPLATE,
    EMPTY_DOCUMENT_PASSAGE_ID,
    EMPTY_DOCUMENT_TEXT,
    PreparedExample,
    build_output_row,
    make_berag_user_prompt_with_sentinel,
    make_evidence_document,
    make_llm,
    make_rag_user_prompt,
    render_chat_prompt,
    select_retrieved_passages,
    split_rendered_berag_prompt,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        content = messages[-1]["content"]
        if isinstance(content, list):
            text = "<image>" + "".join(part.get("text", "") for part in content if part.get("type") == "text")
        else:
            text = content
        return f"<chat>{text}<assistant>" if add_generation_prompt else f"<chat>{text}"


def test_rag_and_berag_prompt_parts_share_components():
    tokenizer = FakeTokenizer()
    docs = ["Title: p1\tContent: alpha\n", "Title: p2\tContent: beta\n"]
    question = "What is shown?"

    rag_user_prompt = make_rag_user_prompt(DEFAULT_PROMPT_TEMPLATE, docs, question)
    berag_user_prompt = make_berag_user_prompt_with_sentinel(DEFAULT_PROMPT_TEMPLATE, question)
    shared_prefix, suffix = split_rendered_berag_prompt(tokenizer, DEFAULT_PROMPT_TEMPLATE, question, include_image=True)
    rag_rendered = render_chat_prompt(tokenizer, rag_user_prompt, include_image=True)

    assert "<<<EVIDENCE>>>" in berag_user_prompt
    assert "<<<EVIDENCE>>>" not in rag_rendered
    assert rag_rendered == f"{shared_prefix}{' '.join(docs)}{suffix}"
    assert "[QUESTION] What is shown?" in suffix
    assert "[ANSWER]" in suffix


def test_evidence_documents_preserve_dataset_order():
    pid_to_content = {"p1": "one two three", "p2": "four five six"}
    retrieved = [{"passage_id": "p2"}, {"passage_id": "p1"}]

    docs = [make_evidence_document(item, pid_to_content, max_words_per_evidence=2) for item in retrieved]

    assert docs == ["Title: p2\tContent: four five\n", "Title: p1\tContent: one two\n"]


def test_k_zero_uses_empty_document_pseudo_passage():
    args = Namespace(
        retrieval_topk=0,
        retrieval_field="retrieved_passage",
        ensure_gt_passage_in_ensemble=True,
    )
    row = {
        "question_id": "q0",
        "retrieved_passage": [{"passage_id": "real", "passage_content": "real document"}],
        "pos_item_ids": ["gt"],
        "pos_item_contents": ["ground truth"],
    }

    retrieved = select_retrieved_passages(row, args)
    docs = [make_evidence_document(item, {}, max_words_per_evidence=2) for item in retrieved]

    assert retrieved == [
        {
            "passage_id": EMPTY_DOCUMENT_PASSAGE_ID,
            "passage_content": EMPTY_DOCUMENT_TEXT,
        }
    ]
    assert docs == [EMPTY_DOCUMENT_TEXT]


def test_build_output_row_records_berag_prior_posterior_telemetry():
    example = PreparedExample(
        question_id="q1",
        question="What color?",
        question_type="automatic",
        gold_answer="blue",
        answers=["blue"],
        image_path="image.jpg",
        image=None,
        documents=["doc0", "doc1", "doc2"],
        retrieved_passage_ids=["p0", "p1", "p2"],
        rag_request={"prompt": "rag"},
        berag_shared_prefix={"prompt": "prefix"},
        berag_suffix="suffix",
        metadata={
            "question_id": "q1",
            "question": "What color?",
            "question_type": "automatic",
            "gold_answer": "blue",
            "answers": ["blue"],
            "retrieval_field": "retrieved_passage",
            "retrieval_topk": 3,
            "retrieved_passage_ids": ["p0", "p1", "p2"],
            "gt_passage_id": "p1",
            "gt_passage_in_zidx": 1,
            "image_path": "image.jpg",
        },
    )
    output = types.SimpleNamespace(
        outputs=[types.SimpleNamespace(text="[ANSWER] blue")],
        berag_info={
            "num_branches": 3,
            "log_prior_by_branch": [-0.2, -1.7, -2.3],
            "log_posterior_by_branch": [-2.1, -0.3, None],
            "prior_max_branch_id": 0,
            "posterior_max_branch_id": 1,
            "prior_sorted_branch_ids": [0, 1, 2],
            "posterior_sorted_branch_ids": [1, 0],
            "active_branch_ids": [0, 1],
            "pruned_branch_ids": [2],
        },
    )

    row = build_output_row(example, Namespace(mode="berag"), output, "ok")

    assert row["response"] == "[ANSWER] blue"
    assert row["generated_answer"] == "blue"
    assert row["berag_log_prior"] == [-0.2, -1.7, -2.3]
    assert row["berag_log_posterior"] == [-2.1, -0.3, None]
    assert row["berag_prior_max_idx"] == 0
    assert row["berag_posterior_max_idx"] == 1
    assert row["berag_prior_sorted_passage_ids"] == ["p0", "p1", "p2"]
    assert row["berag_posterior_sorted_passage_ids"] == ["p1", "p0"]
    assert row["berag_prior_top_passage_id"] == "p0"
    assert row["berag_posterior_top_passage_id"] == "p1"
    assert row["prior_hit"] is False
    assert row["posterior_hit"] is True
    assert row["prior_passage_is_gt"] is False
    assert row["dominant_passage_is_gt"] is True


def test_shared_prior_builder_matches_vllm_wrapper_outputs():
    training_prior = build_beft_prior_head(
        hidden_size=4,
        prior_modeling="mlp_head",
        num_layers=2,
        proj_dim=3,
    )
    wrapper = BeftPriorHead(hidden_size=4, prior_modeling="mlp_head", num_layers=2, proj_dim=3)
    wrapper.load_state_dict(training_prior.state_dict())
    hidden_states = torch.randn(2, 4)

    assert list(training_prior.state_dict()) == ["0.weight", "0.bias", "2.weight", "2.bias"]
    assert torch.allclose(wrapper(hidden_states), training_prior(hidden_states))


def test_mlp_prior_loads_llamafactory_sequential_state_dict():
    prior = BeftPriorHead(hidden_size=4, prior_modeling="mlp_head", num_layers=2, proj_dim=3)
    state = {
        "0.weight": torch.ones(3, 4),
        "0.bias": torch.ones(3),
        "2.weight": torch.ones(1, 3),
        "2.bias": torch.ones(1),
    }

    prior.load_state_dict(state)
    out = prior(torch.ones(2, 4))

    assert out.shape == (2, 1)


def test_linear_prior_loads_llamafactory_linear_state_dict():
    prior = BeftPriorHead(hidden_size=4, prior_modeling="linear_head")
    state = {"weight": torch.ones(1, 4), "bias": torch.ones(1)}

    prior.load_state_dict(state)
    out = prior(torch.ones(2, 4))

    assert out.shape == (2, 1)
    assert torch.allclose(out, torch.full((2, 1), 5.0))


def _make_llm_test_args(mode: str) -> Namespace:
    return Namespace(
        model="test-model",
        tokenizer_path=None,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=128,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        enforce_eager=False,
        mode=mode,
        num_accumulator_rows=16,
        prior_mode="uniform",
        default_prior_token_offset=-4,
        prior_module_cls="unused",
        prior_head_path=None,
        prior_hidden_size=4,
        prior_modeling="mlp_head",
        prior_head_num_layers=2,
        prior_head_proj_dim=1024,
    )


def test_make_llm_disables_async_scheduling_for_rag_and_berag(monkeypatch):
    captured_kwargs = []

    class FakeLLM:
        def __init__(self, **kwargs):
            captured_kwargs.append(kwargs)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    make_llm(_make_llm_test_args("rag"))
    make_llm(_make_llm_test_args("berag"))

    assert [kwargs["async_scheduling"] for kwargs in captured_kwargs] == [False, False]
    assert [kwargs["disable_log_stats"] for kwargs in captured_kwargs] == [False, False]
