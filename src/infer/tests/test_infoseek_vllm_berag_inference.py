from argparse import Namespace

import pytest

from curate.prepare_infoseek_m2kr_retrieval_dataset import (
    build_retrieval_indexes,
    convert_rows,
)
from infer.evqa_vllm_berag_inference import (
    DEFAULT_PROMPT_TEMPLATE,
    build_output_row,
    effective_prior_token_offset,
    resolve_prior_token_offset,
    select_retrieved_passages,
)
from infer.infoseek_vllm_berag_inference import (
    PSEUDO_DOCUMENT_PASSAGE_ID,
    PSEUDO_DOCUMENT_TEXT,
    prepare_infoseek_example,
)


class FakeTokenizer:
    def __init__(self):
        self.tokenize_calls = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    ):
        assert not (add_generation_prompt and continue_final_message)
        rendered = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                text = "<image>" + "".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )
            else:
                text = content
            rendered.append(f"<{message['role']}>{text}")

        if add_generation_prompt:
            rendered.append("<assistant>")
        if continue_final_message:
            assert messages[-1]["role"] == "assistant"
        rendered_text = "".join(rendered)
        if not tokenize:
            return rendered_text

        self.tokenize_calls += 1
        tokenizable_text = rendered_text.replace("[ANSWER]", "\x01\x02\x03\x04")
        return [ord(char) for char in tokenizable_text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def _retrieval(question_id, image_id, question, hits, gold_url="https://wiki/gold"):
    return {
        "question_id": question_id,
        "image_id": image_id,
        "question": question,
        "answer": "answer",
        "ground_truth_url": gold_url,
        "retrieved_pages": hits,
    }


def _hit(index, url, score=1.0):
    return {
        "rank": index + 1,
        "url": url,
        "title": f"title-{index}",
        "score": score,
        "knowledge_base_index": index,
    }


def test_conversion_selects_hf_rows_and_preserves_hf_order():
    kb = [
        {"url": "https://wiki/zero", "title": "Zero"},
        {"url": "https://wiki/one", "title": "One"},
    ]
    retrieval_rows = [
        _retrieval("q1", "img1", "question one", [_hit(0, "https://wiki/zero")]),
        _retrieval("q2", "img2", "question two", [_hit(1, "https://wiki/one")]),
        _retrieval("outside", "img3", "outside", [_hit(0, "https://wiki/zero")]),
    ]
    metadata_rows = [
        {
            "question_id": "q2",
            "img_path": "nested/img2.jpg",
            "question": "question two",
            "gold_answer": "two",
            "answers": ["two"],
        },
        {
            "question_id": "q1",
            "img_path": "nested/img1.jpg",
            "question": "question one",
            "gold_answer": "one",
            "answers": ["one"],
        },
    ]

    rows, summary = convert_rows(
        metadata_rows,
        retrieval_rows,
        kb,
        top_k=100,
        min_mapped_docs=1,
        image_path_mode="basename",
        split="valid_m2kr",
    )

    assert [row["question_id"] for row in rows] == ["q2", "q1"]
    assert rows[0]["img_path"] == "img2.jpg"
    assert rows[0]["retrieved_passage"] == [
        {
            "passage_id": "https://wiki/one",
            "passage_title": "One",
            "score": 1.0,
            "rank": 2,
            "knowledge_base_index": 1,
        }
    ]
    assert rows[0]["source_split"] == "valid_m2kr"
    assert summary["converted_examples"] == 2
    assert summary["unmatched_retrieval_records"] == 1


def test_conversion_uses_only_unambiguous_image_question_fallback():
    kb = [{"url": "https://wiki/zero", "title": "Zero"}]
    retrieval_rows = [
        _retrieval("retrieval-id", "img1", "  Same   Question ", [_hit(0, "https://wiki/zero")])
    ]
    metadata_rows = [
        {
            "example_id": "different-id",
            "image_id": "img1",
            "question": "same question",
            "gold_answer": "answer",
        }
    ]

    rows, summary = convert_rows(
        metadata_rows,
        retrieval_rows,
        kb,
        top_k=1,
        min_mapped_docs=1,
        image_path_mode="basename",
        split="valid_m2kr",
    )

    assert rows[0]["question_id"] == "different-id"
    assert rows[0]["retrieval_match_method"] == "image_question"
    assert summary["counters"]["matched_by_image_question"] == 1


def test_conversion_fails_on_missing_or_duplicate_ids():
    kb = [{"url": "https://wiki/zero", "title": "Zero"}]
    retrieval = _retrieval("q1", "img1", "question", [_hit(0, "https://wiki/zero")])

    with pytest.raises(ValueError, match="no retrieval match"):
        convert_rows(
            [{"question_id": "missing", "question": "other"}],
            [retrieval],
            kb,
            top_k=1,
            min_mapped_docs=1,
            image_path_mode="basename",
            split="valid_m2kr",
        )

    with pytest.raises(ValueError, match="duplicate IDs"):
        build_retrieval_indexes([retrieval, dict(retrieval)])


def test_prepare_infoseek_example_adds_metadata_and_titles():
    args = Namespace(
        img_basedir="/images",
        dry_run_no_images=True,
        retrieval_topk=1,
        retrieval_field="retrieved_passage",
        ensure_gt_passage_in_ensemble=False,
        max_words_per_evidence=3,
        add_pseudo_document=False,
        continue_generation_from_ans=False,
    )
    row = {
        "question_id": "q1",
        "data_id": "data-q1",
        "img_path": "image.jpg",
        "question": "What?",
        "question_type": "infoseek",
        "gold_answer": "answer",
        "answers": ["answer"],
        "entity_id": "entity-1",
        "wiki_title": "Wiki title",
        "wikipedia_url": "https://wiki/gold",
        "pos_item_ids": ["https://wiki/gold"],
        "retrieved_passage": [
            {
                "passage_id": "https://wiki/zero",
                "passage_title": "Readable title",
                "score": 1.0,
            }
        ],
        "source_split": "valid_m2kr",
    }

    example = prepare_infoseek_example(
        row,
        args,
        FakeTokenizer(),
        DEFAULT_PROMPT_TEMPLATE,
        {"https://wiki/zero": "one two three four"},
    )

    assert example.documents == ["Title: Readable title\tContent: one two three\n"]
    assert example.metadata["data_id"] == "data-q1"
    assert example.metadata["entity_id"] == "entity-1"
    assert example.metadata["source_split"] == "valid_m2kr"
    assert example.metadata["add_pseudo_document"] is False


def test_prepare_infoseek_example_appends_pseudo_document():
    args = Namespace(
        img_basedir="/images",
        dry_run_no_images=True,
        retrieval_topk=1,
        retrieval_field="retrieved_passage",
        ensure_gt_passage_in_ensemble=False,
        max_words_per_evidence=3,
        add_pseudo_document=True,
        continue_generation_from_ans=False,
    )
    row = {
        "question_id": "q1",
        "img_path": "image.jpg",
        "question": "What?",
        "question_type": "infoseek",
        "gold_answer": "answer",
        "answers": ["answer"],
        "pos_item_ids": ["https://wiki/gold"],
        "retrieved_passage": [
            {
                "passage_id": "https://wiki/zero",
                "passage_title": "Readable title",
                "score": 1.0,
            }
        ],
    }

    example = prepare_infoseek_example(
        row,
        args,
        FakeTokenizer(),
        DEFAULT_PROMPT_TEMPLATE,
        {"https://wiki/zero": "one two three four"},
    )

    assert example.documents == [
        "Title: Readable title\tContent: one two three\n",
        PSEUDO_DOCUMENT_TEXT,
    ]
    assert example.retrieved_passage_ids == [
        "https://wiki/zero",
        PSEUDO_DOCUMENT_PASSAGE_ID,
    ]
    assert PSEUDO_DOCUMENT_TEXT in example.rag_request["prompt"]
    assert example.metadata["add_pseudo_document"] is True
    assert example.metadata["num_ensemble_passages"] == 2

    rag_row = build_output_row(
        example,
        Namespace(mode="rag", save_full_prompts=False),
        None,
        "dry_run",
    )
    assert rag_row["prompt_text_or_parts"] == {
        "retrieved_passage_ids": [
            "https://wiki/zero",
            PSEUDO_DOCUMENT_PASSAGE_ID,
        ]
    }

    berag_row = build_output_row(
        example,
        Namespace(mode="berag", save_full_prompts=False),
        None,
        "dry_run",
    )
    assert "documents" not in berag_row["prompt_text_or_parts"]
    assert berag_row["prompt_text_or_parts"]["retrieved_passage_ids"] == [
        "https://wiki/zero",
        PSEUDO_DOCUMENT_PASSAGE_ID,
    ]


def test_continue_generation_from_answer_prefills_assistant_turn():
    args = Namespace(
        img_basedir="/images",
        dry_run_no_images=True,
        retrieval_topk=1,
        retrieval_field="retrieved_passage",
        ensure_gt_passage_in_ensemble=False,
        max_words_per_evidence=3,
        add_pseudo_document=False,
        continue_generation_from_ans=True,
        default_prior_token_offset=-5,
    )
    row = {
        "question_id": "q1",
        "img_path": "image.jpg",
        "question": "What?",
        "question_type": "infoseek",
        "gold_answer": "answer",
        "answers": ["answer"],
        "pos_item_ids": ["https://wiki/gold"],
        "retrieved_passage": [
            {
                "passage_id": "https://wiki/zero",
                "passage_title": "Readable title",
                "score": 1.0,
            }
        ],
    }

    example = prepare_infoseek_example(
        row,
        args,
        FakeTokenizer(),
        DEFAULT_PROMPT_TEMPLATE,
        {"https://wiki/zero": "one two three four"},
    )

    assert example.rag_request["prompt"].endswith("<assistant>[ANSWER]")
    assert example.berag_suffix.endswith("<assistant>[ANSWER]")
    assert example.metadata["assistant_prefill"] == "[ANSWER]"

    tokenizer = FakeTokenizer()
    assert resolve_prior_token_offset(args, tokenizer) == -9
    assert effective_prior_token_offset(args) == -9
    assert args.prior_prefill_added_tokens == 4
    assert args.prior_offset_probe_runs == 1
    assert tokenizer.tokenize_calls == 2

    assert resolve_prior_token_offset(args, tokenizer) == -9
    assert tokenizer.tokenize_calls == 2


def test_gt_injection_does_not_require_duplicated_passage_content():
    args = Namespace(
        retrieval_topk=1,
        retrieval_field="retrieved_passage",
        ensure_gt_passage_in_ensemble=True,
    )
    row = {
        "question_id": "q1",
        "pos_item_ids": ["https://wiki/gold"],
        "retrieved_passage": [{"passage_id": "https://wiki/other", "score": 0.5}],
    }

    assert select_retrieved_passages(row, args) == [
        {"passage_id": "https://wiki/gold", "score": 1.0}
    ]
