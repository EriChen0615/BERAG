from argparse import Namespace

import torch

from infer.beft_prior import BeftPriorHead
from infer.evqa_vllm_berag_inference import (
    DEFAULT_PROMPT_TEMPLATE,
    make_berag_user_prompt_with_sentinel,
    make_evidence_document,
    make_rag_user_prompt,
    render_chat_prompt,
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
