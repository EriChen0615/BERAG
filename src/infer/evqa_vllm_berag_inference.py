#!/usr/bin/env python3
"""Run EVQA RAG or BERAG inference with the vLLM-BERAG engine."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vqa_datasets import load_passages  # noqa: E402


logger = logging.getLogger(__name__)
EVIDENCE_SENTINEL = "<<<EVIDENCE>>>"
DEFAULT_PROMPT_TEMPLATE = (
    "Answer the question after [QUESTION] about the image."
    "A retriever has retrieved a relevant document for you and provided it after [EVIDENCE]."
    "Give your answer after [ANSWER] without explanations\n"
    "[EVIDENCE] <<<EVIDENCE>>>"
)
DEFAULT_PRIOR_MODULE_CLS = "infer.beft_prior.BeftPriorHead"


@dataclass
class PreparedExample:
    question_id: str
    question: str
    question_type: str | None
    gold_answer: Any
    answers: Any
    image_path: str
    image: Any
    documents: list[str]
    retrieved_passage_ids: list[str]
    rag_request: dict[str, Any]
    berag_shared_prefix: str | dict[str, Any]
    berag_suffix: str
    metadata: dict[str, Any]


def read_prompt_template(path: str | None) -> str:
    if path is None:
        return DEFAULT_PROMPT_TEMPLATE
    return Path(path).read_text(encoding="utf-8")


def split_prompt_template(prompt_template: str) -> tuple[str, str]:
    parts = prompt_template.split(EVIDENCE_SENTINEL)
    if len(parts) != 2:
        raise ValueError(f"Prompt template must contain {EVIDENCE_SENTINEL!r} exactly once.")
    return parts[0], parts[1]


def render_chat_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    include_image: bool,
    system_prompt: str | None = None,
) -> str:
    user_content: str | list[dict[str, Any]]
    if include_image:
        user_content = [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_evidence_document(
    passage_dict: dict[str, Any],
    pid_to_content_map: dict[str, str],
    max_words_per_evidence: int,
) -> str:
    passage_id = str(passage_dict["passage_id"])
    text = pid_to_content_map.get(passage_id) or passage_dict.get("passage_content") or ""
    words = str(text).split()
    if max_words_per_evidence > 0:
        text = " ".join(words[:max_words_per_evidence])
    return f"Title: {passage_id}\tContent: {text}\n"


def make_rag_user_prompt(prompt_template: str, documents: list[str], question: str) -> str:
    evidence = " ".join(documents)
    return prompt_template.replace(EVIDENCE_SENTINEL, evidence) + f"\n[QUESTION] {question}\n[ANSWER]"


def make_berag_user_prompt_with_sentinel(prompt_template: str, question: str) -> str:
    return prompt_template + f"\n[QUESTION] {question}\n[ANSWER]"


def split_rendered_berag_prompt(
    tokenizer: Any,
    prompt_template: str,
    question: str,
    *,
    include_image: bool,
) -> tuple[str, str]:
    rendered = render_chat_prompt(
        tokenizer,
        make_berag_user_prompt_with_sentinel(prompt_template, question),
        include_image=include_image,
    )
    parts = rendered.split(EVIDENCE_SENTINEL)
    if len(parts) != 2:
        raise ValueError("Rendered BERAG prompt must contain the evidence sentinel exactly once.")
    return parts[0], parts[1]


def make_prompt_dict(prompt: str, image: Any | None, image_uuid: str | None = None) -> dict[str, Any]:
    request: dict[str, Any] = {"prompt": prompt}
    if image is not None:
        request["multi_modal_data"] = {"image": image}
        if image_uuid is not None:
            request["multi_modal_uuids"] = {"image": [image_uuid]}
    return request


def generated_answer_from_response(response: str) -> str:
    if "[ANSWER]" in response:
        return response.split("[ANSWER]", maxsplit=1)[1].strip()
    return response.strip()


def load_image(image_path: str) -> Any:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.convert("RGB")


def select_retrieved_passages(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    retrieved = [dict(item) for item in row[args.retrieval_field][: args.retrieval_topk]]
    if not retrieved:
        raise ValueError(f"No retrieved passages for question_id={row.get('question_id')}.")

    if args.ensure_gt_passage_in_ensemble:
        gt_passage_id = row["pos_item_ids"][0]
        retrieved_passage_ids = [item["passage_id"] for item in retrieved]
        if gt_passage_id not in retrieved_passage_ids:
            score_field = "score" if args.retrieval_field == "retrieved_passage" else "rerank_score"
            retrieved[0] = {
                "passage_id": gt_passage_id,
                "passage_content": row["pos_item_contents"][0],
                score_field: 1.0,
            }
    return retrieved


def prepare_example(
    row: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: Any,
    prompt_template: str,
    pid_to_content_map: dict[str, str],
) -> PreparedExample:
    question_id = str(row["question_id"])
    image_path = os.path.join(args.img_basedir, row["img_path"])
    image = None if args.dry_run_no_images else load_image(image_path)
    retrieved_passages = select_retrieved_passages(row, args)
    documents = [
        make_evidence_document(item, pid_to_content_map, args.max_words_per_evidence)
        for item in retrieved_passages
    ]
    retrieved_passage_ids = [str(item["passage_id"]) for item in retrieved_passages]

    rag_user_prompt = make_rag_user_prompt(prompt_template, documents, row["question"])
    rag_prompt = render_chat_prompt(tokenizer, rag_user_prompt, include_image=image is not None)
    shared_prefix, suffix = split_rendered_berag_prompt(
        tokenizer, prompt_template, row["question"], include_image=image is not None
    )
    image_uuid = f"evqa:{question_id}:query-image"

    metadata = {
        "question_id": question_id,
        "question": row["question"],
        "question_type": row.get("question_type"),
        "gold_answer": row.get("gold_answer"),
        "answers": row.get("answers"),
        "retrieval_field": args.retrieval_field,
        "retrieval_topk": args.retrieval_topk,
        "retrieved_passage_ids": retrieved_passage_ids,
        "image_path": image_path,
    }
    return PreparedExample(
        question_id=question_id,
        question=row["question"],
        question_type=row.get("question_type"),
        gold_answer=row.get("gold_answer"),
        answers=row.get("answers"),
        image_path=image_path,
        image=image,
        documents=documents,
        retrieved_passage_ids=retrieved_passage_ids,
        rag_request=make_prompt_dict(rag_prompt, image),
        berag_shared_prefix=make_prompt_dict(shared_prefix, image, image_uuid),
        berag_suffix=suffix,
        metadata=metadata,
    )


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer_path or args.model
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)


def resolve_hidden_size(args: argparse.Namespace) -> int:
    if args.prior_hidden_size is not None:
        return int(args.prior_hidden_size)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.tokenizer_path or args.model, trust_remote_code=args.trust_remote_code)
    for candidate in (config, getattr(config, "text_config", None), getattr(config, "llm_config", None)):
        hidden_size = getattr(candidate, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    raise ValueError("Could not infer hidden_size. Pass --prior-hidden-size.")


def ensure_vllm_berag_import_path(args: argparse.Namespace) -> None:
    vllm_path = Path(args.vllm_berag_path).resolve()
    if str(vllm_path) not in sys.path:
        sys.path.insert(0, str(vllm_path))


def make_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )


def make_berag_params(args: argparse.Namespace) -> Any:
    from vllm.berag import BeragParams

    return BeragParams(pruning_top_p=args.pruning_top_p)


def make_llm(args: argparse.Namespace) -> Any:
    ensure_vllm_berag_import_path(args)
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer_path or args.model,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "limit_mm_per_prompt": {"image": 1},
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    if args.mode == "berag":
        llm_kwargs.update(
            {
                "berag_num_accumulator_rows": args.num_accumulator_rows,
                "berag_prior_mode": args.prior_mode,
                "berag_default_prior_token_offset": args.default_prior_token_offset,
            }
        )
        if args.prior_mode == "module":
            llm_kwargs.update(
                {
                    "berag_prior_module_cls": args.prior_module_cls,
                    "berag_prior_module_weights_path": args.prior_head_path,
                    "berag_prior_module_kwargs": {
                        "hidden_size": resolve_hidden_size(args),
                        "prior_modeling": args.prior_modeling,
                        "num_layers": args.prior_head_num_layers,
                        "proj_dim": args.prior_head_proj_dim,
                    },
                }
            )
    return LLM(**llm_kwargs)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_output_row(
    example: PreparedExample,
    args: argparse.Namespace,
    response: str | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    row = dict(example.metadata)
    row.update(
        {
            "mode": args.mode,
            "status": status,
            "response": response,
            "generated_answer": generated_answer_from_response(response or "") if response is not None else None,
            "error": error,
            "prompt_text_or_parts": (
                {"rag_prompt": example.rag_request["prompt"]}
                if args.mode == "rag"
                else {
                    "shared_prefix": example.berag_shared_prefix["prompt"]
                    if isinstance(example.berag_shared_prefix, dict)
                    else example.berag_shared_prefix,
                    "documents": example.documents,
                    "suffix": example.berag_suffix,
                }
            ),
        }
    )
    return row


def iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_inference(examples: list[PreparedExample], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dry_run:
        return [build_output_row(example, args, None, "dry_run") for example in examples]

    llm = make_llm(args)
    sampling_params = make_sampling_params(args)
    rows: list[dict[str, Any]] = []
    for batch in iter_batches(examples, args.batch_size):
        try:
            if args.mode == "rag":
                outputs = llm.generate(
                    [example.rag_request for example in batch],
                    sampling_params,
                    use_tqdm=not args.disable_tqdm,
                )
            else:
                outputs = llm.generate_berag(
                    [example.berag_shared_prefix for example in batch],
                    [example.documents for example in batch],
                    [example.berag_suffix for example in batch],
                    sampling_params,
                    berag_params=make_berag_params(args),
                    request_id=[example.question_id for example in batch],
                    use_tqdm=not args.disable_tqdm,
                    debug=args.debug,
                )
            for example, output in zip(batch, outputs, strict=True):
                response = output.outputs[0].text if output.outputs else ""
                rows.append(build_output_row(example, args, response, "ok"))
        except Exception as exc:
            if args.stop_on_error:
                raise
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            logger.exception("Inference batch failed; writing error rows.")
            for example in batch:
                rows.append(build_output_row(example, args, None, "error", error=error))
    return rows


def load_dataset_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    dataset = load_from_disk(args.retrieval_ds_path)
    if args.take_n > 0:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.take_n, len(dataset))))
    return [dict(row) for row in dataset]


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "berag" and args.prior_mode == "module" and not args.prior_head_path:
        raise ValueError("BERAG module-prior mode requires --prior-head-path.")
    if args.prior_modeling == "mlp_head":
        if args.prior_head_num_layers is None or args.prior_head_proj_dim is None:
            raise ValueError("--prior-modeling mlp_head requires --prior-head-num-layers and --prior-head-proj-dim.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than 0.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rag", "berag"), required=True)
    parser.add_argument("--retrieval-ds-path", required=True)
    parser.add_argument("--dataset-name", default="EVQA")
    parser.add_argument("--img-basedir", default="")
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--take-n", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-words-per-evidence", type=int, default=1024)
    parser.add_argument("--ensure-gt-passage-in-ensemble", action="store_true")
    parser.add_argument("--retrieval-field", choices=("retrieved_passage", "reranked_passage"), required=True)
    parser.add_argument("--retrieval-topk", type=int, default=5)

    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--vllm-berag-path", default=str(REPO_ROOT / "src" / "infer" / "vllm-berag"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-no-images", action="store_true")

    parser.add_argument("--prior-mode", choices=("uniform", "module"), default="module")
    parser.add_argument("--prior-head-path", default=None)
    parser.add_argument("--prior-module-cls", default=DEFAULT_PRIOR_MODULE_CLS)
    parser.add_argument("--prior-hidden-size", type=int, default=None)
    parser.add_argument("--prior-modeling", choices=("linear_head", "mlp_head"), default="mlp_head")
    parser.add_argument("--prior-head-num-layers", type=int, default=2)
    parser.add_argument("--prior-head-proj-dim", type=int, default=1024)
    parser.add_argument("--default-prior-token-offset", type=int, default=-4)
    parser.add_argument("--num-accumulator-rows", type=int, default=512)
    parser.add_argument("--pruning-top-p", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    validate_args(args)
    prompt_template = read_prompt_template(args.prompt_template)
    split_prompt_template(prompt_template)
    tokenizer = load_tokenizer(args)
    _, pid_to_content_map = load_passages(args.dataset_name, split="test")
    rows = load_dataset_rows(args)
    examples = [prepare_example(row, args, tokenizer, prompt_template, pid_to_content_map) for row in rows]
    logger.info("Prepared %d EVQA examples for %s inference.", len(examples), args.mode)
    output_rows = run_inference(examples, args)
    write_jsonl(args.output_path, output_rows)
    logger.info("Wrote %d rows to %s.", len(output_rows), args.output_path)


if __name__ == "__main__":
    main()
