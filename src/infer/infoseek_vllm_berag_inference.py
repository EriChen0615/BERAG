#!/usr/bin/env python3
"""Run M2KR InfoSeek RAG or BERAG inference with the vLLM-BERAG engine."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from infer import evqa_vllm_berag_inference as engine  # noqa: E402


logger = logging.getLogger(__name__)
DEFAULT_KNOWLEDGE_BASE = (
    "/workspace/projects/Infoseek-retrieval/data/echosight/"
    "infoseek_100k_wiki/wiki_100_dict_v4.json"
)
PSEUDO_DOCUMENT_PASSAGE_ID = "__infoseek_retrieval_failed__"
PSEUDO_DOCUMENT_TEXT = (
    "Title: Retrieval failed\tContent: retrieval failed, use your own knowledge"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM-BERAG over a prepared M2KR InfoSeek retrieval dataset."
    )
    parser.add_argument("--mode", choices=("rag", "berag"), required=True)
    parser.add_argument("--retrieval-ds-path", required=True)
    parser.add_argument("--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE)
    parser.add_argument("--img-basedir", required=True)
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metrics-output-path", default=None)
    parser.add_argument(
        "--save-full-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save rendered prompts and document text instead of passage IDs only.",
    )
    parser.add_argument(
        "--write-on-batch-complete",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--resume-from-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--take-n", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-words-per-evidence", type=int, default=1024)
    parser.add_argument("--ensure-gt-passage-in-ensemble", action="store_true")
    parser.add_argument(
        "--retrieval-field",
        choices=("retrieved_passage", "reranked_passage"),
        default="retrieved_passage",
    )
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument(
        "--continue_generation_from_ans",
        "--continue-generation-from-ans",
        dest="continue_generation_from_ans",
        action="store_true",
        help=(
            "Prefill the assistant turn with [ANSWER] and continue generation "
            "from that incomplete turn."
        ),
    )
    parser.add_argument(
        "--add-pseudo-document",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append the retrieval-failed pseudo-document used during curation "
            "after the selected top-K passages."
        ),
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--prior-module-cls", default=engine.DEFAULT_PRIOR_MODULE_CLS)
    parser.add_argument("--prior-hidden-size", type=int, default=None)
    parser.add_argument(
        "--prior-modeling",
        choices=("linear_head", "mlp_head"),
        default="mlp_head",
    )
    parser.add_argument("--prior-head-num-layers", type=int, default=2)
    parser.add_argument("--prior-head-proj-dim", type=int, default=1024)
    parser.add_argument(
        "--default-prior-token-offset",
        type=int,
        default=-5,
        help=(
            "Prompt-relative hidden-state index for the BEFT prior head. "
            "InfoSeek Qwen2/Qwen3 checkpoints trained with hidden-state "
            "offset 4 require -5 because training selects L - offset - 1."
        ),
    )
    parser.add_argument("--num-accumulator-rows", type=int, default=512)
    parser.add_argument("--pruning-top-p", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    # The shared loader only uses this to apply EVQA-specific path cleanup.
    args.dataset_name = "InfoseekNew"
    return args


def add_pseudo_document(
    example: engine.PreparedExample,
    tokenizer: Any,
    prompt_template: str,
    assistant_prefill: str | None = None,
) -> None:
    """Append the curation pseudo-document and refresh the RAG prompt."""
    example.documents.append(PSEUDO_DOCUMENT_TEXT)
    example.retrieved_passage_ids.append(PSEUDO_DOCUMENT_PASSAGE_ID)
    example.metadata.update(
        {
            "add_pseudo_document": True,
            "pseudo_document_passage_id": PSEUDO_DOCUMENT_PASSAGE_ID,
            "retrieved_passage_ids": list(example.retrieved_passage_ids),
            "num_ensemble_passages": len(example.documents),
        }
    )

    rag_user_prompt = engine.make_rag_user_prompt(
        prompt_template,
        example.documents,
        example.question,
    )
    rag_prompt = engine.render_chat_prompt(
        tokenizer,
        rag_user_prompt,
        include_image=example.image is not None,
        assistant_prefill=assistant_prefill,
    )
    example.rag_request = engine.make_prompt_dict(rag_prompt, example.image)


def prepare_infoseek_example(
    row: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: Any,
    prompt_template: str,
    pid_to_content_map: dict[str, str],
) -> engine.PreparedExample:
    example = engine.prepare_example(
        row,
        args,
        tokenizer,
        prompt_template,
        pid_to_content_map,
    )
    add_pseudo_document_enabled = bool(
        getattr(args, "add_pseudo_document", False)
    )
    continue_generation_from_ans = bool(
        getattr(args, "continue_generation_from_ans", False)
    )
    assistant_prefill = (
        engine.ANSWER_PREFILL if continue_generation_from_ans else None
    )
    example.metadata.update(
        {
            "add_pseudo_document": add_pseudo_document_enabled,
            "continue_generation_from_ans": continue_generation_from_ans,
            "assistant_prefill": assistant_prefill,
        }
    )
    if add_pseudo_document_enabled:
        add_pseudo_document(
            example,
            tokenizer,
            prompt_template,
            assistant_prefill=assistant_prefill,
        )

    if isinstance(example.berag_shared_prefix, dict) and example.image is not None:
        example.berag_shared_prefix["multi_modal_uuids"] = {
            "image": [f"infoseek:{example.question_id}:query-image"]
        }

    example.metadata.update(
        {
            "data_id": str(row.get("data_id") or example.question_id),
            "example_id": str(row.get("example_id") or example.question_id),
            "entity_id": row.get("entity_id"),
            "wiki_title": row.get("wiki_title"),
            "wikipedia_url": row.get("wikipedia_url"),
            "retrieval_image_id": row.get("retrieval_image_id"),
            "retrieval_match_method": row.get("retrieval_match_method"),
            "source_split": row.get("source_split"),
        }
    )
    return example


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    from vqa_datasets import load_passages

    engine.validate_args(args)
    if args.add_pseudo_document and args.retrieval_topk == 0:
        raise ValueError(
            "--add-pseudo-document requires --retrieval-topk greater than 0."
        )

    prompt_template = engine.read_prompt_template(
        args.prompt_template,
        args.retrieval_topk,
    )
    engine.split_prompt_template(prompt_template)
    tokenizer = engine.load_tokenizer(args)
    if args.mode == "berag":
        engine.resolve_prior_token_offset(args, tokenizer)

    passage_set_name = f"EchoSightKB:{Path(args.knowledge_base).resolve()}"
    _, pid_to_content_map = load_passages(passage_set_name, split="validation")

    existing_rows = (
        engine.read_jsonl(args.output_path) if args.resume_from_jsonl else []
    )
    completed_question_ids = {
        str(row["question_id"])
        for row in existing_rows
        if row.get("question_id") is not None
    }
    if args.write_on_batch_complete and not args.resume_from_jsonl:
        engine.write_jsonl(args.output_path, [])

    rows = engine.load_dataset_rows(args)
    if completed_question_ids:
        first_pending_idx = next(
            (
                idx
                for idx, row in enumerate(rows)
                if str(row.get("question_id")) not in completed_question_ids
            ),
            len(rows),
        )
        rows = [
            row
            for row in rows
            if str(row.get("question_id")) not in completed_question_ids
        ]
        logger.info(
            "Resuming from %s: loaded %d generated rows; "
            "first pending dataset index is %d.",
            args.output_path,
            len(existing_rows),
            first_pending_idx,
        )

    examples = [
        prepare_infoseek_example(
            row,
            args,
            tokenizer,
            prompt_template,
            pid_to_content_map,
        )
        for row in rows
    ]
    logger.info("Prepared %d M2KR InfoSeek examples for %s inference.", len(examples), args.mode)

    if examples:
        result = engine.run_inference(examples, args)
    else:
        result = engine.InferenceRunResult(
            rows=[],
            metrics=engine._aggregate_run_metrics(
                [],
                args=args,
                vllm_run_wall_time_s=0.0,
                num_request_batches=0,
            ),
        )

    result.metrics["add_pseudo_document"] = args.add_pseudo_document
    result.metrics["save_full_prompts"] = args.save_full_prompts
    result.metrics["continue_generation_from_ans"] = (
        args.continue_generation_from_ans
    )
    if args.mode == "berag":
        result.metrics["effective_prior_token_offset"] = (
            engine.effective_prior_token_offset(args)
        )
        result.metrics["configured_prior_token_offset"] = (
            args.default_prior_token_offset
        )
        result.metrics["prior_prefill_added_tokens"] = (
            args.prior_prefill_added_tokens
        )
        result.metrics["prior_offset_probe_runs"] = args.prior_offset_probe_runs
        result.metrics["prior_target_token_id"] = args.prior_target_token_id
        result.metrics["prior_target_token_text"] = args.prior_target_token_text
    if args.add_pseudo_document:
        result.metrics["pseudo_document_passage_id"] = (
            PSEUDO_DOCUMENT_PASSAGE_ID
        )

    output_rows = existing_rows + result.rows
    if not args.write_on_batch_complete:
        engine.write_jsonl(args.output_path, output_rows)
    metrics_path = (
        Path(args.metrics_output_path)
        if args.metrics_output_path
        else Path(args.output_path).with_name("run_metrics.json")
    )
    engine.write_json(metrics_path, result.metrics)
    logger.info(
        "Output JSONL at %s has %d total rows; %d rows generated in this run.",
        args.output_path,
        len(output_rows),
        len(result.rows),
    )
    logger.info("Wrote vLLM timing metrics to %s.", metrics_path)


if __name__ == "__main__":
    main()
