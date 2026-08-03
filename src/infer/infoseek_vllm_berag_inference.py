#!/usr/bin/env python3
"""Run M2KR InfoSeek RAG or BERAG inference with the vLLM-BERAG engine."""

from __future__ import annotations

import argparse
import json
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
        "--explode-page-sections",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat every nonempty section from each selected EchoSight page "
            "as a separate RAG/BERAG evidence document."
        ),
    )
    parser.add_argument(
        "--max-exploded-docs-per-page",
        "--max_exploded_docs_per_page",
        dest="max_exploded_docs_per_page",
        type=int,
        default=5,
        help=(
            "Maximum number of nonempty sections retained from each page when "
            "--explode-page-sections is enabled (default: 5)."
        ),
    )
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


def load_echosight_kb_entries(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load EchoSight entries indexed by their canonical page URL."""
    with Path(path).open(encoding="utf-8") as f:
        raw_kb = json.load(f)
    if not isinstance(raw_kb, dict):
        raise ValueError(f"EchoSight KB must be a JSON object: {path}")

    kb_by_url: dict[str, dict[str, Any]] = {}
    for key, value in raw_kb.items():
        if not isinstance(value, dict):
            raise ValueError(f"EchoSight KB entry {key!r} is not an object.")
        page_url = str(value.get("url") or key)
        if page_url in kb_by_url:
            raise ValueError(f"Duplicate EchoSight page URL: {page_url}")
        kb_by_url[page_url] = value
    return kb_by_url


def explode_retrieved_page_sections(
    retrieved_pages: list[dict[str, Any]],
    kb_by_url: dict[str, dict[str, Any]],
    max_exploded_docs_per_page: int = 5,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand selected pages into aligned section documents and parent IDs."""
    if max_exploded_docs_per_page <= 0:
        raise ValueError("max_exploded_docs_per_page must be greater than 0.")

    sections: list[dict[str, Any]] = []
    parent_page_ids: list[str] = []
    for page in retrieved_pages:
        page_id = str(page["passage_id"])
        entry = kb_by_url.get(page_id)
        if entry is None:
            raise KeyError(f"Retrieved page is absent from EchoSight KB: {page_id}")

        section_titles = entry.get("section_titles") or []
        section_texts = entry.get("section_texts") or []
        if not isinstance(section_titles, list) or not isinstance(section_texts, list):
            raise ValueError(
                f"EchoSight sections must be lists for page {page_id}."
            )
        if len(section_titles) != len(section_texts):
            raise ValueError(
                "EchoSight section title/text length mismatch for "
                f"{page_id}: {len(section_titles)} titles versus "
                f"{len(section_texts)} texts."
            )

        page_title = str(
            entry.get("title")
            or page.get("passage_title")
            or page.get("title")
            or page_id
        ).strip()
        page_section_count = 0
        for section_index, (section_title_value, section_text_value) in enumerate(
            zip(section_titles, section_texts, strict=True)
        ):
            section_text = str(section_text_value or "").strip()
            if not section_text:
                continue
            section_title = str(section_title_value or "").strip()
            document_title = (
                f"{page_title} — {section_title}" if section_title else page_title
            )
            section = dict(page)
            section.update(
                {
                    "passage_id": f"{page_id}::section::{section_index}",
                    "passage_title": document_title,
                    "passage_content": section_text,
                    "parent_passage_id": page_id,
                    "section_index": section_index,
                    "section_title": section_title,
                }
            )
            sections.append(section)
            parent_page_ids.append(page_id)
            page_section_count += 1
            if page_section_count >= max_exploded_docs_per_page:
                break

        if page_section_count == 0:
            fallback_section = dict(page)
            fallback_section.update(
                {
                    "passage_id": f"{page_id}::section::0",
                    "passage_title": page_title,
                    "passage_content": page_title,
                    "parent_passage_id": page_id,
                    "section_index": 0,
                    "section_title": "",
                    "empty_section_fallback": True,
                }
            )
            sections.append(fallback_section)
            parent_page_ids.append(page_id)
    return sections, parent_page_ids


def add_pseudo_document(
    example: engine.PreparedExample,
    tokenizer: Any,
    prompt_template: str,
    assistant_prefill: str | None = None,
) -> None:
    """Append the curation pseudo-document and refresh the RAG prompt."""
    example.documents.append(PSEUDO_DOCUMENT_TEXT)
    example.retrieved_passage_ids.append(PSEUDO_DOCUMENT_PASSAGE_ID)
    branch_parent_page_ids = list(
        example.metadata.get("berag_branch_parent_page_ids") or []
    )
    branch_parent_page_ids.append(PSEUDO_DOCUMENT_PASSAGE_ID)
    example.metadata.update(
        {
            "add_pseudo_document": True,
            "pseudo_document_passage_id": PSEUDO_DOCUMENT_PASSAGE_ID,
            "retrieved_passage_ids": list(example.retrieved_passage_ids),
            "berag_branch_passage_ids": list(example.retrieved_passage_ids),
            "berag_branch_parent_page_ids": branch_parent_page_ids,
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
    kb_by_url: dict[str, dict[str, Any]] | None = None,
) -> engine.PreparedExample:
    retrieved_pages = engine.select_retrieved_passages(row, args)
    explode_page_sections = bool(
        getattr(args, "explode_page_sections", False)
    )
    if explode_page_sections:
        if kb_by_url is None:
            raise ValueError(
                "kb_by_url is required when --explode-page-sections is enabled."
            )
        retrieved_branches, branch_parent_page_ids = (
            explode_retrieved_page_sections(
                retrieved_pages,
                kb_by_url,
                max_exploded_docs_per_page=getattr(
                    args, "max_exploded_docs_per_page", 5
                ),
            )
        )
    else:
        retrieved_branches = retrieved_pages
        branch_parent_page_ids = [
            str(item["passage_id"]) for item in retrieved_pages
        ]

    example = engine.prepare_example(
        row,
        args,
        tokenizer,
        prompt_template,
        pid_to_content_map,
        retrieved_passages_override=retrieved_branches,
    )
    retrieved_page_ids = [str(item["passage_id"]) for item in retrieved_pages]
    branch_passage_ids = list(example.retrieved_passage_ids)
    gt_passage_id = example.metadata.get("gt_passage_id")
    gt_branch_indices = [
        index
        for index, parent_id in enumerate(branch_parent_page_ids)
        if parent_id == gt_passage_id
    ]
    example.metadata["gt_passage_in_zidx"] = (
        gt_branch_indices[0] if gt_branch_indices else -1
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
            "explode_page_sections": explode_page_sections,
            "max_exploded_docs_per_page": (
                getattr(args, "max_exploded_docs_per_page", 5)
                if explode_page_sections
                else None
            ),
            "retrieved_page_ids": retrieved_page_ids,
            "retrieved_section_ids": (
                branch_passage_ids if explode_page_sections else []
            ),
            "berag_branch_passage_ids": branch_passage_ids,
            "berag_branch_parent_page_ids": list(branch_parent_page_ids),
            "gt_passage_branch_indices": gt_branch_indices,
            "num_retrieved_pages": len(retrieved_page_ids),
            "num_ensemble_passages": len(example.documents),
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
    if args.explode_page_sections and args.retrieval_topk == 0:
        raise ValueError(
            "--explode-page-sections requires --retrieval-topk greater than 0."
        )
    if args.max_exploded_docs_per_page <= 0:
        raise ValueError("--max-exploded-docs-per-page must be greater than 0.")
    max_exploded_branches = (
        args.retrieval_topk * args.max_exploded_docs_per_page
    )
    if (
        args.mode == "berag"
        and args.explode_page_sections
        and max_exploded_branches >= args.num_accumulator_rows
    ):
        logger.warning(
            "retrieval_topk (%d) * max_exploded_docs_per_page (%d) = %d, "
            "which is not smaller than num_accumulator_rows (%d). BERAG also "
            "needs a mixture row; reduce K/the per-page cap or increase "
            "--num-accumulator-rows.",
            args.retrieval_topk,
            args.max_exploded_docs_per_page,
            max_exploded_branches,
            args.num_accumulator_rows,
        )

    prompt_template = engine.read_prompt_template(
        args.prompt_template,
        args.retrieval_topk,
    )
    engine.split_prompt_template(prompt_template)
    tokenizer = engine.load_tokenizer(args)
    if args.mode == "berag":
        engine.resolve_prior_token_offset(args, tokenizer)

    kb_by_url = None
    if args.explode_page_sections:
        kb_by_url = load_echosight_kb_entries(args.knowledge_base)
        pid_to_content_map = {}
    else:
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
            kb_by_url,
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

    result.metrics["explode_page_sections"] = args.explode_page_sections
    if args.explode_page_sections:
        result.metrics["max_exploded_docs_per_page"] = (
            args.max_exploded_docs_per_page
        )
        section_counts = [
            len(example.metadata.get("retrieved_section_ids") or [])
            for example in examples
        ]
        result.metrics["num_exploded_sections"] = sum(section_counts)
        result.metrics["mean_exploded_sections_per_example"] = (
            sum(section_counts) / len(section_counts) if section_counts else 0.0
        )
        result.metrics["max_exploded_sections_per_example"] = max(
            section_counts, default=0
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
