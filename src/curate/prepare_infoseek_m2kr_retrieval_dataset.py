#!/usr/bin/env python3
"""Build the M2KR InfoSeek evaluation split with EchoSight retrieval.

The EVA-CLIP JSONL covers the full InfoSeek validation set. This utility
selects the requested Hugging Face split (``valid_m2kr`` by default), matches
it to retrieval records, preserves HF dataset order, and saves a dataset that
the vLLM-BERAG InfoSeek entry point can consume with ``load_from_disk``.

The saved rows contain passage URLs and scores. Passage text stays in the
EchoSight knowledge-base JSON and is loaded once by inference.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RETRIEVAL_JSONL = (
    "/workspace/projects/Infoseek-retrieval/outputs/"
    "infoseek_test_evaclip_top100.full.jsonl"
)
DEFAULT_KNOWLEDGE_BASE = (
    "/workspace/projects/Infoseek-retrieval/data/echosight/"
    "infoseek_100k_wiki/wiki_100_dict_v4.json"
)
ID_FIELDS = ("question_id", "example_id", "data_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach full-validation EVA-CLIP retrieval to an InfoSeek HF split."
    )
    parser.add_argument("--retrieval-jsonl", default=DEFAULT_RETRIEVAL_JSONL)
    parser.add_argument("--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE)
    parser.add_argument("--output-data-dir", required=True)
    parser.add_argument("--dataset-repo", default="Jingbiao/aravqa")
    parser.add_argument("--dataset-config", default="Infoseek_data")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Maximum unique retrieved documents to retain per example.",
    )
    parser.add_argument("--min-mapped-docs", type=int, default=1)
    parser.add_argument(
        "--image-path-mode",
        choices=("keep", "basename"),
        default="basename",
    )
    parser.add_argument(
        "--allow-missing-retrieval",
        action="store_true",
        help="Drop unmatched HF rows instead of failing.",
    )
    parser.add_argument("--examples", type=int, default=5)
    return parser.parse_args()


def canonical_id(row: dict[str, Any]) -> str | None:
    """Return the first supported, non-empty example identifier."""
    for field in ID_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def first_image_id(value: Any) -> str:
    return str(value or "").split("|", maxsplit=1)[0].strip()


def normalized_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def image_question_key(row: dict[str, Any]) -> tuple[str, str] | None:
    """Build a conservative fallback key when dataset ID fields differ."""
    image_id = first_image_id(
        row.get("image_id")
        or row.get("retrieval_image_id")
        or row.get("dataset_image_ids")
        or row.get("img_path")
        or row.get("image_path")
    )
    if Path(image_id).suffix:
        image_id = Path(image_id).stem
    question = normalized_question(row.get("question") or row.get("question_original"))
    return (image_id, question) if image_id and question else None


def read_retrieval_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(record)
    return rows


def load_kb_by_index(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        kb_by_url = json.load(f)
    if not isinstance(kb_by_url, dict):
        raise ValueError(f"Expected a URL-keyed JSON object in {path}.")
    # Retrieval used list(kb_by_url.values()); this ordering is part of its schema.
    return list(kb_by_url.values())


def unique_index(
    rows: Iterable[dict[str, Any]],
    key_fn: Any,
) -> tuple[dict[Any, dict[str, Any]], set[Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is not None:
            grouped[key].append(row)
    duplicates = {key for key, values in grouped.items() if len(values) > 1}
    unique = {key: values[0] for key, values in grouped.items() if len(values) == 1}
    return unique, duplicates


def build_retrieval_indexes(
    retrieval_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    by_id, duplicate_ids = unique_index(retrieval_rows, canonical_id)
    if duplicate_ids:
        examples = sorted(map(str, duplicate_ids))[:5]
        raise ValueError(
            f"Retrieval JSONL has {len(duplicate_ids)} duplicate IDs; examples={examples}."
        )

    # Ambiguous fallback keys are deliberately omitted by unique_index.
    by_image_question, _ = unique_index(retrieval_rows, image_question_key)
    return by_id, by_image_question


def match_retrieval(
    metadata: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    by_image_question: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    metadata_id = canonical_id(metadata)
    if metadata_id and metadata_id in by_id:
        return by_id[metadata_id], "id"
    fallback_key = image_question_key(metadata)
    if fallback_key and fallback_key in by_image_question:
        return by_image_question[fallback_key], "image_question"
    return None, None


def resolve_img_path(
    metadata: dict[str, Any],
    retrieval: dict[str, Any],
    mode: str,
) -> str:
    candidate = str(metadata.get("img_path") or metadata.get("image_path") or "")
    if mode == "keep" and candidate:
        return candidate
    if candidate and Path(candidate).suffix:
        return Path(candidate).name

    image_id = first_image_id(
        metadata.get("image_id")
        or metadata.get("dataset_image_ids")
        or retrieval.get("image_id")
    )
    return f"{image_id}.JPEG"


def convert_retrieved_passages(
    record: dict[str, Any],
    kb: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for hit in record.get("retrieved_pages") or []:
        kb_index = hit.get("knowledge_base_index")
        if kb_index is None:
            continue
        try:
            entry = kb[int(kb_index)]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid knowledge_base_index={kb_index!r} for {canonical_id(record)!r}."
            ) from exc

        hit_url = str(hit.get("url") or "")
        entry_url = str(entry.get("url") or "")
        if hit_url and entry_url and hit_url != entry_url:
            raise ValueError(
                "Retrieval/KB URL mismatch for "
                f"{canonical_id(record)!r} at index {kb_index}: "
                f"{hit_url!r} != {entry_url!r}."
            )
        url = entry_url or hit_url
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        passages.append(
            {
                "passage_id": url,
                "passage_title": str(entry.get("title") or hit.get("title") or url),
                "score": float(hit.get("score", 0.0)),
                "rank": int(hit.get("rank") or len(passages) + 1),
                "knowledge_base_index": int(kb_index),
            }
        )
        if len(passages) >= top_k:
            break
    return passages


def make_converted_row(
    metadata: dict[str, Any],
    retrieval: dict[str, Any],
    kb: list[dict[str, Any]],
    *,
    top_k: int,
    image_path_mode: str,
    split: str,
    match_method: str,
) -> dict[str, Any]:
    question_id = canonical_id(metadata) or canonical_id(retrieval)
    if question_id is None:
        raise ValueError("Matched row has no question_id, example_id, or data_id.")
    gold_url = str(
        retrieval.get("ground_truth_url")
        or metadata.get("wikipedia_url")
        or metadata.get("entity_id")
        or ""
    )
    return {
        "question_id": question_id,
        "example_id": str(metadata.get("example_id") or question_id),
        "data_id": str(metadata.get("data_id") or question_id),
        "img_path": resolve_img_path(metadata, retrieval, image_path_mode),
        "question": metadata.get("question") or retrieval.get("question"),
        "question_type": metadata.get("question_type") or retrieval.get("question_type"),
        "gold_answer": metadata.get("gold_answer") or retrieval.get("answer"),
        "answers": metadata.get("answers"),
        "entity_id": metadata.get("entity_id"),
        "wiki_title": metadata.get("wiki_title"),
        "wikipedia_url": gold_url,
        "pos_item_ids": [gold_url] if gold_url else [],
        "retrieved_passage": convert_retrieved_passages(retrieval, kb, top_k),
        "retrieval_image_id": retrieval.get("image_id"),
        "retrieval_ground_truth_url": retrieval.get("ground_truth_url"),
        "retrieval_match_method": match_method,
        "source_split": split,
    }


def convert_rows(
    metadata_rows: Iterable[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    kb: list[dict[str, Any]],
    *,
    top_k: int,
    min_mapped_docs: int,
    image_path_mode: str,
    split: str,
    allow_missing_retrieval: bool = False,
    examples: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 1 <= min_mapped_docs <= top_k:
        raise ValueError("min_mapped_docs must be between 1 and top_k.")

    by_id, by_image_question = build_retrieval_indexes(retrieval_rows)
    converted: list[dict[str, Any]] = []
    counters = Counter()
    missing_examples: list[str] = []
    few_docs_examples: list[dict[str, Any]] = []
    matched_retrieval_ids: set[str] = set()

    # Iterate over HF metadata, not retrieval, to select only the requested split
    # and to retain its deterministic evaluation ordering.
    for raw_metadata in metadata_rows:
        metadata = dict(raw_metadata)
        retrieval, match_method = match_retrieval(metadata, by_id, by_image_question)
        if retrieval is None or match_method is None:
            counters["missing_retrieval"] += 1
            if len(missing_examples) < examples:
                missing_examples.append(str(canonical_id(metadata)))
            continue

        row = make_converted_row(
            metadata,
            retrieval,
            kb,
            top_k=top_k,
            image_path_mode=image_path_mode,
            split=split,
            match_method=match_method,
        )
        num_docs = len(row["retrieved_passage"])
        if num_docs < min_mapped_docs:
            counters["too_few_retrieved_docs"] += 1
            if len(few_docs_examples) < examples:
                few_docs_examples.append(
                    {"question_id": row["question_id"], "retrieved_docs": num_docs}
                )
            continue

        retrieval_id = canonical_id(retrieval)
        if retrieval_id:
            matched_retrieval_ids.add(retrieval_id)
        counters[f"matched_by_{match_method}"] += 1
        gold_url = row["pos_item_ids"][0] if row["pos_item_ids"] else None
        retrieved_ids = {item["passage_id"] for item in row["retrieved_passage"]}
        counters[
            "gold_in_retrieved" if gold_url in retrieved_ids else "gold_not_in_retrieved"
        ] += 1
        converted.append(row)

    if counters["missing_retrieval"] and not allow_missing_retrieval:
        raise ValueError(
            f"{counters['missing_retrieval']} HF rows have no retrieval match; "
            f"examples={missing_examples}. Pass --allow-missing-retrieval to drop them."
        )
    if counters["too_few_retrieved_docs"]:
        raise ValueError(
            f"{counters['too_few_retrieved_docs']} matched rows have fewer than "
            f"{min_mapped_docs} KB-resolved documents; examples={few_docs_examples}."
        )

    return converted, {
        "retrieval_records": len(retrieval_rows),
        "converted_examples": len(converted),
        "unmatched_retrieval_records": len(by_id) - len(matched_retrieval_ids),
        "top_k": top_k,
        "min_mapped_docs": min_mapped_docs,
        "counters": dict(counters),
        "examples_missing_retrieval": missing_examples,
        "examples_too_few_retrieved_docs": few_docs_examples,
    }


def main() -> None:
    args = parse_args()

    from datasets import Dataset, load_dataset

    print(f"Loading {args.dataset_repo}/{args.dataset_config}:{args.split}")
    metadata_ds = load_dataset(
        args.dataset_repo,
        data_dir=args.dataset_config,
        split=args.split,
    )

    print(f"Reading retrieval from {args.retrieval_jsonl}")
    retrieval_rows = read_retrieval_jsonl(args.retrieval_jsonl)
    print(f"Reading EchoSight knowledge base from {args.knowledge_base}")
    kb = load_kb_by_index(args.knowledge_base)

    rows, summary = convert_rows(
        metadata_ds,
        retrieval_rows,
        kb,
        top_k=args.top_k,
        min_mapped_docs=args.min_mapped_docs,
        image_path_mode=args.image_path_mode,
        split=args.split,
        allow_missing_retrieval=args.allow_missing_retrieval,
        examples=args.examples,
    )

    output_dir = Path(args.output_data_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(output_dir))
    summary.update(output_data_dir=str(output_dir), source_split=args.split)

    print("\n=== Conversion summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
