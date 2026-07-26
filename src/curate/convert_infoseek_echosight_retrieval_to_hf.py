#!/usr/bin/env python3
"""Convert EchoSight-style InfoSeek retrieval JSONL to BERAG curation datasets.

The converted VQA dataset uses EchoSight Wikipedia URLs as document/passsage ids:

  gold pos_item_ids[0] = row["wikipedia_url"]
  retrieved_passage[i]["passage_id"] = EchoSight KB entry URL

Passage text is loaded later by ragk_answer_ppl.py via:

  --passage_set_name EchoSightKB:/path/to/wiki_100_dict_v4.json

This keeps EchoSight KB loading centralized in vqa_datasets.load_passages().
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_jsonl", required=True, help="Output JSONL from Infoseek-retrieval EVA-CLIP run.")
    parser.add_argument("--echosight_train_csv", default=None, help="Optional EchoSight train CSV fallback for metadata by data_id.")
    parser.add_argument("--knowledge_base", required=True, help="EchoSight KB JSON, e.g. wiki_100_dict_v4.json.")
    parser.add_argument("--output_data_dir", required=True, help="save_to_disk path for converted VQA dataset.")
    parser.add_argument("--dataset_repo", default="Jingbiao/aravqa")
    parser.add_argument("--data_dir", default="Infoseek_data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_mapped_docs", type=int, default=1, help="Drop examples with fewer than this many URL-resolved retrieved docs.")
    parser.add_argument("--image_path_mode", choices=["keep", "basename"], default="basename")
    parser.add_argument("--examples", type=int, default=5)
    return parser.parse_args()


def load_kb_by_index(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        kb_by_url = json.load(f)
    # EchoSight also does list(json_dict.values()). The retrieval JSONL's
    # knowledge_base_index was produced against this same ordering.
    return list(kb_by_url.values())



def first_image_id(value: str | None) -> str:
    return (value or "").split("|")[0]


def load_retrieval_jsonl(path: str) -> dict[str, dict[str, Any]]:
    by_qid: dict[str, dict[str, Any]] = {}
    with open(path) as f:
        for line in tqdm(f, desc="Reading retrieval JSONL"):
            record = json.loads(line)
            qid = record.get("question_id")
            if not qid:
                raise ValueError(f"Retrieval record missing question_id: {record}")
            if qid in by_qid:
                raise ValueError(f"Duplicate question_id in retrieval JSONL: {qid}")
            by_qid[qid] = record
    return by_qid


def load_csv_by_qid(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    by_qid = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading EchoSight CSV metadata"):
            qid = row.get("data_id") or row.get("question_id")
            if qid:
                by_qid[qid] = row
    return by_qid


def build_infoseek_metadata(split: str, dataset_repo: str, data_dir: str) -> dict[str, dict[str, Any]]:
    ds = load_dataset(dataset_repo, data_dir=data_dir, split=split)
    metadata: dict[str, dict[str, Any]] = {}
    for row in tqdm(ds, desc=f"Indexing {dataset_repo}/{data_dir}:{split}"):
        metadata[row["question_id"]] = row
    return metadata


def resolve_img_path(meta: dict[str, Any], retrieval: dict[str, Any], image_path_mode: str) -> str:
    candidate = meta.get("img_path") or meta.get("image_path") or retrieval.get("image_id") or ""
    if image_path_mode == "basename":
        image_id = meta.get("image_id") or retrieval.get("image_id") or first_image_id(str(candidate))
        image_id = first_image_id(str(image_id))
        return Path(str(candidate)).name if Path(str(candidate)).suffix else f"{image_id}.JPEG"
    return str(candidate)


def converted_retrieved_docs(record: dict[str, Any], kb: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any]]]]:
    retrieved_passage: list[dict[str, Any]] = []
    used_entries: list[tuple[int, dict[str, Any]]] = []
    seen_urls: set[str] = set()
    for hit in record.get("retrieved_pages") or []:
        kb_index = hit.get("knowledge_base_index")
        if kb_index is None:
            continue
        try:
            entry = kb[int(kb_index)]
        except (IndexError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid knowledge_base_index={kb_index!r} in record {record.get('question_id')}") from exc
        url = entry.get("url") or hit.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        retrieved_passage.append({"passage_id": url, "score": float(hit.get("score", 0.0))})
        used_entries.append((int(kb_index), entry))
        if len(retrieved_passage) >= top_k:
            break
    return retrieved_passage, used_entries


def main() -> None:
    args = parse_args()
    kb = load_kb_by_index(args.knowledge_base)
    retrieval_by_qid = load_retrieval_jsonl(args.retrieval_jsonl)
    infoseek_meta = build_infoseek_metadata(args.split, args.dataset_repo, args.data_dir)
    csv_meta = load_csv_by_qid(args.echosight_train_csv)

    rows: list[dict[str, Any]] = []
    counters = Counter()
    examples_missing_meta = []
    examples_few_docs = []

    for qid, retrieval in tqdm(retrieval_by_qid.items(), desc="Converting retrieval records"):
        meta = infoseek_meta.get(qid) or csv_meta.get(qid)
        if meta is None:
            counters["missing_metadata"] += 1
            if len(examples_missing_meta) < args.examples:
                examples_missing_meta.append(qid)
            continue

        retrieved_passage, used_entries = converted_retrieved_docs(retrieval, kb, args.top_k)
        if len(retrieved_passage) < args.min_mapped_docs:
            counters["dropped_too_few_retrieved_docs"] += 1
            if len(examples_few_docs) < args.examples:
                examples_few_docs.append({"question_id": qid, "retrieved_docs": len(retrieved_passage)})
            continue

        gold_url = retrieval.get("ground_truth_url") or meta.get("wikipedia_url")
        if not gold_url:
            # Infoseek_data has entity_id/wiki_title but not always URL. Retrieval JSONL
            # from our EchoSight run should carry ground_truth_url, so missing this is anomalous.
            counters["missing_gold_url"] += 1
            gold_url = str(meta.get("entity_id") or "")

        gold_in_topk = any(p["passage_id"] == gold_url for p in retrieved_passage)
        counters["gold_in_topk" if gold_in_topk else "gold_not_in_topk"] += 1

        rows.append(
            {
                "question_id": qid,
                "img_path": resolve_img_path(meta, retrieval, args.image_path_mode),
                "question": meta.get("question") or retrieval.get("question"),
                "gold_answer": meta.get("gold_answer") or retrieval.get("answer"),
                "answers": meta.get("answers"),
                "entity_id": meta.get("entity_id"),
                "wiki_title": meta.get("wiki_title"),
                "wikipedia_url": gold_url,
                "pos_item_ids": [gold_url],
                "retrieved_passage": retrieved_passage,
                "source_row_index": retrieval.get("source_row_index"),
                "retrieval_image_id": retrieval.get("image_id"),
                "retrieval_ground_truth_url": retrieval.get("ground_truth_url"),
            }
        )
        counters["converted"] += 1

    data_ds = Dataset.from_list(rows)

    output_data_dir = Path(args.output_data_dir)
    output_data_dir.parent.mkdir(parents=True, exist_ok=True)
    data_ds.save_to_disk(str(output_data_dir))

    print("\n=== Conversion summary ===")
    print(json.dumps({
        "retrieval_records": len(retrieval_by_qid),
        "converted_examples": len(data_ds),
        "top_k": args.top_k,
        "min_mapped_docs": args.min_mapped_docs,
        "counters": dict(counters),
        "examples_missing_metadata": examples_missing_meta,
        "examples_too_few_retrieved_docs": examples_few_docs,
        "output_data_dir": str(output_data_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
