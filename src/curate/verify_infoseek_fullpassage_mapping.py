#!/usr/bin/env python3
"""Verify InfoSeek_data entity IDs and title mapping against InfoseekFull_passages.

This is a diagnostic script only. It does not write datasets or mutate inputs.

Typical usage with local snapshot/downloaded folders:

  python src/curate/verify_infoseek_fullpassage_mapping.py \
    --infoseek_data_path /path/to/Jingbiao_aravqa/Infoseek_data \
    --full_passages_path /path/to/Jingbiao_aravqa/InfoseekFull_passages \
    --retrieval_jsonl /workspace/projects/Infoseek-retrieval/outputs/infoseek_train_evaclip_top100.sample64000.seed42.jsonl

Or load the needed HF sub-datasets directly, matching the expected repo layout:

  python src/curate/verify_infoseek_fullpassage_mapping.py \
    --dataset_repo Jingbiao/aravqa \
    --retrieval_jsonl /workspace/projects/Infoseek-retrieval/outputs/infoseek_train_evaclip_top100.sample64000.seed42.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infoseek_data_path", type=str, default=None, help="Local load_from_disk path for Infoseek_data.")
    parser.add_argument("--full_passages_path", type=str, default=None, help="Local load_from_disk path for InfoseekFull_passages.")
    parser.add_argument("--dataset_repo", type=str, default="Jingbiao/aravqa", help="HF repo used if local paths are not provided.")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--data_subdir", type=str, default="Infoseek_data")
    parser.add_argument("--passage_subdir", type=str, default="InfoseekFull_passages")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--passage_split", type=str, default="train")
    parser.add_argument("--retrieval_jsonl", type=str, default=None, help="Optional EVA-CLIP retrieval JSONL to test title/url -> passage_id coverage.")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional cap for data split diagnostics; 0 means all rows.")
    parser.add_argument("--max_retrieval_rows", type=int, default=0, help="Optional cap for retrieval JSONL diagnostics; 0 means all rows.")
    parser.add_argument("--examples", type=int, default=10, help="Number of mismatch/unmapped examples to print.")
    return parser.parse_args()


def normalize_title(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        text = parsed.path.rsplit("/", 1)[-1]
    text = unquote(text)
    text = text.replace("_", " ")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def title_from_url(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return unquote(parsed.path.rsplit("/", 1)[-1]).replace("_", " ")


def get_split(ds_or_dd: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(ds_or_dd, DatasetDict):
        if split not in ds_or_dd:
            raise KeyError(f"Split {split!r} not found. Available: {list(ds_or_dd.keys())}")
        return ds_or_dd[split]
    return ds_or_dd


def load_infoseek_sources(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.infoseek_data_path and args.full_passages_path:
        print(f"Loading Infoseek_data from local disk path: {args.infoseek_data_path}")
        data_dd = load_from_disk(args.infoseek_data_path)
        print(f"Loading InfoseekFull_passages from local disk path: {args.full_passages_path}")
        passage_dd = load_from_disk(args.full_passages_path)
    else:
        print(f"Loading Infoseek_data from {args.dataset_repo}, data_dir={args.data_subdir}")
        data_dd = load_dataset(
            args.dataset_repo,
            data_dir=args.data_subdir,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        print(f"Loading InfoseekFull_passages from {args.dataset_repo}, data_dir={args.passage_subdir}")
        passage_dd = load_dataset(
            args.dataset_repo,
            data_dir=args.passage_subdir,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )

    return get_split(data_dd, args.data_split), get_split(passage_dd, args.passage_split)


def iter_limited(ds: Dataset, max_rows: int):
    limit = len(ds) if max_rows <= 0 else min(max_rows, len(ds))
    for i in range(limit):
        yield i, ds[i]


def build_title_map(passage_ds: Dataset) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in tqdm(passage_ds, desc="Indexing full-passage titles"):
        key = normalize_title(row.get("title"))
        if key:
            grouped[key].append((str(row["passage_id"]), row.get("title", "")))

    unique = {key: vals[0][0] for key, vals in grouped.items() if len(vals) == 1}
    ambiguous = {key: vals for key, vals in grouped.items() if len(vals) > 1}
    return unique, ambiguous


def check_data_mapping(data_ds: Dataset, passage_ds: Dataset, max_rows: int, examples: int) -> dict[str, Any]:
    passage_ids = set(map(str, passage_ds["passage_id"]))
    title_to_pid, ambiguous_titles = build_title_map(passage_ds)

    entity_ids = set(map(str, data_ds["entity_id"])) if max_rows <= 0 else set()
    if max_rows > 0:
        for _, row in iter_limited(data_ds, max_rows):
            entity_ids.add(str(row["entity_id"]))

    direct_missing = []
    title_missing = []
    title_ambiguous = []
    title_disagree = []
    pos_missing = []
    counters = Counter()
    checked = 0

    for _, row in tqdm(iter_limited(data_ds, max_rows), desc="Checking InfoSeek_data rows"):
        checked += 1
        entity_id = str(row["entity_id"])
        wiki_title = row.get("wiki_title")
        title_key = normalize_title(wiki_title)

        if entity_id in passage_ids:
            counters["rows_entity_id_is_passage_id"] += 1
        else:
            counters["rows_entity_id_missing_from_passages"] += 1
            if len(direct_missing) < examples:
                direct_missing.append({"question_id": row.get("question_id"), "entity_id": entity_id, "wiki_title": wiki_title})

        if title_key in ambiguous_titles:
            counters["rows_wiki_title_ambiguous"] += 1
            if len(title_ambiguous) < examples:
                title_ambiguous.append({"question_id": row.get("question_id"), "entity_id": entity_id, "wiki_title": wiki_title, "candidates": ambiguous_titles[title_key][:5]})
        else:
            title_pid = title_to_pid.get(title_key)
            if title_pid is None:
                counters["rows_wiki_title_unmapped"] += 1
                if len(title_missing) < examples:
                    title_missing.append({"question_id": row.get("question_id"), "entity_id": entity_id, "wiki_title": wiki_title, "normalized": title_key})
            elif title_pid == entity_id:
                counters["rows_wiki_title_maps_to_entity_id"] += 1
            else:
                counters["rows_wiki_title_maps_to_different_pid"] += 1
                if len(title_disagree) < examples:
                    title_disagree.append({"question_id": row.get("question_id"), "entity_id": entity_id, "wiki_title": wiki_title, "title_pid": title_pid})

        for pos_id in row.get("pos_item_ids") or []:
            if str(pos_id) in passage_ids:
                counters["pos_item_ids_in_passages"] += 1
            else:
                counters["pos_item_ids_missing_from_passages"] += 1
                if len(pos_missing) < examples:
                    pos_missing.append({"question_id": row.get("question_id"), "pos_id": str(pos_id), "entity_id": entity_id})

    direct_overlap = len(entity_ids & passage_ids)
    stats = {
        "checked_rows": checked,
        "unique_entity_ids_checked": len(entity_ids),
        "passage_ids": len(passage_ids),
        "entity_id_passage_id_overlap": direct_overlap,
        "all_checked_entity_ids_are_passage_ids": direct_overlap == len(entity_ids),
        "unique_title_keys": len(title_to_pid),
        "ambiguous_title_keys": len(ambiguous_titles),
        "row_counters": dict(counters),
        "examples": {
            "direct_entity_missing": direct_missing,
            "title_missing": title_missing,
            "title_ambiguous": title_ambiguous,
            "title_disagree": title_disagree,
            "pos_item_missing": pos_missing,
        },
        "title_to_pid": title_to_pid,
        "ambiguous_titles": ambiguous_titles,
    }
    return stats


def check_retrieval_jsonl(path: str, title_to_pid: dict[str, str], ambiguous_titles: dict[str, list[tuple[str, str]]], max_rows: int, examples: int) -> dict[str, Any]:
    counters = Counter()
    unmapped_examples = []
    ambiguous_examples = []
    rows_with_at_least_2 = 0
    rows_with_gold_in_top2 = 0
    rows_with_gold_url_in_top2 = 0

    with open(path) as f:
        for row_idx, line in enumerate(tqdm(f, desc="Checking retrieval JSONL")):
            if max_rows and row_idx >= max_rows:
                break
            record = json.loads(line)
            counters["rows"] += 1
            gold_url_key = normalize_title(record.get("ground_truth_url"))
            mapped_top = []
            seen = set()
            for hit in record.get("retrieved_pages") or []:
                title = hit.get("title") or title_from_url(hit.get("url"))
                key = normalize_title(title)
                if not key:
                    counters["hits_empty_title"] += 1
                    continue
                if key in ambiguous_titles:
                    counters["hits_ambiguous_title"] += 1
                    if len(ambiguous_examples) < examples:
                        ambiguous_examples.append({"question_id": record.get("question_id"), "hit": hit, "normalized": key, "candidates": ambiguous_titles[key][:5]})
                    continue
                pid = title_to_pid.get(key)
                if pid is None:
                    counters["hits_unmapped"] += 1
                    if len(unmapped_examples) < examples:
                        unmapped_examples.append({"question_id": record.get("question_id"), "hit": hit, "normalized": key})
                    continue
                counters["hits_mapped"] += 1
                if pid in seen:
                    counters["hits_duplicate_pid"] += 1
                    continue
                seen.add(pid)
                mapped_top.append({"passage_id": pid, "score": hit.get("score"), "title": title, "url": hit.get("url")})
                if len(mapped_top) >= 2:
                    break

            if len(mapped_top) >= 2:
                rows_with_at_least_2 += 1
            if gold_url_key and any(normalize_title(hit.get("url")) == gold_url_key for hit in mapped_top):
                rows_with_gold_url_in_top2 += 1
            # If retrieval record already has a true passage/entity id in the future, this will measure it.
            gold_pid = record.get("entity_id") or record.get("gold_passage_id")
            if gold_pid and any(hit["passage_id"] == str(gold_pid) for hit in mapped_top):
                rows_with_gold_in_top2 += 1

    return {
        "counters": dict(counters),
        "rows_with_at_least_2_mapped_docs": rows_with_at_least_2,
        "rows_with_gold_url_in_top2": rows_with_gold_url_in_top2,
        "rows_with_gold_pid_in_top2_if_available": rows_with_gold_in_top2,
        "examples": {
            "unmapped_hits": unmapped_examples,
            "ambiguous_hits": ambiguous_examples,
        },
    }


def print_jsonable(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def pct(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def print_interpretation(data_stats: dict[str, Any], retrieval_stats: dict[str, Any] | None) -> None:
    counters = Counter(data_stats.get("row_counters", {}))
    checked = data_stats.get("checked_rows", 0)
    unique_entities = data_stats.get("unique_entity_ids_checked", 0)
    overlap = data_stats.get("entity_id_passage_id_overlap", 0)

    print("\n=== Interpretation summary ===")
    print(
        "1. Gold document ID check: "
        f"{overlap}/{unique_entities} unique entity_id values are present in InfoseekFull_passages.passage_id "
        f"({pct(overlap, unique_entities)})."
    )
    if data_stats.get("all_checked_entity_ids_are_passage_ids"):
        print("   ✅ This means Infoseek_data.entity_id can be used directly as the gold full-document passage_id.")
    else:
        print("   ⚠️ Some entity_id values are missing from InfoseekFull_passages; inspect examples.direct_entity_missing.")

    title_ok = counters.get("rows_wiki_title_maps_to_entity_id", 0)
    title_bad = counters.get("rows_wiki_title_maps_to_different_pid", 0)
    title_missing = counters.get("rows_wiki_title_unmapped", 0)
    title_ambiguous = counters.get("rows_wiki_title_ambiguous", 0)
    print(
        "2. Wiki title agreement check over rows: "
        f"{title_ok}/{checked} row titles map back to the row entity_id ({pct(title_ok, checked)}); "
        f"different={title_bad}, unmapped={title_missing}, ambiguous={title_ambiguous}."
    )
    if checked and title_ok / checked > 0.99:
        print("   ✅ Normalized title matching is very likely safe for converting EVA-CLIP retrieved page titles to passage_id.")
    else:
        print("   ⚠️ Title matching has noticeable misses/disagreements; inspect title_missing/title_disagree examples before conversion.")

    pos_ok = counters.get("pos_item_ids_in_passages", 0)
    pos_missing = counters.get("pos_item_ids_missing_from_passages", 0)
    print(
        "3. pos_item_ids check: "
        f"in_passages={pos_ok}, missing={pos_missing}."
    )

    if retrieval_stats is None:
        print("4. Retrieval JSONL check: skipped because --retrieval_jsonl was not provided.")
        return

    rc = Counter(retrieval_stats.get("counters", {}))
    rows = rc.get("rows", 0)
    hits_mapped = rc.get("hits_mapped", 0)
    hits_unmapped = rc.get("hits_unmapped", 0)
    hits_ambiguous = rc.get("hits_ambiguous_title", 0)
    total_hit_attempts = hits_mapped + hits_unmapped + hits_ambiguous + rc.get("hits_empty_title", 0)
    rows_with_2 = retrieval_stats.get("rows_with_at_least_2_mapped_docs", 0)
    gold_url_top2 = retrieval_stats.get("rows_with_gold_url_in_top2", 0)
    print(
        "4. Retrieval title mapping check: "
        f"mapped_hits={hits_mapped}/{total_hit_attempts} ({pct(hits_mapped, total_hit_attempts)}), "
        f"unmapped_hits={hits_unmapped}, ambiguous_hits={hits_ambiguous}."
    )
    print(
        "   Rows with at least 2 mapped retrieved docs: "
        f"{rows_with_2}/{rows} ({pct(rows_with_2, rows)})."
    )
    print(
        "   Gold URL appears in mapped top-2 retrieved docs: "
        f"{gold_url_top2}/{rows} ({pct(gold_url_top2, rows)})."
    )
    print("   Note: gold URL top-2 is the realistic retrieval recall proxy; it is expected to be well below 100%.")

def main() -> None:
    args = parse_args()
    data_ds, passage_ds = load_infoseek_sources(args)
    print(f"InfoSeek data split={args.data_split}: {len(data_ds)} rows, columns={data_ds.column_names}")
    print(f"Full passages split={args.passage_split}: {len(passage_ds)} rows, columns={passage_ds.column_names}")

    stats = check_data_mapping(data_ds, passage_ds, args.max_rows, args.examples)
    title_to_pid = stats.pop("title_to_pid")
    ambiguous_titles = stats.pop("ambiguous_titles")
    print("\n=== Infoseek_data ↔ InfoseekFull_passages diagnostics ===")
    print_jsonable(stats)

    retrieval_stats = None
    if args.retrieval_jsonl:
        retrieval_stats = check_retrieval_jsonl(args.retrieval_jsonl, title_to_pid, ambiguous_titles, args.max_retrieval_rows, args.examples)
        print("\n=== EVA-CLIP retrieval title/url ↔ InfoseekFull_passages diagnostics ===")
        print_jsonable(retrieval_stats)

    print_interpretation(stats, retrieval_stats)


if __name__ == "__main__":
    main()
