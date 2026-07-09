#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter, defaultdict


NONTRAIN_FILES = [
    "dev_15.jsonl",
    "dev_20.jsonl",
    "evaluation_15.jsonl",
    "evaluation_20.jsonl",
]
TRAIN_FILE = "train.jsonl"


def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_nontrain(path):
    count = 0
    keys = set()
    missing_field_counts = Counter()
    text_counts = []
    img_counts = []
    gold_counts = []
    evidence_modality = Counter()
    question_types = Counter()

    required_fields = [
        "q_id",
        "doc_name",
        "domain",
        "question",
        "evidence_modality_type",
        "question_type",
        "text_quotes",
        "img_quotes",
        "gold_quotes",
        "answer_short",
        "answer_interleaved",
    ]

    for obj in stream_jsonl(path):
        count += 1
        keys.update(obj.keys())
        for field in required_fields:
            if field not in obj:
                missing_field_counts[field] += 1

        text_quotes = obj.get("text_quotes", []) or []
        img_quotes = obj.get("img_quotes", []) or []
        gold_quotes = obj.get("gold_quotes", []) or []
        text_counts.append(len(text_quotes))
        img_counts.append(len(img_quotes))
        gold_counts.append(len(gold_quotes))

        for item in obj.get("evidence_modality_type", []) or []:
            evidence_modality[item] += 1
        question_types[obj.get("question_type", "UNKNOWN")] += 1

    def _stats(values):
        if not values:
            return {"min": 0, "avg": 0.0, "max": 0}
        return {
            "min": min(values),
            "avg": sum(values) / len(values),
            "max": max(values),
        }

    return {
        "count": count,
        "keys": keys,
        "missing": missing_field_counts,
        "text_stats": _stats(text_counts),
        "img_stats": _stats(img_counts),
        "gold_stats": _stats(gold_counts),
        "evidence_modality": evidence_modality,
        "question_types": question_types,
    }


def summarize_train(path):
    count = 0
    keys = set()
    message_roles = Counter()
    message_counts = []

    for obj in stream_jsonl(path):
        breakpoint()
        count += 1
        keys.update(obj.keys())
        messages = obj.get("messages", [])
        message_counts.append(len(messages))
        for message in messages:
            role = message.get("role", "UNKNOWN")
            message_roles[role] += 1

    stats = {
        "min": min(message_counts) if message_counts else 0,
        "avg": (sum(message_counts) / len(message_counts)) if message_counts else 0.0,
        "max": max(message_counts) if message_counts else 0,
    }

    return {
        "count": count,
        "keys": keys,
        "message_roles": message_roles,
        "message_stats": stats,
    }


def format_counter(counter, limit=10):
    items = counter.most_common(limit)
    return ", ".join([f"{key}:{value}" for key, value in items]) or "none"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize MMDocRAG dataset splits and schema."
    )
    parser.add_argument(
        "--dataset_dir",
        default="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/MMDocRAG/dataset",
        help="Path to MMDocRAG dataset directory.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    print("MMDocRAG dataset audit")
    print(f"Directory: {dataset_dir}")
    print("-" * 80)

    train_path = os.path.join(dataset_dir, TRAIN_FILE)
    if os.path.exists(train_path):
        train_summary = summarize_train(train_path)
        print(f"{TRAIN_FILE}: {train_summary['count']} lines")
        print(f"  keys: {sorted(train_summary['keys'])}")
        print(
            "  message roles (top): "
            + format_counter(train_summary["message_roles"])
        )
        stats = train_summary["message_stats"]
        print(
            "  messages per record: "
            f"min={stats['min']} avg={stats['avg']:.2f} max={stats['max']}"
        )
    else:
        print(f"{TRAIN_FILE}: missing")

    print("-" * 80)

    for filename in NONTRAIN_FILES:
        path = os.path.join(dataset_dir, filename)
        if not os.path.exists(path):
            print(f"{filename}: missing")
            continue

        summary = summarize_nontrain(path)
        print(f"{filename}: {summary['count']} lines")
        print(f"  keys: {sorted(summary['keys'])}")
        if summary["missing"]:
            print("  missing fields:")
            for key, value in summary["missing"].most_common():
                print(f"    - {key}: {value}")
        else:
            print("  missing fields: none")

        t = summary["text_stats"]
        i = summary["img_stats"]
        g = summary["gold_stats"]
        print(
            "  text_quotes count: "
            f"min={t['min']} avg={t['avg']:.2f} max={t['max']}"
        )
        print(
            "  img_quotes count: "
            f"min={i['min']} avg={i['avg']:.2f} max={i['max']}"
        )
        print(
            "  gold_quotes count: "
            f"min={g['min']} avg={g['avg']:.2f} max={g['max']}"
        )
        print(
            "  evidence_modality_type (top): "
            + format_counter(summary["evidence_modality"])
        )
        print(
            "  question_type (top): "
            + format_counter(summary["question_types"])
        )
        print("-" * 80)


if __name__ == "__main__":
    main()
