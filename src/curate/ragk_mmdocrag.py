#!/usr/bin/env python3
"""
Convert MMDocRAG data to ShareGPT format for BEFT training.

Supports:
  - train.jsonl (messages format)
  - dev/evaluation jsonl (structured quotes format)
"""

import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple

from tqdm import tqdm


TEXT_QUOTE_RE = re.compile(
    r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\nImage Quotes are:|\Z)",
    re.DOTALL,
)
IMAGE_QUOTE_RE = re.compile(
    r"image(\d+)\s+is described as:\s*(.*?)(?=\nimage\d+\s+is described as:|\nThe user question is:|\Z)",
    re.DOTALL,
)
QUESTION_RE = re.compile(r"The user question is:\s*(.*)", re.DOTALL)
TEXT_CITATION_RE = re.compile(r"\[(\d+)\]")
IMAGE_CITATION_RE = re.compile(r"!\[[^\]]*\]\(image(\d+)\)")


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_messages_item(item: Dict) -> Tuple[str, List[Dict], List[Dict], List[str], str, str]:
    messages = item["messages"]
    system_prompt = messages[0]["content"]
    user_content = messages[1]["content"]
    gold_answer = messages[2]["content"]

    text_quotes = []
    for match in TEXT_QUOTE_RE.finditer(user_content):
        idx = match.group(1)
        text = normalize_text(match.group(2))
        text_quotes.append(
            {
                "quote_id": f"text{idx}",
                "type": "text",
                "text": text,
            }
        )

    img_quotes = []
    for match in IMAGE_QUOTE_RE.finditer(user_content):
        idx = match.group(1)
        desc = normalize_text(match.group(2))
        img_quotes.append(
            {
                "quote_id": f"image{idx}",
                "type": "image",
                "img_description": desc,
                "img_path": None,
            }
        )

    question_match = QUESTION_RE.search(user_content)
    question = normalize_text(question_match.group(1)) if question_match else ""

    gold_quotes = set()
    for t_idx in TEXT_CITATION_RE.findall(gold_answer):
        gold_quotes.add(f"text{t_idx}")
    for i_idx in IMAGE_CITATION_RE.findall(gold_answer):
        gold_quotes.add(f"image{i_idx}")

    return question, text_quotes, img_quotes, sorted(gold_quotes), gold_answer, system_prompt


def parse_structured_item(item: Dict) -> Tuple[str, List[Dict], List[Dict], List[str], str, str]:
    question = item["question"]
    text_quotes = item.get("text_quotes", [])
    img_quotes = item.get("img_quotes", [])
    gold_quotes = item.get("gold_quotes", [])
    gold_answer = item.get("answer_interleaved", "")
    prompt_path = "third_party/MMDocRAG/prompt_bank/multimodal_infer.txt" # use the prompt provided by MMDocRAG
    with open(prompt_path, "r", encoding="utf-8") as handle:
        system_prompt = handle.read().strip()
    return question, text_quotes, img_quotes, gold_quotes, gold_answer, system_prompt


def build_passages(
    text_quotes: List[Dict],
    img_quotes: List[Dict],
    dataset_dir: str,
) -> List[Dict]:
    passages = []
    for tq in text_quotes:
        quote_id = tq["quote_id"]
        match = re.search(r"\d+", quote_id)
        index = match.group(0) if match else ""
        prefix = f"[{index}] " if index else ""
        passages.append(
            {
                "quote_id": quote_id,
                "type": "text",
                "text": f"{prefix}{tq.get('text', '')}",
                "images": [],
            }
        )
    for iq in img_quotes:
        quote_id = iq["quote_id"]
        match = re.search(r"\d+", quote_id)
        index = match.group(0) if match else ""
        prefix = f"(image {index}) " if index else ""
        img_path = iq.get("img_path")
        if img_path:
            img_path = os.path.join(dataset_dir, img_path)
        passages.append(
            {
                "quote_id": quote_id,
                "type": "image",
                "text": f"{prefix}{iq.get('img_description', '')}",
                "images": [img_path] if img_path else [],
            }
        )
    return passages


def sample_passages(
    passages: List[Dict],
    gold_quotes: List[str],
    rng: random.Random,
    target_count: int,
) -> Tuple[List[Dict], List[int]]:
    passage_by_id = {p["quote_id"]: p for p in passages}
    gold_set = {qid for qid in gold_quotes if qid in passage_by_id}
    gold_passages = [passage_by_id[qid] for qid in gold_set]

    non_gold = [p for p in passages if p["quote_id"] not in gold_set]
    non_gold_text = [p for p in non_gold if p["type"] == "text"]
    non_gold_img = [p for p in non_gold if p["type"] == "image"]

    selected = []

    if len(gold_passages) >= target_count:
        gold_img = [p for p in gold_passages if p["type"] == "image"]
        gold_text = [p for p in gold_passages if p["type"] == "text"]
        if len(gold_img) >= target_count:
            selected = rng.sample(gold_img, target_count)
        else:
            selected = gold_img + rng.sample(gold_text, min(target_count - len(gold_img), len(gold_text)))
            if len(selected) < target_count:
                remaining = [p for p in gold_passages if p not in selected]
                if remaining:
                    selected += rng.sample(remaining, min(target_count - len(selected), len(remaining)))
    else:
        selected = list(gold_passages)
        remaining = target_count - len(selected)
        if remaining >= 2:
            if non_gold_img:
                selected.append(rng.choice(non_gold_img))
            if non_gold_text:
                selected.append(rng.choice(non_gold_text))
            remaining = target_count - len(selected)
            remaining_pool = [p for p in non_gold if p not in selected]
            if remaining_pool and remaining > 0:
                selected += rng.sample(remaining_pool, min(remaining, len(remaining_pool)))
        elif remaining == 1:
            if non_gold_text:
                selected.append(rng.choice(non_gold_text))
            elif non_gold:
                selected.append(rng.choice(non_gold))

    if len(selected) < target_count:
        remaining_pool = [p for p in passages if p not in selected]
        if remaining_pool:
            selected += rng.sample(remaining_pool, min(target_count - len(selected), len(remaining_pool)))

    rng.shuffle(selected)
    gt_indices = [i for i, p in enumerate(selected) if p["quote_id"] in gold_set]
    return selected, gt_indices


def convert_to_sharegpt(dataset: List[Dict], output_dir: str) -> None:
    sharegpt_data = []
    for item in dataset:
        conversation = {
            "messages": [
                {"content": item["system_prompt"], "role": "system"},
                {"content": f"{item['prompt']}", "role": "user"},
                {"content": item["gold_answer"], "role": "assistant"},
            ],
            "images": [],
            "gt_passage_idx": item["gt_passage_idx"],
            "passages": item["passages"],
            "passage_scores": item["passage_scores"],
        }
        sharegpt_data.append(conversation)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_sharegpt.json"), "w", encoding="utf-8") as handle:
        json.dump(sharegpt_data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Optional override for input jsonl (otherwise uses dev/train based on mode).",
    )
    parser.add_argument(
        "--do_multimodal_training",
        action="store_true",
        help="If set, use dev set only (multimodal). Otherwise, use train set only.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/vqa_data/MMDocRAG/dataset",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--topk_docs", type=int, default=8)
    parser.add_argument("--sanity_check_count", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.input_jsonl:
        input_path = args.input_jsonl
    else:
        input_name = "dev_20.jsonl" if args.do_multimodal_training else "train.jsonl"
        input_path = os.path.join(args.dataset_dir, input_name)

    data = load_jsonl(input_path)
    rng.shuffle(data)
    if args.sample_size > 0:
        data = data[:args.sample_size]
    elif args.sample_offset > 0:
        data = data[args.sample_offset:]

    processed = []
    for item in tqdm(data, desc="Processing MMDocRAG"):
        if "messages" in item:
            question, text_quotes, img_quotes, gold_quotes, gold_answer, system_prompt = parse_messages_item(item)
            doc_name = None
        else:
            question, text_quotes, img_quotes, gold_quotes, gold_answer, system_prompt = parse_structured_item(item)
            doc_name = item.get("doc_name")

        passages = build_passages(text_quotes, img_quotes, args.dataset_dir)
        total_passages = len(passages)
        selected, gt_indices = sample_passages(passages, gold_quotes, rng, args.topk_docs)

        prompt = f"[QUESTION] {question}\n<<<EVIDENCE>>>"

        processed.append(
            {
                "doc_name": doc_name,
                "question": question,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "passages": selected,
                "passage_scores": [0.0 for _ in selected],
                "gt_passage_idx": gt_indices,
                "gold_answer": gold_answer,
                "gold_quotes": gold_quotes,
                "total_passages": total_passages,
            }
        )

    if args.sanity_check_count > 0:
        for item in processed[: args.sanity_check_count]:
            passage_ids = [p["quote_id"] for p in item["passages"]]
            gt_ids = {passage_ids[i] for i in item["gt_passage_idx"]}
            expected_len = min(args.topk_docs, item["total_passages"])
            assert len(item["passages"]) == expected_len
            for gt_idx in item["gt_passage_idx"]:
                assert 0 <= gt_idx < len(item["passages"])
            missing = [
                qid
                for qid in item["gold_quotes"]
                if qid in passage_ids and qid not in gt_ids
            ]
            assert not missing

    convert_to_sharegpt(processed, args.output_dir)
    print(f"Saved ShareGPT data to {args.output_dir}")


if __name__ == "__main__":
    main()
