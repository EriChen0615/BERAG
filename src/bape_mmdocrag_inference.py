import sys
sys.path.append('./src')

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from hf_backend import HFQwen2VLBackend
from bape_inference_engine import BAPEInferenceEngine


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_passages(item: Dict, dataset_dir: str) -> List[Dict]:
    passages = []
    for tq in item.get("text_quotes", []) or []:
        quote_id = tq["quote_id"]
        index = "".join([c for c in quote_id if c.isdigit()])
        prefix = f"[{index}] " if index else ""
        passages.append(
            {
                "quote_id": quote_id,
                "type": "text",
                "text": f"{prefix}{tq.get('text', '')}",
                "images": [],
            }
        )
    for iq in item.get("img_quotes", []) or []:
        quote_id = iq["quote_id"]
        index = "".join([c for c in quote_id if c.isdigit()])
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


def read_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def make_prompt(prompt_template: str, question: str) -> str:
    return f"{prompt_template}\n\n[QUESTION] {question}\n<<<EVIDENCE>>>"


def process_dataset_with_bape(engine, data, args):
    all_results = []
    for item in tqdm(data, total=len(data), desc="MMDocRAG inference"):
        prompt_text = make_prompt(args.prompt_template, item["question"])
        x = {"text": prompt_text, "image": None}

        passages = build_passages(item, args.dataset_dir)
        passage_inputs = [{"text": p["text"], "images": p["images"]} for p in passages]
        passage_ids = [p["quote_id"] for p in passages]

        passage_prior = None
        if args.passage_prior == "log_softmax":
            passage_scores = torch.zeros(len(passages), device=engine.device)
            passage_prior = torch.log_softmax(passage_scores, dim=0)
        elif args.passage_prior == "uniform":
            passage_prior = torch.ones(len(passages), device=engine.device) / max(1, len(passages))
        elif args.passage_prior == "prior_head":
            passage_prior = None
        else:
            raise NotImplementedError(f"Passage prior {args.passage_prior} not implemented")

        generated_token_ids, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits = engine.generate(
            x, passage_inputs, passage_prior, max_new_tokens=args.max_new_tokens
        )
        response = engine.backend.processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        generated_answer = response.split("[ANSWER]")[-1].strip()

        prior_sorted_passage_ids = []
        if prior_logits is not None and len(prior_logits) > 0:
            prior_sorted_indices = np.argsort(prior_logits)[::-1]
            prior_sorted_passage_ids = [passage_ids[i] for i in prior_sorted_indices]

        all_results.append(
            {
                "q_id": item["q_id"],
                "response": response,
                "generated_answer": generated_answer,
                "passages": passages,
                "prompt_text": prompt_text,
                "posterior_max_idx": posterior_max_idx,
                "prior_max_idx": prior_max_idx,
                "log_all_tokens_llk": log_all_tokens_llk,
                "prior_logits": prior_logits if prior_logits is not None else None,
                "prior_sorted_passage_ids": prior_sorted_passage_ids,
            }
        )

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="20", choices=["15", "20"])
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Override path to evaluation jsonl. If not set, uses dataset_dir/evaluation_{setting}.jsonl",
    )
    parser.add_argument("--prompt_template", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--take_n", type=int, default=-1)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, required=True)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)
    parser.add_argument("--prior_head_path", type=str, default=None)
    parser.add_argument("--prior_head_modeling", type=str, default="mlp_head", choices=["mlp_head", "linear_head"])
    parser.add_argument("--prior_head_num_layers", type=int, default=2)
    parser.add_argument("--prior_head_proj_dim", type=int, default=1024)
    parser.add_argument("--hidden_state_offset", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--passage_prior", type=str, default="prior_head")
    parser.add_argument("--max_batch_size_per_forward", type=int, default=5)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    args = parser.parse_args()

    if args.input_jsonl:
        eval_path = args.input_jsonl
    else:
        eval_path = os.path.join(args.dataset_dir, f"evaluation_{args.setting}.jsonl")
    data = load_jsonl(eval_path)
    n_multimodal = sum(1 for item in data if (item.get("img_quotes") or []))
    print(f"Loaded {len(data)} items from {eval_path} ({n_multimodal} with at least one img_quote)")
    if args.take_n > 0:
        data = data[: args.take_n]

    backend = HFQwen2VLBackend(
        model_path=args.model_path,
        processor_path=args.processor_path,
        adapter_name_or_path=args.adapter_name_or_path,
        max_batch_size_per_forward=args.max_batch_size_per_forward,
        attn_implementation=args.attn_implementation,
    )
    prior_head_config = {
        "modeling": args.prior_head_modeling,
        "num_layers": args.prior_head_num_layers,
        "proj_dim": args.prior_head_proj_dim,
    }
    engine = BAPEInferenceEngine(
        backend=backend,
        prior_head_path=args.prior_head_path,
        prior_head_config=prior_head_config,
        hidden_state_offset=args.hidden_state_offset,
    )

    results = process_dataset_with_bape(engine, data, args)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps({"q_id": item["q_id"], "response": item["response"]}, ensure_ascii=False) + "\n")

    print(f"Saved responses to {args.output_path}")


if __name__ == "__main__":
    main()
