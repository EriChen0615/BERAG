import sys
sys.path.append('./src')

import argparse
import collections
import json
import multiprocessing
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

from hf_backend import HFQwen2VLBackend


VLM_PROMPT_FOR_VQA = (
    "Answer the question directly without explanations based on the provided slides."
    "<<<EVIDENCE>>>"
)


def compute_token_f1(gold_answer: str, generated_answer: str) -> float:
    def _normalize(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    gold_tokens = _normalize(gold_answer).split()
    pred_tokens = _normalize(generated_answer).split()

    if len(gold_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


class RegularRAGSlideVQAInferenceEngine:
    def __init__(self, backend):
        self.backend = backend
        self.device = backend.device

    @staticmethod
    def _build_generation_stats(prefill_ms, decode_ms, decode_tokens, input_tokens, output_tokens):
        decode_tokens_per_ms = (decode_tokens / decode_ms) if decode_ms > 0 else 0.0
        return {
            "prefill_ms": float(prefill_ms),
            "decode_ms": float(decode_ms),
            "decode_tokens": int(decode_tokens),
            "decode_tokens_per_ms": float(decode_tokens_per_ms),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
        }

    @torch.no_grad()
    def generate(self, input_context, merged_passage, max_new_tokens=512):
        generated_tokens = []
        log_all_tokens_llk = []

        prefill_inputs = self.backend.prepare_input(input_context, generated_tokens, merged_passage)
        input_tokens = int(prefill_inputs["input_ids"].shape[1])

        prefill_start = time.perf_counter()
        log_probs, past_key_values, _ = self.backend.forward(prefill_inputs, past_key_values=None, return_hidden_states=False)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        decode_ms = 0.0

        token_idx = torch.argmax(log_probs[0]).item()
        log_all_tokens_llk.append(log_probs[0, token_idx].item())
        generated_tokens.append(token_idx)

        while len(generated_tokens) < max_new_tokens:
            if self.backend.is_stop_token(token_idx):
                break

            decode_inputs = self.backend.prepare_input(
                input_context,
                generated_tokens,
                merged_passage,
                past_key_values=past_key_values,
            )
            decode_start = time.perf_counter()
            log_probs, past_key_values, _ = self.backend.forward(
                decode_inputs,
                past_key_values=past_key_values,
                return_hidden_states=False,
            )
            decode_ms += (time.perf_counter() - decode_start) * 1000.0

            token_idx = torch.argmax(log_probs[0]).item()
            log_all_tokens_llk.append(log_probs[0, token_idx].item())
            generated_tokens.append(token_idx)

        stats = self._build_generation_stats(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            decode_tokens=max(len(generated_tokens) - 1, 0),
            input_tokens=input_tokens,
            output_tokens=len(generated_tokens),
        )
        return generated_tokens, log_all_tokens_llk, stats


def get_prompt_template(prompt_template: str):
    if prompt_template is None or prompt_template == '':
        return VLM_PROMPT_FOR_VQA
    with open(prompt_template, 'r') as f:
        return f.read()


def load_image_from_path(img_path):
    try:
        if isinstance(img_path, Image.Image):
            return img_path.convert('RGB') if img_path.mode != 'RGB' else img_path
        if isinstance(img_path, str):
            if os.path.exists(img_path):
                return Image.open(img_path).convert('RGB')
            return None
        return None
    except Exception:
        return None


def write_inference_report(df: pd.DataFrame, output_dir: str, total_inference_runtime_sec=None):
    metric_cols = [
        "prefill_ms",
        "decode_ms",
        "decode_tokens_per_ms",
        "input_tokens",
        "output_tokens",
        "decode_tokens",
    ]
    numeric_df = df.copy()
    for col in metric_cols:
        if col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    if "num_prefill_branches" in numeric_df.columns:
        numeric_df["num_prefill_branches"] = pd.to_numeric(numeric_df["num_prefill_branches"], errors="coerce").fillna(1.0)
    else:
        numeric_df["num_prefill_branches"] = 1.0
    numeric_df["input_tokens_total"] = numeric_df["input_tokens"] * numeric_df["num_prefill_branches"]
    timing_df = numeric_df.iloc[1:].copy() if len(numeric_df) > 1 else numeric_df.copy()
    if "decode_ms" in timing_df.columns and "decode_tokens" in timing_df.columns:
        per_token_decode_ms = (
            timing_df.loc[timing_df["decode_tokens"] > 0, "decode_ms"]
            / timing_df.loc[timing_df["decode_tokens"] > 0, "decode_tokens"]
        )
        avg_per_token_decode_ms = float(per_token_decode_ms.mean()) if len(per_token_decode_ms) > 0 else 0.0
    else:
        avg_per_token_decode_ms = 0.0

    report = {
        "num_examples": int(len(numeric_df)),
        "num_examples_for_timing": int(len(timing_df)),
        "num_valid_decode_examples": int(timing_df["decode_tokens_per_ms"].notna().sum()) if "decode_tokens_per_ms" in timing_df.columns else 0,
        "avg_prefill_ms": float(timing_df["prefill_ms"].mean()) if "prefill_ms" in timing_df.columns else 0.0,
        "avg_decode_tokens_per_ms": float(timing_df["decode_tokens_per_ms"].mean()) if "decode_tokens_per_ms" in timing_df.columns else 0.0,
        "avg_input_tokens_per_branch": float(numeric_df["input_tokens"].mean()) if "input_tokens" in numeric_df.columns else 0.0,
        "avg_input_tokens_total": float(numeric_df["input_tokens_total"].mean()) if "input_tokens_total" in numeric_df.columns else 0.0,
        "avg_output_tokens": float(numeric_df["output_tokens"].mean()) if "output_tokens" in numeric_df.columns else 0.0,
        "avg_decode_ms": float(timing_df["decode_ms"].mean()) if "decode_ms" in timing_df.columns else 0.0,
        "avg_per_token_decode_ms": avg_per_token_decode_ms,
        "avg_decode_tokens": float(timing_df["decode_tokens"].mean()) if "decode_tokens" in timing_df.columns else 0.0,
        "total_inference_runtime_sec": float(total_inference_runtime_sec) if total_inference_runtime_sec is not None else None,
    }
    with open(f"{output_dir}/inference_report.json", "w") as f:
        json.dump(report, f, indent=2)


def process_dataset_with_rag(engine, slidevqa_dataset, max_new_tokens, args):
    all_results = []
    split_name = args.split if args.split else 'test'
    prompt_template_str = get_prompt_template(args.prompt_template)

    for item in tqdm(slidevqa_dataset, total=len(slidevqa_dataset)):
        qa_id = item['qa_id']
        question = item['question']
        gold_answer = item['answer']

        if args.use_oracle_slides:
            evidence_pages = item.get('evidence_pages', [])
            if not isinstance(evidence_pages, list):
                evidence_pages = [evidence_pages] if evidence_pages is not None else []
            pages_to_collect = []
            for ev_page in evidence_pages:
                if isinstance(ev_page, int):
                    pages_to_collect.append(ev_page)
                elif isinstance(ev_page, str) and ev_page.startswith('page_'):
                    pages_to_collect.append(int(ev_page.split('_')[1]))
                else:
                    try:
                        pages_to_collect.append(int(ev_page))
                    except Exception:
                        continue
            if len(pages_to_collect) == 0:
                pages_to_collect = list(range(1, 21))
        else:
            pages_to_collect = list(range(1, 21))

        passages = []
        passage_page_nums = []
        for page_num in pages_to_collect:
            page_key = f'page_{page_num}'
            if page_key not in item or item[page_key] is None:
                continue
            page_image = item[page_key]
            if isinstance(page_image, Image.Image):
                image_to_use = page_image.convert('RGB') if page_image.mode != 'RGB' else page_image
            elif isinstance(page_image, str):
                candidate_path = page_image if os.path.isabs(page_image) else os.path.join(args.img_basedir, page_image)
                image_to_use = load_image_from_path(candidate_path)
                if image_to_use is None:
                    png_path = os.path.join(args.img_basedir, f"{split_name}_{qa_id}_page_{page_num}.png")
                    jpg_path = os.path.join(args.img_basedir, f"{split_name}_{qa_id}_page_{page_num}.jpg")
                    image_to_use = load_image_from_path(png_path) or load_image_from_path(jpg_path)
            else:
                image_to_use = None

            if image_to_use is None:
                continue

            passages.append({'images': [image_to_use], 'text': '', 'page_num': page_num})
            passage_page_nums.append(page_num)

        if len(passages) > args.retrieval_topk:
            passages = passages[:args.retrieval_topk]
            passage_page_nums = passage_page_nums[:args.retrieval_topk]
        if len(passages) == 0:
            continue

        evidence_pages = item.get('evidence_pages', [])
        if not isinstance(evidence_pages, list):
            evidence_pages = [evidence_pages] if evidence_pages is not None else []
        evidence_page_nums = []
        for ev_page in evidence_pages:
            if isinstance(ev_page, int):
                evidence_page_nums.append(ev_page)
            elif isinstance(ev_page, str) and ev_page.startswith('page_'):
                evidence_page_nums.append(int(ev_page.split('_')[1]))
            else:
                try:
                    evidence_page_nums.append(int(ev_page))
                except Exception:
                    continue

        merged_images = []
        for p in passages:
            merged_images.extend(p['images'])
        merged_passage = {"images": merged_images, "text": ""}

        gt_passage_in_zidx = -1
        if evidence_page_nums:
            for ev_page_num in evidence_page_nums:
                if ev_page_num in passage_page_nums:
                    gt_passage_in_zidx = passage_page_nums.index(ev_page_num)
                    break

        prompt_text = prompt_template_str
        prompt_text += f"\n[QUESTION] {question}"
        if args.prefill_ans_token:
            prompt_text += "\n[ANSWER]"
        x = {"text": prompt_text, "image": None}

        generated_token_ids, log_all_tokens_llk, inference_stats = engine.generate(
            x, merged_passage, max_new_tokens=max_new_tokens
        )
        response = engine.backend.processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        generated_answer = response.split('[ANSWER]')[-1].strip() if '[ANSWER]' in response else response.strip()

        all_results.append({
            'qa_id': qa_id,
            'question': question,
            'gold_answer': gold_answer,
            'response': response,
            'generated_answer': generated_answer,
            'prompt_text': prompt_text,
            'passage_page_nums': passage_page_nums,
            'evidence_page_nums': evidence_page_nums,
            'gt_passage_in_zidx': gt_passage_in_zidx,
            'log_all_tokens_llk': log_all_tokens_llk,
            'prefill_ms': inference_stats.get('prefill_ms', 0.0),
            'decode_ms': inference_stats.get('decode_ms', 0.0),
            'decode_tokens': inference_stats.get('decode_tokens', 0),
            'decode_tokens_per_ms': inference_stats.get('decode_tokens_per_ms', 0.0),
            'input_tokens': inference_stats.get('input_tokens', 0),
            'output_tokens': inference_stats.get('output_tokens', len(generated_token_ids)),
            'num_prefill_branches': 1,
            'retrieval_topk': args.retrieval_topk,
        })
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_path", type=str, default="NTT-hil-insight/SlideVQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--take_n", type=int, default=-1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--img_basedir", type=str, default="../../shared_space/vqa_data/KBVQA_data/SlideVQA")
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--prefill_ans_token", action="store_true")
    parser.add_argument("--use_oracle_slides", action="store_true")
    parser.add_argument("--retrieval_topk", type=int, default=20)

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)
    parser.add_argument("--max_batch_size_per_forward", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--use_bem", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_filepath = f"{args.exp_name}/inference_results.csv"
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    total_inference_runtime_sec = None

    if os.path.exists(output_filepath) and args.use_cache:
        all_results = pd.read_csv(output_filepath)
    else:
        try:
            slidevqa_dataset = load_from_disk(args.hf_dataset_path)
            if args.split in slidevqa_dataset:
                slidevqa_dataset = slidevqa_dataset[args.split]
        except Exception:
            slidevqa_dataset = load_dataset(args.hf_dataset_path, split=args.split)

        if args.take_n > 0:
            slidevqa_dataset = slidevqa_dataset.shuffle(seed=args.seed).select([i for i in range(args.offset, args.offset + args.take_n)])

        backend = HFQwen2VLBackend(
            model_path=args.model_path,
            processor_path=args.processor_path if args.processor_path else args.model_path,
            adapter_name_or_path=args.adapter_name_or_path,
            max_batch_size_per_forward=args.max_batch_size_per_forward,
        )
        engine = RegularRAGSlideVQAInferenceEngine(backend)

        inference_start_time = time.perf_counter()
        all_results = process_dataset_with_rag(engine, slidevqa_dataset, max_new_tokens=args.max_new_tokens, args=args)
        total_inference_runtime_sec = time.perf_counter() - inference_start_time
        all_results = pd.DataFrame(all_results)
        all_results.to_csv(output_filepath, index=False)
        del engine
        torch.cuda.empty_cache()

    if not isinstance(all_results, pd.DataFrame):
        all_results = pd.DataFrame(all_results)
    write_inference_report(all_results, os.path.dirname(output_filepath), total_inference_runtime_sec=total_inference_runtime_sec)

    if args.do_eval:
        df = all_results.copy()

        def relaxed_exact_match(gold_answer: str, generated_answer: str) -> bool:
            if not gold_answer or not generated_answer:
                return False
            gold_normalized = str(gold_answer).lower().strip()
            generated_normalized = str(generated_answer).lower().strip()
            return gold_normalized in generated_normalized

        exact_matches = []
        token_f1_scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing Relaxed Exact Match scores"):
            gold_answer = row.get('gold_answer', '')
            generated_answer = row.get('generated_answer', '')
            exact_matches.append(1 if relaxed_exact_match(gold_answer, generated_answer) else 0)
            token_f1_scores.append(compute_token_f1(gold_answer, generated_answer))
        df['exact_match'] = exact_matches
        df['token_f1'] = token_f1_scores

        dict_to_report = {
            'relaxed_exact_match_accuracy': float(np.mean(exact_matches)) if exact_matches else 0.0,
            'token_f1': float(np.mean(token_f1_scores)) if token_f1_scores else 0.0,
            'total_samples': len(exact_matches),
            'correct_predictions': int(sum(exact_matches))
        }

        # Optional BEM (same dependency usage as bape_slidevqa_inference)
        if args.use_bem:
            try:
                from evaluation_utils import evaluate_example
                bem_scores = []
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing BEM scores"):
                    bem_scores.append(
                        evaluate_example(
                            question=row.get('question', ''),
                            reference_list=[str(row.get('gold_answer', ''))],
                            candidate=str(row.get('generated_answer', '')),
                            question_type='string',
                        )
                    )
                df['bem_score'] = bem_scores
                dict_to_report['bem_accuracy'] = float(np.mean(bem_scores)) if bem_scores else 0.0
            except Exception as e:
                dict_to_report['bem_error'] = str(e)

        df.to_csv(f"{os.path.dirname(output_filepath)}/marked_inference_results.csv", index=False)
        with open(f"{os.path.dirname(output_filepath)}/scores.json", 'w') as f:
            json.dump(dict_to_report, f, indent=2)
