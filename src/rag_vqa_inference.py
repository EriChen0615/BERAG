import sys
sys.path.append('./src')

import argparse
import ast
import json
import multiprocessing
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm

from hf_backend import HFQwen2VLBackend
from rag_inference_engine import RegularRAGInferenceEngine
from vqa_datasets import load_passages


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda out of memory" in msg)


def get_prompt_template(prompt_template: str):
    if prompt_template is None or prompt_template == '':
        return (
            "Answer the question after [QUESTION] about the image."
            "A retriever has retrieved relevant documents for you and provided them after [EVIDENCE]."
            "Give your answer after [ANSWER]\n"
        )
    with open(prompt_template, 'r') as f:
        return f.read()


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
    print(f"Saved inference report to {output_dir}/inference_report.json")


def process_dataset_with_rag(engine, vqa_dataset, max_new_tokens, args, pid_to_content_map, prompt_template):
    def _make_evidence_part(passage_dict):
        pid = passage_dict['passage_id']
        if pid == 'z0':
            return "[EVIDENCE] No passage provided.\n"
        text = pid_to_content_map[pid]
        text = ' '.join(text.split(' ')[:args.max_words_per_evidence])
        return (
            "[EVIDENCE] "
            f"Title: {pid}\t"
            f"Content: {text}\n"
        )

    def _make_prompt(question):
        prompt = prompt_template + "<<<EVIDENCE>>>"
        prompt += f"\n[QUESTION] {question}"
        if args.prefill_ans_token:
            prompt += "\n[ANSWER]"
        return prompt

    all_results = []
    for item in tqdm(vqa_dataset, total=len(vqa_dataset)):
        score_field = 'score' if args.retrieval_field == 'retrieved_passage' else 'rerank_score'
        z = item[args.retrieval_field][:args.retrieval_topk]
        retrieved_passage_ids = [zk['passage_id'] for zk in z]
        gt_passage_id = item['pos_item_ids'][0]

        if args.ensure_gt_passage_in_ensemble and gt_passage_id not in retrieved_passage_ids and len(z) > 0:
            z[0] = {'passage_id': item['pos_item_ids'][0], 'passage_content': item['pos_item_contents'][0], score_field: 1.0}
            retrieved_passage_ids = [zk['passage_id'] for zk in z]

        passage_contents = [_make_evidence_part(zk) for zk in z]
        prompt_text = _make_prompt(item['question'])
        x = {"text": prompt_text, "image": os.path.join(args.img_basedir, item['img_path'])}

        try:
            generated_token_ids, log_all_tokens_llk, inference_stats = engine.generate(
                x,
                passage_contents,
                max_new_tokens=max_new_tokens,
            )
            response = engine.backend.processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            generated_answer = response.split('[ANSWER]')[-1].strip()
        except RuntimeError as e:
            if _is_cuda_oom_error(e):
                print(
                    f"[OOM-Recover] question_id={item.get('question_id', 'unknown')} "
                    f"retrieval_topk={args.retrieval_topk}; returning fallback answer."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                generated_token_ids = []
                log_all_tokens_llk = []
                response = "Don't know"
                generated_answer = "Don't know"
                inference_stats = {
                    "prefill_ms": 0.0,
                    "decode_ms": 0.0,
                    "decode_tokens": 0,
                    "decode_tokens_per_ms": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            else:
                raise

        gt_passage_in_zidx = retrieved_passage_ids.index(gt_passage_id) if gt_passage_id in retrieved_passage_ids else -1
        all_results.append({
            'question_id': item['question_id'],
            'question': item['question'],
            'question_type': item.get('question_type', ''),
            'img_path': item['img_path'],
            'image_id': item['img_id'] if 'img_id' in item else item['image_id'],
            'gold_answer': item['gold_answer'],
            'answers': item['answers'],
            'response': response,
            'generated_answer': generated_answer,
            'passages': z,
            'prompt_text': prompt_text,
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
    parser.add_argument("--retrieval_ds_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="EVQA")
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--take_n", type=int, default=-1)
    parser.add_argument("--img_basedir", type=str, default='.')
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--prefill_ans_token", action="store_true")
    parser.add_argument("--max_words_per_evidence", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=0)

    parser.add_argument("--retrieval_field", type=str, default='retrieved_passage', choices=['retrieved_passage', 'reranked_passage'])
    parser.add_argument("--retrieval_topk", type=int, default=5)
    parser.add_argument("--ensure_gt_passage_in_ensemble", action="store_true")

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)
    parser.add_argument("--max_batch_size_per_forward", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_filepath = f"{args.exp_name}/inference_results.csv"
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    total_inference_runtime_sec = None

    vqa_dataset = load_from_disk(args.retrieval_ds_path)
    if os.path.exists(output_filepath) and args.use_cache:
        all_results = pd.read_csv(output_filepath)
    else:
        if args.take_n > 0:
            print(f"Taking {args.take_n} examples from the dataset, starting from index {args.offset}.")
            vqa_dataset = vqa_dataset.shuffle(seed=42).select([i for i in range(args.offset, args.offset + args.take_n)])
        else:
            print("Using the entire dataset.")

        passages, pid_to_content_map = load_passages(args.dataset_name, split=args.split)
        prompt_template = get_prompt_template(args.prompt_template)

        backend = HFQwen2VLBackend(
            model_path=args.model_path,
            processor_path=args.processor_path if args.processor_path else args.model_path,
            adapter_name_or_path=args.adapter_name_or_path,
            max_batch_size_per_forward=args.max_batch_size_per_forward,
        )
        engine = RegularRAGInferenceEngine(backend)

        inference_start_time = time.perf_counter()
        all_results = process_dataset_with_rag(
            engine=engine,
            vqa_dataset=vqa_dataset,
            max_new_tokens=args.max_new_tokens,
            args=args,
            pid_to_content_map=pid_to_content_map,
            prompt_template=prompt_template,
        )
        total_inference_runtime_sec = time.perf_counter() - inference_start_time
        all_results = pd.DataFrame(all_results)
        all_results.to_csv(output_filepath, index=False)
        del engine
        torch.cuda.empty_cache()

    if not isinstance(all_results, pd.DataFrame):
        all_results = pd.DataFrame(all_results)
    write_inference_report(
        all_results,
        os.path.dirname(output_filepath),
        total_inference_runtime_sec=total_inference_runtime_sec,
    )

    if args.do_eval:
        if args.dataset_name == "EVQA":
            sys.path.append('./src/evaluation')
            from evqa_eval_1004 import process_row_mp as eval_process_row_mp

            df = all_results.copy()
            df['answers'] = df['answers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df['prediction'] = df['generated_answer']

            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            multiprocessing.set_start_method('spawn', force=True)

            with multiprocessing.Pool(processes=8) as pool:
                all_eval_results = list(tqdm(pool.imap(eval_process_row_mp, df.iterrows(), chunksize=1), total=len(df)))

            dict_to_report = {"avg_score": float(sum(all_eval_results) / len(all_eval_results))}
            df['score'] = all_eval_results

            df.to_csv(f"{os.path.dirname(output_filepath)}/marked_inference_results.csv", index=False)
            with open(f"{os.path.dirname(output_filepath)}/scores.json", 'w') as f:
                json.dump(dict_to_report, f, indent=2)
            print("Evaluation results saved to", os.path.dirname(output_filepath))
        elif args.dataset_name in ['Infoseek', 'InfoseekNew', 'InfoseekNew_FullPassage']:
            sys.path.append("./third_party/infoseek_eval")
            df = all_results.copy()
            predictions = []
            df['prediction'] = df['generated_answer']

            df.to_csv(f"{os.path.dirname(output_filepath)}/inference_results_with_predictions.csv", index=False)
            print(f"Saved inference results (before scoring) to {os.path.dirname(output_filepath)}/inference_results_with_predictions.csv")

            for item in df.itertuples():
                predictions.append({
                    'data_id': item.question_id,
                    'prediction': item.prediction
                })

            pred_path = f"{os.path.dirname(output_filepath)}/predictions.jsonl"
            with open(pred_path, 'w') as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + '\n')

            if args.split in ['test', 'valid', 'valid_m2kr']:
                reference_path = "third_party/infoseek_eval/infoseek/infoseek_val.jsonl"
                reference_qtype_path = "third_party/infoseek_eval/infoseek/infoseek_val_qtype.jsonl"
            elif args.split in ['train']:
                reference_path = "third_party/infoseek_eval/infoseek/infoseek_train.jsonl"
                reference_qtype_path = None
            else:
                raise ValueError(f"Unknown split: {args.split}")

            from infoseek_eval import evaluate
            result = evaluate(pred_path, reference_path, reference_qtype_path)
            score_report = result.get('final_score', {})

            score_dict = {
                "score": score_report.get("score", 0.0),
                "score_num": score_report.get("score_num", 0.0),
                "acc_score": score_report.get("acc_score", 0.0),
            }

            answer_type_to_score = {
                "string": score_report.get("answer_type_score", {}).get("String", 0.0),
                "numerical": score_report.get("answer_type_score", {}).get("Numerical", 0.0),
                "time": score_report.get("answer_type_score", {}).get("Time", 0.0),
            }
            answer_type_to_score_num = {
                "string": score_report.get("answer_type_score_num", {}).get("String", 0.0),
                "numerical": score_report.get("answer_type_score_num", {}).get("Numerical", 0.0),
                "time": score_report.get("answer_type_score_num", {}).get("Time", 0.0),
            }
            score_dict['answer_type_to_score'] = answer_type_to_score
            score_dict['answer_type_to_score_num'] = answer_type_to_score_num

            # Maintain similarity with bape_vqa_inference outputs
            df['score'] = np.nan
            df.to_csv(f"{os.path.dirname(output_filepath)}/marked_inference_results.csv", index=False)
            with open(f"{os.path.dirname(output_filepath)}/scores.json", 'w') as f:
                json.dump(score_dict, f, indent=2)
            print("Evaluation results saved to", os.path.dirname(output_filepath))
        else:
            raise NotImplementedError(f"Evaluation for dataset {args.dataset_name} is not implemented.")
