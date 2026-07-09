#!/usr/bin/env python3
"""Prompt-based baseline for predicting GT document presence in retrieved passages."""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


DEFAULT_OUTPUT_ROOT = "analysis/output/predict_retrieval_success"

PROMPT_TEMPLATE = """You are given an image, a question, and retrieved document(s).

Question: {question}

Document:
{passage_text}

Does the document contain sufficient information to answer the question?
Reply with only "yes" or "no"."""


def parse_list_like(val) -> list:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def flatten_passage_text(passages: list) -> str:
    texts = []
    for passage in passages:
        if isinstance(passage, dict):
            text = passage.get("text", "")
            if text:
                texts.append(str(text).strip())
        elif isinstance(passage, str) and passage.strip():
            texts.append(passage.strip())
    return "\n\n".join(texts)


def safe_gt_label(gt_passage_in_zidx) -> int:
    try:
        idx = int(float(gt_passage_in_zidx))
    except (TypeError, ValueError):
        return 0
    return 1 if idx != -1 else 0


def resolve_img_path(img_path: str, img_basedir: str) -> str:
    if not img_path or (isinstance(img_path, float) and pd.isna(img_path)):
        return ""
    img_path = str(img_path)
    if os.path.isabs(img_path):
        return img_path
    return str((Path(img_basedir) / img_path).resolve())


def build_prompt(question: str, passage_text: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip(), passage_text=passage_text.strip())


def parse_yes_no_response(raw_response: str) -> Optional[int]:
    if raw_response is None:
        return None
    text = str(raw_response).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    if text.startswith("yes"):
        return 1
    if text.startswith("no"):
        return 0
    return None


def model_slug(model_path: str) -> str:
    name = model_path.rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def experiment_basename(csv_path: str) -> str:
    return Path(csv_path).parent.name


def load_inference_rows(csv_path: str, img_basedir: str, take_n: int = -1) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if take_n > 0:
        df = df.head(take_n)

    rows = []
    for _, row in df.iterrows():
        passages = parse_list_like(row.get("passages"))
        passage_text = flatten_passage_text(passages)
        question_id = str(row["question_id"])
        rows.append(
            {
                "question_id": question_id,
                "question": str(row.get("question", "")),
                "img_path": resolve_img_path(row.get("img_path", ""), img_basedir),
                "passage_text": passage_text,
                "label": safe_gt_label(row.get("gt_passage_in_zidx")),
                "prompt": build_prompt(str(row.get("question", "")), passage_text),
            }
        )
    return rows


def load_cached_predictions(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    cached: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return cached
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cached[str(record["question_id"])] = record
    return cached


def append_prediction(jsonl_path: Path, record: Dict[str, Any]) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = [r for r in records if r.get("decision") in (0, 1)]
    labels = [int(r["label"]) for r in parsed]
    decisions = [int(r["decision"]) for r in parsed]

    tp = sum(1 for y, p in zip(labels, decisions) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, decisions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, decisions) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, decisions) if y == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(parsed) if parsed else 0.0

    return {
        "total_samples": len(records),
        "evaluated_samples": len(parsed),
        "unparsed_count": len(records) - len(parsed),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def run_vllm_inference(
    pending_rows: List[Dict[str, Any]],
    engine: Any,
    batch_size: int,
    jsonl_path: Path,
) -> None:
    for start in tqdm(range(0, len(pending_rows), batch_size), desc="vLLM batches"):
        batch = pending_rows[start : start + batch_size]
        requests = []
        valid_rows = []

        for row in batch:
            image = engine.load_image_from_dataloader(row["img_path"])
            if image is None:
                record = {
                    **row,
                    "raw_response": "",
                    "decision": None,
                    "error": f"Missing image: {row['img_path']}",
                }
                append_prediction(jsonl_path, record)
                continue

            formatted_prompt, filtered_image = engine._prepare_conversation_format(row["prompt"], image)
            multi_modal_data = engine._prepare_multimodal_data(filtered_image)
            requests.append({"prompt": formatted_prompt, "multi_modal_data": multi_modal_data})
            valid_rows.append(row)

        if not requests:
            continue

        outputs = engine.llm.generate(
            requests,
            engine.sampling_params,
            lora_request=engine.lora_request,
        )

        for row, output in zip(valid_rows, outputs):
            raw_response = output.outputs[0].text
            record = {
                "question_id": row["question_id"],
                "label": row["label"],
                "raw_response": raw_response,
                "decision": parse_yes_no_response(raw_response),
                "prompt": row["prompt"],
                "img_path": row["img_path"],
                "passage_text": row["passage_text"],
            }
            append_prediction(jsonl_path, record)


def run_openai_inference(
    pending_rows: List[Dict[str, Any]],
    model_path: str,
    max_new_tokens: int,
    jsonl_path: Path,
) -> None:
    from vlms import OpenAI_VLM  # noqa: E402

    client = OpenAI_VLM(
        model_path=model_path,
        generation_config={"temperature": 0, "max_new_tokens": max_new_tokens},
    )

    for row in tqdm(pending_rows, desc="OpenAI requests"):
        if not row["img_path"] or not os.path.exists(row["img_path"]):
            record = {
                **row,
                "raw_response": "",
                "decision": None,
                "error": f"Missing image: {row['img_path']}",
            }
            append_prediction(jsonl_path, record)
            continue

        raw_response = client.generate_response(row["prompt"], row["img_path"])
        record = {
            "question_id": row["question_id"],
            "label": row["label"],
            "raw_response": raw_response,
            "decision": parse_yes_no_response(raw_response),
            "prompt": row["prompt"],
            "img_path": row["img_path"],
            "passage_text": row["passage_text"],
        }
        append_prediction(jsonl_path, record)


def run_single_experiment(
    inference_csv: str,
    output_dir: str,
    model_backend: str,
    model_path: str,
    processor_path: Optional[str],
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    tensor_parallel_size: Optional[int],
    img_basedir: str,
    take_n: int,
    force_override: bool,
    k_value: Optional[int] = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "predictions.jsonl"

    if force_override and jsonl_path.exists():
        jsonl_path.unlink()

    all_rows = load_inference_rows(inference_csv, img_basedir=img_basedir, take_n=take_n)
    cached = load_cached_predictions(jsonl_path)
    pending = [row for row in all_rows if row["question_id"] not in cached]

    if pending:
        if model_backend == "vllm":
            from vllm_vqa_inference import VLLMInferenceEngine  # noqa: E402

            engine_kwargs = {
                "base_model_path": model_path,
                "processor_path": processor_path or model_path,
                "max_model_len": max_model_len,
                "max_tokens": max_tokens,
            }
            if tensor_parallel_size is not None:
                engine_kwargs["tensor_parallel_size"] = tensor_parallel_size
            engine = VLLMInferenceEngine(**engine_kwargs)
            run_vllm_inference(pending, engine, batch_size=batch_size, jsonl_path=jsonl_path)
        elif model_backend == "openai":
            run_openai_inference(
                pending,
                model_path=model_path,
                max_new_tokens=max_tokens,
                jsonl_path=jsonl_path,
            )
        else:
            raise ValueError(f"Unknown model_backend: {model_backend}")

    records = list(load_cached_predictions(jsonl_path).values())
    metrics = compute_metrics(records)
    scores = {
        "inference_csv": inference_csv,
        "output_dir": str(output_path),
        "model_path": model_path,
        "model_backend": model_backend,
        "processor_path": processor_path or model_path,
        "k": k_value,
        **metrics,
    }

    with open(output_path / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(json.dumps(scores, indent=2))
    return scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt-based baseline for GT document presence prediction."
    )
    parser.add_argument("--inference_csv", type=str, default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--Ks", nargs="+", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report_csv_path", type=str, default=None)
    parser.add_argument("--model_backend", type=str, choices=["vllm", "openai"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--img_basedir", type=str, default=".")
    parser.add_argument("--take_n", type=int, default=-1)
    parser.add_argument("--force_override", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.experiments:
        if not args.Ks:
            raise ValueError("--Ks is required when --experiments is provided")
        if len(args.experiments) != len(args.Ks):
            raise ValueError("Number of --experiments must match number of --Ks")

        slug = model_slug(args.model_path)
        report_rows = []
        for csv_path, k in zip(args.experiments, args.Ks):
            exp_name = experiment_basename(csv_path)
            if args.output_dir:
                out_dir = args.output_dir
            else:
                out_dir = str(Path(args.output_root) / slug / exp_name / f"K={k}")

            scores = run_single_experiment(
                inference_csv=csv_path,
                output_dir=out_dir,
                model_backend=args.model_backend,
                model_path=args.model_path,
                processor_path=args.processor_path,
                batch_size=args.batch_size,
                max_model_len=args.max_model_len,
                max_tokens=args.max_tokens,
                tensor_parallel_size=args.tensor_parallel_size,
                img_basedir=args.img_basedir,
                take_n=args.take_n,
                force_override=args.force_override,
                k_value=k,
            )
            report_rows.append(
                {
                    "Experiment": exp_name,
                    "K": k,
                    "Accuracy (%)": scores["accuracy"] * 100,
                    "Precision (%)": scores["precision"] * 100,
                    "Recall (%)": scores["recall"] * 100,
                    "F1": scores["f1"],
                    "Unparsed Count": scores["unparsed_count"],
                    "Total Samples": scores["total_samples"],
                }
            )

        if report_rows:
            report_path = args.report_csv_path or str(
                Path(args.output_root) / slug / "report.csv"
            )
            report_dir = Path(report_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(report_rows).to_csv(report_path, index=False)
            print(f"Report saved to: {report_path}")
        return

    if not args.inference_csv:
        raise ValueError("Either --inference_csv or --experiments must be provided")
    if not args.output_dir:
        slug = model_slug(args.model_path)
        exp_name = experiment_basename(args.inference_csv)
        args.output_dir = str(Path(args.output_root) / slug / exp_name)

    run_single_experiment(
        inference_csv=args.inference_csv,
        output_dir=args.output_dir,
        model_backend=args.model_backend,
        model_path=args.model_path,
        processor_path=args.processor_path,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        img_basedir=args.img_basedir,
        take_n=args.take_n,
        force_override=args.force_override,
        k_value=None,
    )


if __name__ == "__main__":
    main()
