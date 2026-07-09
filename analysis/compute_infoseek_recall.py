#!/usr/bin/env python3
import argparse
import ast
import json
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None


def _to_list(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return parsed
            except (SyntaxError, ValueError):
                return []
    return []


def _normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_retrieved_ids(retrieved_obj: Any) -> List[Any]:
    items = _to_list(retrieved_obj)
    if not items:
        return []
    if isinstance(items[0], dict):
        return [_normalize_id(item.get("passage_id")) for item in items if isinstance(item, dict) and "passage_id" in item]
    return [_normalize_id(item) for item in items]


def compute_recall(df: pd.DataFrame, ks: Iterable[int], retrieval_field: str) -> Dict[str, float]:
    scores = {f"Recall@{k}": [] for k in ks}
    empty_pos = 0
    empty_ret = 0

    for _, row in df.iterrows():
        pos_ids = set(_normalize_id(x) for x in _to_list(row.get("pos_item_ids", [])))
        ret_ids = _extract_retrieved_ids(row.get(retrieval_field, []))
        if not pos_ids:
            empty_pos += 1
        if not ret_ids:
            empty_ret += 1

        for k in ks:
            hit = 1.0 if any(doc_id in pos_ids for doc_id in ret_ids[:k]) else 0.0
            scores[f"Recall@{k}"].append(hit)

    results = {metric: float(np.mean(values)) if values else 0.0 for metric, values in scores.items()}
    results["_rows"] = int(len(df))
    results["_rows_with_empty_pos_item_ids"] = int(empty_pos)
    results["_rows_with_empty_retrieval"] = int(empty_ret)
    return results


def load_df(input_path: str) -> pd.DataFrame:
    # 1) HuggingFace dataset directory saved via save_to_disk.
    if load_from_disk is not None:
        try:
            ds = load_from_disk(input_path)
            return ds.to_pandas()
        except Exception:
            pass

    # 2) Flat files.
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    if input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    if input_path.endswith(".jsonl"):
        return pd.read_json(input_path, lines=True)

    raise ValueError(
        f"Unsupported input: {input_path}. "
        "Use a HF dataset directory, or a csv/parquet/jsonl file."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Path to saved retrieval dataset directory or csv/parquet/jsonl file.",
        default="outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval",
    )
    parser.add_argument(
        "--retrieval_field",
        default="retrieved_passage",
        help="Column containing retrieval results (default: retrieved_passage).",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 15, 20, 30, 40, 50, 100, 500],
        help="K values to compute Recall@K for (default: 1 3 5 10 15 20 30 40 50 100 500).",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save metrics JSON.",
    )
    args = parser.parse_args()

    df = load_df(args.input)
    results = compute_recall(df=df, ks=args.ks, retrieval_field=args.retrieval_field)
    print(json.dumps(results, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as file:
            json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
