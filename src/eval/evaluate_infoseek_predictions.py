#!/usr/bin/env python3
"""Evaluate BERAG InfoSeek JSONL predictions with the official toolkit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOLKIT_DIR = PROJECT_ROOT / "third_party" / "infoseek_eval"
DEFAULT_REFERENCE_FILE = DEFAULT_TOOLKIT_DIR / "infoseek" / "infoseek_val.jsonl"
DEFAULT_REFERENCE_QTYPE_FILE = (
    DEFAULT_TOOLKIT_DIR / "infoseek" / "infoseek_val_qtype.jsonl"
)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} of {path} is not a JSON object.")
            rows.append(row)
    return rows


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return as_text(value[0]) if value else ""
    return str(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return as_list(decoded)
    return [value]


def first_text(value: Any) -> str:
    values = as_list(value)
    return as_text(values[0]).strip() if values else ""


def first_nonempty_text_list(
    row: dict[str, Any], keys: tuple[str, ...]
) -> list[str]:
    for key in keys:
        values = [as_text(value).strip() for value in as_list(row.get(key))]
        values = [value for value in values if value]
        if values:
            return values
    return []


def get_gt_passage_id(row: dict[str, Any]) -> str:
    for key in ("gt_passage_id", "gold_passage_id", "positive_passage_id"):
        passage_id = as_text(row.get(key)).strip()
        if passage_id:
            return passage_id
    return first_text(row.get("pos_item_ids"))


def compute_recall_at_k(
    rows: list[dict[str, Any]], ranked_id_keys: tuple[str, ...]
) -> tuple[dict[str, float], int]:
    examples: list[tuple[str, list[str]]] = []
    for row in rows:
        gt_passage_id = get_gt_passage_id(row)
        ranked_passage_ids = first_nonempty_text_list(row, ranked_id_keys)
        if gt_passage_id and ranked_passage_ids:
            examples.append((gt_passage_id, ranked_passage_ids))

    if not examples:
        return {}, 0

    max_k = max(len(ranked_passage_ids) for _, ranked_passage_ids in examples)
    recall_at_k = {}
    for k in range(1, max_k + 1):
        hits = sum(
            gt_passage_id in ranked_passage_ids[:k]
            for gt_passage_id, ranked_passage_ids in examples
        )
        recall_at_k[str(k)] = hits / len(examples)
    return recall_at_k, len(examples)


def add_recall_metrics(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    retrieval_recall, retrieval_count = compute_recall_at_k(
        rows, ("retrieved_passage_ids",)
    )
    prior_recall, prior_count = compute_recall_at_k(
        rows,
        ("berag_prior_sorted_passage_ids", "prior_sorted_passage_ids"),
    )
    posterior_recall, posterior_count = compute_recall_at_k(
        rows, ("berag_posterior_sorted_passage_ids",)
    )

    report["retrieval_recall_at_k"] = retrieval_recall
    report["recall_at_k"] = retrieval_recall
    report["retrieval_recall_num_examples"] = retrieval_count
    if prior_recall:
        report["prior_recall_at_k"] = prior_recall
        report["prior_recall_num_examples"] = prior_count
    if posterior_recall:
        report["posterior_recall_at_k"] = posterior_recall
        report["posterior_recall_num_examples"] = posterior_count


def select_prediction(
    row: dict[str, Any], prediction_field: str
) -> tuple[str, str]:
    fields = (
        ("prediction", False),
        ("generated_answer", False),
        ("response", True),
    )
    if prediction_field != "auto":
        fields = ((prediction_field, prediction_field == "response"),)

    for field, split_answer in fields:
        if field not in row:
            continue
        prediction = as_text(row.get(field)).strip()
        if split_answer and "[ANSWER]" in prediction:
            prediction = prediction.split("[ANSWER]")[-1].strip()
        if prediction or prediction_field != "auto":
            return prediction, field
    return "", prediction_field


def prepare_predictions(
    rows: list[dict[str, Any]], prediction_field: str
) -> tuple[list[dict[str, str]], list[dict[str, Any]], str]:
    official_predictions: list[dict[str, str]] = []
    marked_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    selected_fields: set[str] = set()

    for row_idx, row in enumerate(rows):
        data_id = as_text(
            row.get("data_id")
            or row.get("question_id")
            or row.get("example_id")
        ).strip()
        if not data_id:
            raise ValueError(f"Prediction row {row_idx} has no data/question/example ID.")
        if data_id in seen_ids:
            raise ValueError(f"Duplicate prediction data_id: {data_id}")
        seen_ids.add(data_id)

        prediction, selected_field = select_prediction(row, prediction_field)
        selected_fields.add(selected_field)
        official_predictions.append({"data_id": data_id, "prediction": prediction})

        marked = dict(row)
        marked.update(
            {
                "data_id": data_id,
                "prediction": prediction,
                "prediction_field": selected_field,
            }
        )
        marked_rows.append(marked)

    selected = next(iter(selected_fields)) if len(selected_fields) == 1 else "mixed"
    return official_predictions, marked_rows, selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "data_id",
        "question_id",
        "question",
        "question_type",
        "data_split",
        "gold_answer",
        "answers",
        "reference_answers",
        "prediction",
        "prediction_field",
        "generated_answer",
        "response",
        "status",
        "infoseek_score",
        "eval_error",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def score_example(toolkit: Any, prediction: str, reference: dict[str, Any]) -> int:
    question_type = as_text(reference.get("question_type")).lower()
    answers = reference.get("answer_eval") or []
    if question_type == "numerical":
        pred_range = toolkit.process_numerical_answer(prediction)
        answer_range = [float(answer) for answer in answers]
        return int(toolkit.metric_numerical_range(pred_range, answer_range))
    return int(
        toolkit.metric_max_over_ground_truths(
            toolkit.exact_match_score,
            prediction,
            answers,
        )
    )


def build_instances(marked_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    instances = []
    for row in marked_rows:
        instance = {
            "data_id": row.get("data_id", ""),
            "question": row.get("question", ""),
            "question_type": row.get("question_type", ""),
            "data_split": row.get("data_split", ""),
            "reference_answers": row.get("reference_answers", []),
            "generated_response": row.get("prediction", ""),
            "infoseek_score": row.get("infoseek_score", 0),
            "eval_error": row.get("eval_error", ""),
        }
        for key in (
            "gt_passage_id",
            "gt_passage_in_zidx",
            "retrieved_passage_ids",
            "berag_prior_top_passage_id",
            "berag_posterior_top_passage_id",
            "prior_hit",
            "posterior_hit",
        ):
            if key in row:
                instance[key] = row[key]
        instances.append(instance)
    return instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate InfoSeek prediction JSONL with the official evaluator."
    )
    parser.add_argument("--predictions_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--prediction_field", default="auto")
    parser.add_argument("--toolkit_dir", type=Path, default=DEFAULT_TOOLKIT_DIR)
    parser.add_argument("--reference_file", type=Path, default=DEFAULT_REFERENCE_FILE)
    parser.add_argument(
        "--reference_qtype_file",
        type=Path,
        default=DEFAULT_REFERENCE_QTYPE_FILE,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow_unknown_ids", action="store_true")
    parser.add_argument("--scores_filename", default="score.json")
    parser.add_argument("--marked_filename", default="marked_inference_results.csv")
    parser.add_argument("--instances_filename", default="evaluated_instances.csv")
    parser.add_argument(
        "--official_predictions_filename",
        default="infoseek_predictions.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_file = args.predictions_file.resolve()
    output_dir = (args.output_dir or predictions_file.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    toolkit_dir = args.toolkit_dir.resolve()
    reference_file = args.reference_file.resolve()
    reference_qtype_file = args.reference_qtype_file.resolve()
    for path in (toolkit_dir / "infoseek_eval.py", reference_file, reference_qtype_file):
        if not path.exists():
            raise FileNotFoundError(f"Required InfoSeek evaluation file not found: {path}")

    sys.path.insert(0, str(toolkit_dir))
    import infoseek_eval as toolkit

    rows = read_jsonl(predictions_file, limit=args.limit)
    official_predictions, marked_rows, selected_field = prepare_predictions(
        rows, args.prediction_field
    )

    references = toolkit.load_jsonl(str(reference_file))
    reference_qtypes = toolkit.load_jsonl(str(reference_qtype_file))
    qid_to_example = toolkit.prepare_qid2example(references, reference_qtypes)

    matched_predictions = [
        prediction
        for prediction in official_predictions
        if prediction["data_id"] in qid_to_example
    ]
    unknown_ids = [
        prediction["data_id"]
        for prediction in official_predictions
        if prediction["data_id"] not in qid_to_example
    ]
    if unknown_ids and not args.allow_unknown_ids:
        examples = ", ".join(unknown_ids[:5])
        raise ValueError(
            f"{len(unknown_ids)} prediction IDs are absent from the InfoSeek references; "
            f"examples: {examples}. Pass --allow_unknown_ids to ignore them."
        )

    unseen_question = []
    unseen_entity = []
    for prediction in matched_predictions:
        reference = qid_to_example[prediction["data_id"]]
        if as_text(reference.get("data_split")).endswith("unseen_question"):
            unseen_question.append(prediction)
        else:
            unseen_entity.append(prediction)

    official_result = toolkit.evaluate_infoseek_full(
        [unseen_question, unseen_entity],
        [qid_to_example, qid_to_example],
    )

    matched_scores = []
    for row in marked_rows:
        reference = qid_to_example.get(row["data_id"])
        if reference is None:
            row.update({"infoseek_score": 0, "eval_error": "unknown_reference"})
            continue
        try:
            score = score_example(toolkit, row["prediction"], reference)
            row.update({"infoseek_score": score, "eval_error": ""})
            matched_scores.append(score)
        except Exception as exc:
            row.update({"infoseek_score": 0, "eval_error": str(exc)})
            matched_scores.append(0)
        row["question_type"] = reference.get("question_type", "")
        row["data_split"] = reference.get("data_split", "")
        row["reference_answers"] = reference.get("answer_eval", [])

    report = dict(official_result)
    report.update(
        {
            "score": official_result.get("final_score", 0.0),
            "mean_example_score": (
                100.0 * sum(matched_scores) / len(matched_scores)
                if matched_scores
                else 0.0
            ),
            "num_examples": len(rows),
            "num_matched_references": len(matched_predictions),
            "num_unknown_reference_ids": len(unknown_ids),
            "num_unseen_question": len(unseen_question),
            "num_unseen_entity": len(unseen_entity),
            "num_failed_generation": sum(
                1
                for row in rows
                if as_text(row.get("status")).lower() not in ("", "ok")
            ),
            "prediction_field": selected_field,
        }
    )
    add_recall_metrics(report, rows)

    official_predictions_path = output_dir / args.official_predictions_filename
    marked_path = output_dir / args.marked_filename
    instances_path = output_dir / args.instances_filename
    scores_path = output_dir / args.scores_filename
    write_jsonl(official_predictions_path, official_predictions)
    write_csv(marked_path, marked_rows)
    write_csv(instances_path, build_instances(marked_rows))
    with scores_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("--------------------------------")
    print("InfoSeek evaluation results:")
    print(json.dumps(report, indent=2))
    print("--------------------------------")
    print(f"Official predictions saved to {official_predictions_path}")
    print(f"Marked predictions saved to {marked_path}")
    print(f"Evaluated instances saved to {instances_path}")
    print(f"Scores saved to {scores_path}")


if __name__ == "__main__":
    main()
