#!/usr/bin/env python3
"""Evaluate EVQA prediction JSONL files with the existing BEM scorer."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import multiprocessing
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal envs.
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "src" / "evaluation"
DEFAULT_BEM_MODEL_PATH = Path(__file__).resolve().parent / "models" / "bem"
DEFAULT_LOCAL_VOCAB_PATH = Path(__file__).resolve().parent / "models" / "vocab.txt"
VALID_QUESTION_TYPES = {"templated", "automatic", "multi_answer", "2_hop"}

_SCORING_FUNCTION = None


def _looks_remote_path(value: str) -> bool:
    return value.startswith(("gs://", "http://", "https://"))


def _decode_if_serialized(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, str) and parsed.strip().startswith(("[", "{", "(")):
            return _decode_if_serialized(parsed)
        return parsed
    return value


def _as_list(value: Any) -> list[Any]:
    value = _decode_if_serialized(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_text(value: Any) -> str:
    value = _decode_if_serialized(value)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return _as_text(value[0]) if value else ""
    return str(value)


def _to_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


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
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} of {path} is not a JSON object.")
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def _select_prediction(row: dict[str, Any], prediction_field: str) -> tuple[str, str]:
    candidate_fields = (
        ("prediction", False),
        ("generated_answer", False),
        ("response", True),
    )
    if prediction_field != "auto":
        candidate_fields = ((prediction_field, prediction_field == "response"),)

    for field, split_answer in candidate_fields:
        if field not in row:
            continue
        prediction = _as_text(row.get(field)).strip()
        if split_answer and "[ANSWER]" in prediction:
            prediction = prediction.split("[ANSWER]")[-1].strip()
        if prediction:
            return prediction, field
    return "", prediction_field


def _normalize_answers(row: dict[str, Any]) -> list[str]:
    raw_answers = row.get("answers")
    if raw_answers is None:
        raw_answers = row.get("gold_answer", row.get("answer"))

    answers = []
    for answer in _as_list(raw_answers):
        text = _as_text(answer).replace("\n", "").replace("' '", "', '").strip()
        if text:
            answers.append(text)
    return answers


def _normalize_question_type(value: Any) -> str:
    question_type = _as_text(value).strip()
    if question_type in VALID_QUESTION_TYPES:
        return question_type
    return "automatic"


def prepare_eval_rows(
    rows: list[dict[str, Any]],
    prediction_field: str,
    multiple_prediction_mode: str,
) -> tuple[list[dict[str, Any]], str]:
    has_multiple_predictions = any(_as_list(row.get("all_generated_answers")) for row in rows)
    use_all_predictions = multiple_prediction_mode == "all" or (
        multiple_prediction_mode == "auto" and has_multiple_predictions
    )

    prepared: list[dict[str, Any]] = []
    selected_field = prediction_field
    for row_idx, row in enumerate(rows):
        base = dict(row)
        base["original_question_id"] = row.get("question_id", row_idx)
        base["question"] = _as_text(row.get("question")).strip()
        base["answers"] = _normalize_answers(row)
        base["question_type"] = _normalize_question_type(row.get("question_type"))

        predictions: list[tuple[str, str, int]]
        if use_all_predictions:
            all_predictions = [_as_text(x).strip() for x in _as_list(row.get("all_generated_answers"))]
            all_predictions = [x for x in all_predictions if x]
            if all_predictions:
                predictions = [(prediction, "all_generated_answers", i) for i, prediction in enumerate(all_predictions)]
            else:
                prediction, field = _select_prediction(row, prediction_field)
                predictions = [(prediction, field, 0)]
        else:
            prediction, field = _select_prediction(row, prediction_field)
            predictions = [(prediction, field, 0)]

        for prediction, field, sequence_index in predictions:
            eval_row = dict(base)
            eval_row["prediction"] = prediction
            eval_row["prediction_field"] = field
            eval_row["sequence_index"] = sequence_index
            prepared.append(eval_row)
            if selected_field == "auto" and field != "auto":
                selected_field = field

    return prepared, selected_field


def validate_bem_inputs(bem_model_path: str, vocab_path: str | None) -> None:
    if not _looks_remote_path(bem_model_path) and not Path(bem_model_path).exists():
        raise FileNotFoundError(
            "BEM model path does not exist: "
            f"{bem_model_path}. Put the BEM SavedModel under src/eval/models/bem "
            "or pass --bem_model_path."
        )

    if vocab_path and not _looks_remote_path(vocab_path) and not Path(vocab_path).exists():
        raise FileNotFoundError(f"BEM vocab path does not exist: {vocab_path}")

    missing = [
        module_name
        for module_name in ("tensorflow", "tensorflow_hub", "tensorflow_text", "scipy", "numpy")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing EVQA/BEM dependencies in this Python environment: "
            + ", ".join(missing)
            + ". Create/activate the eval venv and install the BEM stack there."
        )


def _configure_tensorflow_device(tf: Any, device: str) -> None:
    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "TensorFlow was asked to use GPU, but no GPU devices are visible. "
            "Check CUDA_VISIBLE_DEVICES and tf.config.list_physical_devices('GPU')."
        )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Memory growth can only be set before TF initializes the device.
            pass


def init_worker(bem_model_path: str, vocab_path: str | None, device: str) -> None:
    global _SCORING_FUNCTION

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if str(EVALUATION_DIR) not in sys.path:
        sys.path.insert(0, str(EVALUATION_DIR))

    try:
        import tensorflow as tf

        _configure_tensorflow_device(tf, device)

        import evaluation_utils
    except Exception as exc:
        raise RuntimeError(
            "Could not import the EVQA BEM evaluator. Check that TensorFlow, "
            "tensorflow_hub, tensorflow_text, scipy, and numpy are installed "
            "in the active eval venv."
        ) from exc

    resolved_vocab_path = vocab_path or getattr(evaluation_utils, "_VOCAB_PATH")
    _SCORING_FUNCTION = evaluation_utils.initialize_encyclopedic_vqa_evaluation_function(
        vocab_path=resolved_vocab_path,
        model_path=bem_model_path,
    )


def score_prepared_row(row: dict[str, Any]) -> dict[str, Any]:
    if _SCORING_FUNCTION is None:
        raise RuntimeError("BEM scoring function was not initialized.")

    status = _as_text(row.get("status")).strip().lower()
    if status and status != "ok":
        return {"score": 0.0, "eval_error": f"skipped_status={status}"}

    question = _as_text(row.get("question")).strip()
    prediction = _as_text(row.get("prediction")).strip()
    answers = _normalize_answers(row)
    question_type = _normalize_question_type(row.get("question_type"))

    if not question:
        return {"score": 0.0, "eval_error": "missing_question"}
    if not prediction:
        return {"score": 0.0, "eval_error": "missing_prediction"}
    if not answers:
        return {"score": 0.0, "eval_error": "missing_answers"}

    scores = []
    try:
        for reference in answers:
            example = {
                "question": question,
                "reference": reference,
                "candidate": prediction,
                "question_type": question_type,
            }
            scores.append(float(_SCORING_FUNCTION(example)))
    except Exception as exc:
        return {"score": 0.0, "eval_error": str(exc)}

    return {"score": max(scores) if scores else 0.0, "eval_error": ""}


def score_rows(
    rows: list[dict[str, Any]],
    bem_model_path: str,
    vocab_path: str | None,
    num_processes: int,
    device: str,
) -> list[dict[str, Any]]:
    if device == "gpu" and num_processes != 1:
        raise ValueError(
            "GPU EVQA/BEM evaluation should run in a single TensorFlow process. "
            "Set --num_processes 1."
        )

    if num_processes <= 1:
        init_worker(bem_model_path, vocab_path, device)
        return [
            score_prepared_row(row)
            for row in tqdm(rows, total=len(rows), desc="Computing EVQA BEM scores")
        ]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(bem_model_path, vocab_path, device),
    ) as pool:
        return list(
            tqdm(
                pool.imap(score_prepared_row, rows, chunksize=1),
                total=len(rows),
                desc="Computing EVQA BEM scores",
            )
        )


def _get_gt_passage_id(row: dict[str, Any]) -> str | None:
    for key in ("gt_passage_id", "gold_passage_id", "positive_passage_id"):
        if row.get(key):
            return _as_text(row[key])

    pos_item_ids = _as_list(row.get("pos_item_ids"))
    if pos_item_ids:
        return _as_text(pos_item_ids[0])

    gt_passage_in_zidx = _to_int(row.get("gt_passage_in_zidx"))
    passages = _as_list(row.get("passages"))
    if gt_passage_in_zidx < 0 or gt_passage_in_zidx >= len(passages):
        return None

    passage = _decode_if_serialized(passages[gt_passage_in_zidx])
    if isinstance(passage, dict) and passage.get("passage_id"):
        return _as_text(passage["passage_id"])
    return None


def _first_nonempty_text_list(row: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        values = [_as_text(value).strip() for value in _as_list(row.get(key))]
        values = [value for value in values if value]
        if values:
            return values
    return []


def _top_passage_id(
    row: dict[str, Any],
    explicit_keys: tuple[str, ...],
    idx_keys: tuple[str, ...],
) -> str | None:
    for key in explicit_keys:
        passage_id = _as_text(row.get(key)).strip()
        if passage_id:
            return passage_id

    retrieved_passage_ids = _first_nonempty_text_list(row, ("retrieved_passage_ids",))
    for key in idx_keys:
        idx = _to_int(row.get(key))
        if 0 <= idx < len(retrieved_passage_ids):
            return retrieved_passage_ids[idx]
    return None


def _compute_hit_rate(
    rows: list[dict[str, Any]],
    *,
    hit_keys: tuple[str, ...],
    explicit_top_keys: tuple[str, ...],
    idx_keys: tuple[str, ...],
) -> float | None:
    hits = []
    for row in rows:
        gt_passage_id = _get_gt_passage_id(row)
        if not gt_passage_id:
            continue

        top_passage_id = _top_passage_id(row, explicit_top_keys, idx_keys)
        if top_passage_id is not None:
            hits.append(1.0 if top_passage_id == gt_passage_id else 0.0)
            continue

        for key in hit_keys:
            if key in row and row.get(key) not in (None, ""):
                hits.append(1.0 if _to_bool(row.get(key)) else 0.0)
                break
    return _mean(hits) if hits else None


def _compute_retrieval_hit_rate(rows: list[dict[str, Any]]) -> float | None:
    hits = []
    for row in rows:
        gt_passage_id = _get_gt_passage_id(row)
        if not gt_passage_id:
            continue
        retrieved_passage_ids = _first_nonempty_text_list(row, ("retrieved_passage_ids",))
        if retrieved_passage_ids:
            hits.append(1.0 if gt_passage_id in retrieved_passage_ids else 0.0)
        elif "gt_passage_in_zidx" in row:
            hits.append(1.0 if _to_int(row.get("gt_passage_in_zidx")) != -1 else 0.0)
    return _mean(hits) if hits else None


def _compute_recall_at_k(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[int, float]:
    sorted_lists = [_first_nonempty_text_list(row, keys) for row in rows]
    non_empty_lists = [ids for ids in sorted_lists if ids]
    if not non_empty_lists:
        return {}

    max_k = max(len(ids) for ids in non_empty_lists)
    recall_at_k = {}
    for k in range(1, max_k + 1):
        hits = []
        for row, passage_ids in zip(rows, sorted_lists):
            if not passage_ids:
                continue
            gt_passage_id = _get_gt_passage_id(row)
            if not gt_passage_id:
                continue
            hits.append(1.0 if gt_passage_id in passage_ids[:k] else 0.0)
        if hits:
            recall_at_k[k] = _mean(hits)
    return recall_at_k


def compute_prior_recall_at_k(rows: list[dict[str, Any]]) -> dict[int, float]:
    return _compute_recall_at_k(
        rows, ("berag_prior_sorted_passage_ids", "prior_sorted_passage_ids")
    )


def compute_posterior_recall_at_k(rows: list[dict[str, Any]]) -> dict[int, float]:
    return _compute_recall_at_k(rows, ("berag_posterior_sorted_passage_ids",))


def add_monitoring_metrics(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    retrieval_topk: int | None,
) -> None:
    unavailable = []

    prior_hit_rate = _compute_hit_rate(
        rows,
        hit_keys=("prior_hit", "prior_passage_is_gt"),
        explicit_top_keys=("berag_prior_top_passage_id",),
        idx_keys=("berag_prior_max_idx", "prior_max_idx"),
    )
    if prior_hit_rate is not None:
        report["prior_hit_rate"] = prior_hit_rate
        report["prior_passage_hit_rate"] = prior_hit_rate
    else:
        unavailable.extend(["prior_hit_rate", "prior_passage_hit_rate"])

    posterior_hit_rate = _compute_hit_rate(
        rows,
        hit_keys=("posterior_hit", "dominant_passage_is_gt"),
        explicit_top_keys=("berag_posterior_top_passage_id",),
        idx_keys=("berag_posterior_max_idx", "z_dominant_idx"),
    )
    if posterior_hit_rate is not None:
        report["posterior_hit_rate"] = posterior_hit_rate
        report["posterior_passage_hit_rate"] = posterior_hit_rate
    else:
        unavailable.extend(["posterior_hit_rate", "posterior_passage_hit_rate"])

    retrieval_hit_rate = _compute_retrieval_hit_rate(rows)
    if retrieval_hit_rate is not None:
        report["retrieval_hit_rate"] = retrieval_hit_rate
    else:
        unavailable.append("retrieval_hit_rate")

    effective_topk = retrieval_topk
    if effective_topk is None:
        row_topks = [_to_int(row.get("retrieval_topk"), default=-1) for row in rows]
        row_topks = [value for value in row_topks if value >= 0]
        effective_topk = row_topks[0] if row_topks and len(set(row_topks)) == 1 else None

    if effective_topk is not None and all("gt_passage_in_zidx" in row and "z_dominant_idx" in row for row in rows):
        rows_without_gt = [row for row in rows if _to_int(row.get("gt_passage_in_zidx")) == -1]
        report["correct_ignore_rate"] = _mean(
            1.0 if _to_int(row.get("z_dominant_idx")) == effective_topk else 0.0
            for row in rows_without_gt
        )
    else:
        unavailable.append("correct_ignore_rate")

    prior_recall_at_k = compute_prior_recall_at_k(rows)
    if prior_recall_at_k:
        report["prior_recall_at_k"] = prior_recall_at_k
    else:
        unavailable.append("prior_recall_at_k")

    posterior_recall_at_k = compute_posterior_recall_at_k(rows)
    if posterior_recall_at_k:
        report["posterior_recall_at_k"] = posterior_recall_at_k
    else:
        unavailable.append("posterior_recall_at_k")

    if unavailable:
        report["unavailable_metrics"] = sorted(set(unavailable))


def aggregate_multiple_prediction_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        question_id = _as_text(row.get("original_question_id", row.get("question_id")))
        grouped.setdefault(question_id, []).append(float(row.get("score", 0.0)))

    aggregated = []
    for question_id, scores in grouped.items():
        aggregated.append(
            {
                "question_id": question_id,
                "median_score": float(statistics.median(scores)),
                "best_score": float(max(scores)),
                "mean_score": _mean(scores),
                "all_scores": scores,
            }
        )
    return aggregated


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "question_id",
        "original_question_id",
        "sequence_index",
        "question",
        "question_type",
        "gold_answer",
        "answers",
        "prediction",
        "prediction_field",
        "generated_answer",
        "response",
        "status",
        "score",
        "eval_error",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _compact_gold_answer(row: dict[str, Any]) -> str:
    gold_answer = _as_text(row.get("gold_answer")).strip()
    if gold_answer:
        return gold_answer
    answers = _normalize_answers(row)
    return answers[0] if answers else ""


def _compact_image_path(row: dict[str, Any]) -> str:
    for key in ("image_path", "img_path", "image"):
        image_path = _as_text(row.get(key)).strip()
        if image_path:
            return image_path
    return ""


def _compact_generated_response(row: dict[str, Any]) -> str:
    for key in ("response", "generated_answer", "prediction"):
        response = _as_text(row.get(key)).strip()
        if response:
            return response
    return ""


def build_evaluated_instances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_rows = []
    for row in rows:
        compact_row = {
            "question_id": row.get("question_id", ""),
            "question": _as_text(row.get("question")).strip(),
            "gold_answer": _compact_gold_answer(row),
            "image_path": _compact_image_path(row),
            "generated_response": _compact_generated_response(row),
            "evqa_score": float(row.get("score", 0.0)),
        }
        for key in (
            "gt_passage_id",
            "gt_passage_in_zidx",
            "retrieved_passage_ids",
            "berag_log_prior",
            "berag_log_posterior",
            "berag_prior_max_idx",
            "berag_posterior_max_idx",
            "berag_prior_sorted_indices",
            "berag_posterior_sorted_indices",
            "berag_prior_sorted_passage_ids",
            "berag_posterior_sorted_passage_ids",
            "berag_prior_top_passage_id",
            "berag_posterior_top_passage_id",
            "prior_hit",
            "posterior_hit",
        ):
            if key in row:
                compact_row[key] = row.get(key)
        compact_rows.append(compact_row)
    return compact_rows


def build_report(
    eval_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
    selected_prediction_field: str,
    retrieval_topk: int | None,
    multiple_predictions_scored: bool,
) -> dict[str, Any]:
    scores = [float(row.get("score", 0.0)) for row in eval_rows]
    report: dict[str, Any] = {
        "avg_score": _mean(scores),
        "num_examples": int(len(original_rows)),
        "num_eval_rows": int(len(eval_rows)),
        "num_failed": int(sum(1 for row in eval_rows if row.get("eval_error"))),
        "prediction_field": selected_prediction_field,
    }

    if multiple_predictions_scored:
        aggregated = aggregate_multiple_prediction_scores(eval_rows)
        report["avg_median_score"] = _mean(row["median_score"] for row in aggregated)
        report["avg_best_score"] = _mean(row["best_score"] for row in aggregated)
        report["avg_mean_score"] = _mean(row["mean_score"] for row in aggregated)

    add_monitoring_metrics(report, original_rows, retrieval_topk)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EVQA prediction JSONL files with BEM and write old-style score reports."
    )
    parser.add_argument("--predictions_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--dataset_name", type=str, default="EVQA", choices=["EVQA"])
    parser.add_argument("--prediction_field", type=str, default="auto")
    parser.add_argument(
        "--multiple_prediction_mode",
        type=str,
        default="auto",
        choices=["auto", "first", "all"],
        help="Use all_generated_answers when present, only the first prediction, or auto-detect.",
    )
    parser.add_argument("--bem_model_path", type=str, default=str(DEFAULT_BEM_MODEL_PATH))
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Optional BERT vocab path. Defaults to src/eval/models/vocab.txt if present, else EVQA's gs:// default.",
    )
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieval_topk", type=int, default=None)
    parser.add_argument("--marked_filename", type=str, default="marked_inference_results.csv")
    parser.add_argument("--instances_filename", type=str, default="evaluated_instances.csv")
    parser.add_argument("--scores_filename", type=str, default="scores.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_file = args.predictions_file.resolve()
    output_dir = (args.output_dir or predictions_file.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = args.vocab_path
    if vocab_path is None and DEFAULT_LOCAL_VOCAB_PATH.exists():
        vocab_path = str(DEFAULT_LOCAL_VOCAB_PATH)

    rows = read_jsonl(predictions_file, limit=args.limit)
    eval_rows, selected_prediction_field = prepare_eval_rows(
        rows,
        prediction_field=args.prediction_field,
        multiple_prediction_mode=args.multiple_prediction_mode,
    )
    multiple_predictions_scored = len(eval_rows) > len(rows)

    validate_bem_inputs(args.bem_model_path, vocab_path)
    score_results = score_rows(
        eval_rows,
        bem_model_path=args.bem_model_path,
        vocab_path=vocab_path,
        num_processes=args.num_processes,
        device=args.device,
    )

    for row, result in zip(eval_rows, score_results):
        row.update(result)

    report = build_report(
        eval_rows=eval_rows,
        original_rows=rows,
        selected_prediction_field=selected_prediction_field,
        retrieval_topk=args.retrieval_topk,
        multiple_predictions_scored=multiple_predictions_scored,
    )
    report["device"] = args.device
    report["num_processes"] = args.num_processes

    marked_path = output_dir / args.marked_filename
    instances_path = output_dir / args.instances_filename
    scores_path = output_dir / args.scores_filename
    write_csv(marked_path, eval_rows)
    write_csv(instances_path, build_evaluated_instances(eval_rows))

    if multiple_predictions_scored:
        aggregated_rows = aggregate_multiple_prediction_scores(eval_rows)
        aggregated_path = output_dir / "aggregated_scores.csv"
        write_csv(aggregated_path, aggregated_rows)
        print(f"Aggregated scores saved to {aggregated_path}")

    with scores_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("--------------------------------")
    print("Evaluation results:")
    print(json.dumps(report, indent=2))
    print("--------------------------------")
    print(f"Marked predictions saved to {marked_path}")
    print(f"Evaluated instances saved to {instances_path}")
    print(f"Scores saved to {scores_path}")


if __name__ == "__main__":
    main()
