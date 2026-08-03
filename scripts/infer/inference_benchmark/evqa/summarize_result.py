#!/usr/bin/env python3
"""Create one compact table row from an E-VQA benchmark experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    args = parser.parse_args()

    metrics = load_json(args.experiment_dir / "run_metrics.json")
    scores = load_json(args.experiment_dir / "score.json")
    row = {
        "method": args.method,
        "num_examples": metrics.get("num_ok"),
        "batch_size": metrics.get("batch_size"),
        "retrieval_topk": metrics.get("retrieval_topk"),
        "top_p": args.top_p,
        "max_model_len": args.max_model_len,
        "mean_input_tokens": metrics.get("mean_prompt_tokens"),
        "p90_ttft_s": metrics.get("p90_ttft_s"),
        "p90_tpot_s": metrics.get("p90_tpot_s"),
        "wall_time_s": metrics.get("vllm_run_wall_time_s"),
        "requests_per_second": metrics.get("requests_per_second"),
        "score": scores.get("avg_score"),
    }

    output_path = args.experiment_dir / "benchmark_row.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
        handle.write("\n")

    columns = (
        "method",
        "mean_input_tokens",
        "p90_ttft_s",
        "p90_tpot_s",
        "wall_time_s",
        "score",
    )
    print("\t".join(columns))
    print("\t".join(str(row[column]) for column in columns))
    print(f"Benchmark row written to {output_path}")


if __name__ == "__main__":
    main()
