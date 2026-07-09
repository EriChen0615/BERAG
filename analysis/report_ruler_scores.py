#!/usr/bin/env python3
"""Aggregate per-task RULER scores across context lengths for a result root.

Example:
  python3 analysis/report_ruler_scores.py \
    --root third_party/RULER/scripts/outputs/0326/ruler/BERAG/Qwen2.5-3B-Instruct/b4_s10_mc1_sg4_scoreproposal_chunk/synthetic
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_summary(summary_path: Path) -> dict[str, float]:
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise ValueError(f"Unexpected summary format in {summary_path}")

    tasks = rows[1][1:]
    scores = rows[2][1:]
    return {task: float(score) for task, score in zip(tasks, scores)}


def length_sort_key(length_str: str) -> tuple[int, str]:
    try:
        return (int(length_str), length_str)
    except ValueError:
        return (10**18, length_str)


def collect_scores(root: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    summaries = sorted(root.glob("*/pred/summary.csv"), key=lambda p: length_sort_key(p.parent.parent.name))
    if not summaries:
        raise FileNotFoundError(f"No summary.csv files found under {root}")

    all_tasks: list[str] = []
    table: dict[str, dict[str, float]] = {}

    for summary in summaries:
        length = summary.parent.parent.name
        scores = parse_summary(summary)
        table[length] = scores
        for task in scores:
            if task not in all_tasks:
                all_tasks.append(task)

    return all_tasks, table


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-task RULER scores across context lengths.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing <length>/pred/summary.csv files.")
    parser.add_argument("--format", choices=["csv", "markdown"], default="csv", help="Output table format.")
    parser.add_argument("--output", type=Path, default=None, help="Optional file to write the table to.")
    args = parser.parse_args()

    tasks, table = collect_scores(args.root)
    lengths = sorted(table.keys(), key=length_sort_key)

    rows: list[list[str]] = [["Length"] + tasks]
    for length in lengths:
        row = [length] + [f"{table[length].get(task, float('nan')):.2f}" for task in tasks]
        rows.append(row)

    if args.format == "csv":
        lines = [",".join(row) for row in rows]
    else:
        header = "| " + " | ".join(rows[0]) + " |"
        sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
        body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
        lines = [header, sep] + body

    text = "\n".join(lines)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
