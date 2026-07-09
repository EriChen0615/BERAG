#!/usr/bin/env python3
"""
Report RULER benchmark results in a table format similar to the paper (Table 1).

Reads summary.csv files from a directory structure like:
  {root}/ruler/{model}/synthetic/{seq_len}/pred/summary.csv
or a single model:
  {root}/{model}/synthetic/{seq_len}/pred/summary.csv
  e.g. .../Llama-3.2-3B-Instruct/synthetic/64000/pred/summary.csv

Outputs: Models | Claimed Length | Effective Length | 4K | 8K | 16K | 32K | 64K | 128K | Avg. | wAvg. (inc) | wAvg. (dec)
"""

import argparse
import csv
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Standard context lengths (tokens) and their display names
STANDARD_LENGTHS = [4096, 8192, 16384, 32000, 64000, 128000]
LENGTH_LABELS = ["4K", "8K", "16K", "32K", "64K", "128K"]
SEQ_TO_LABEL = dict(zip(STANDARD_LENGTHS, LENGTH_LABELS))

# Model config: claimed length, effective length (from RULER README / paper)
# Use "-" when unknown. Add entries as needed.
MODEL_CONFIG = {
    "Llama-3.2-3B-Instruct": ("128K", "-"),
    "Llama-3.1-3B-Instruct": ("128K", "-"),
    "Qwen2.5-3B-Instruct": ("128K", "-"),
    "Qwen2.5-7B-Instruct-1M": ("1M", ">128K"),
    "BERAG/Qwen2.5-3B-Instruct": ("128K", "-"),
    "BERAG/Llama-3.2-3B-Instruct": ("128K", "-"),
}


def _format_length(n: int) -> str:
    """Format sequence length as 4K, 8K, etc."""
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)


def parse_summary_csv(
    path: Path,
    score_mode: str = "mean_all",
    single_task_prefix: str = "niah_single_",
) -> Optional[float]:
    """
    Parse summary.csv and return a score based on score_mode.
    Row 0: indices, Row 1: task names, Row 2: Score, Row 3: Nulls.

    score_mode:
      - mean_all: mean score across all tasks (default, previous behavior)
      - worst_single: minimum score among tasks with prefix single_task_prefix
    """
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) < 3:
            return None
        # Row index 1 is task names, row index 2 is scores
        task_row = rows[1]
        score_row = rows[2]
        if task_row[0] != "Tasks" or score_row[0] != "Score":
            return None

        tasks = task_row[1:]
        raw_scores = score_row[1:]
        values = []
        for t, v in zip(tasks, raw_scores):
            try:
                sv = float(v)
            except (ValueError, TypeError):
                continue
            values.append((t, sv))

        if not values:
            return None

        if score_mode == "mean_all":
            score_vals = [v for _, v in values]
            return sum(score_vals) / len(score_vals)

        if score_mode == "worst_single":
            single_vals = [v for t, v in values if t.startswith(single_task_prefix)]
            if not single_vals:
                return None
            return min(single_vals)

        return None
    except (OSError, csv.Error):
        return None


def collect_results(
    root: Path,
    score_mode: str = "mean_all",
    single_task_prefix: str = "niah_single_",
):
    """
    Walk root to find all model/synthetic/{seq_len}/pred/summary.csv.
    Returns: {model_name: {seq_len: score}}
    """
    results = defaultdict(dict)
    orig_root = root.resolve()
    # If the user points directly at a `.../MODEL/synthetic` directory,
    # search from its parent but later filter back to that synthetic dir.
    is_synthetic_root = orig_root.name == "synthetic"
    search_root = orig_root.parent if is_synthetic_root else orig_root
    # For the synthetic-root case, the real model name is the parent directory.
    synthetic_model_name = orig_root.parent.name if is_synthetic_root else None

    for path in search_root.rglob("pred/summary.csv"):
        try:
            # path: .../model/synthetic/seq_len/pred/summary.csv
            if is_synthetic_root and "synthetic" not in path.parts:
                # Skip other directories when user specifically pointed at one synthetic folder.
                continue

            parts = path.relative_to(search_root).parts
            if "synthetic" not in parts:
                continue
            idx = parts.index("synthetic")
            # If `synthetic` is the first component and the user rooted at that
            # specific synthetic directory, use its parent dir name as model.
            if idx == 0 and synthetic_model_name is not None:
                model = synthetic_model_name
            else:
                model = "/".join(parts[:idx])
            seq_str = parts[idx + 1]
            try:
                seq_len = int(seq_str)
            except ValueError:
                continue
            score = parse_summary_csv(
                path,
                score_mode=score_mode,
                single_task_prefix=single_task_prefix,
            )
            if score is not None:
                results[model][seq_len] = round(score, 2)
        except ValueError:
            continue
    return dict(results)


def compute_weighted_avg(
    scores,
    inc: bool,
    ref_lengths=STANDARD_LENGTHS,
) -> Optional[float]:
    """
    Compute weighted average. inc=True: longer contexts weighted more.
    inc=False: shorter contexts weighted more.
    Uses linear weights: inc -> [1,2,3,4,5,6], dec -> [6,5,4,3,2,1].
    """
    available = [(seq, scores[seq]) for seq in ref_lengths if seq in scores]
    if not available:
        return None
    n = len(ref_lengths)
    weights = list(range(1, n + 1)) if inc else list(range(n, 0, -1))
    total = 0.0
    wsum = 0.0
    for i, seq in enumerate(ref_lengths):
        if seq in scores:
            total += scores[seq] * weights[i]
            wsum += weights[i]
    if wsum == 0:
        return None
    return round(total / wsum, 2)


def rank_values(values: list) -> dict:
    """Assign ordinal rankings (1st, 2nd, ...). Higher is better. None excluded."""
    valid = [(k, v) for k, v in values if v is not None]
    valid.sort(key=lambda x: -x[1])
    ranks = {}
    for i, (k, _) in enumerate(valid):
        ordinals = ["1st", "2nd", "3rd"] + [f"{j}th" for j in range(4, len(valid) + 1)]
        ranks[k] = ordinals[i]
    return ranks


def build_table(
    results,
    include_claimed_effective: bool = True,
):
    """Build table rows for printing and CSV."""
    # Sort models for consistent output
    models = sorted(results.keys())

    # Compute wAvg inc/dec and rankings
    wavg_inc = {m: compute_weighted_avg(results[m], inc=True) for m in models}
    wavg_dec = {m: compute_weighted_avg(results[m], inc=False) for m in models}
    rank_inc = rank_values([(m, wavg_inc[m]) for m in models])
    rank_dec = rank_values([(m, wavg_dec[m]) for m in models])

    rows = []
    for model in models:
        scores = results[model]
        avg_scores = [scores.get(seq) for seq in STANDARD_LENGTHS]
        avg = None
        valid = [s for s in avg_scores if s is not None]
        if valid:
            avg = round(sum(valid) / len(valid), 2)

        claimed, effective = "-", "-"
        if include_claimed_effective and model in MODEL_CONFIG:
            claimed, effective = MODEL_CONFIG[model]

        row = {
            "Models": model,
            "Claimed Length": claimed,
            "Effective Length": effective,
        }
        for seq, label in zip(STANDARD_LENGTHS, LENGTH_LABELS):
            row[label] = scores.get(seq)
        row["Avg."] = avg
        row["wAvg. (inc)"] = wavg_inc.get(model)
        row["wAvg. (dec)"] = wavg_dec.get(model)
        row["_rank_inc"] = rank_inc.get(model)
        row["_rank_dec"] = rank_dec.get(model)
        rows.append(row)
    return rows


def format_cell(val) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.2f}" if val == int(val) else f"{val:.2f}"
    return str(val)


def print_table(rows) -> None:
    """Print table to terminal."""
    cols = ["Models", "Claimed Length", "Effective Length"] + LENGTH_LABELS + ["Avg.", "wAvg. (inc)", "wAvg. (dec)"]
    # Build string rows
    str_rows = []
    for r in rows:
        sr = [format_cell(r.get(c)) for c in cols]
        # Append rankings to wAvg columns
        if r.get("_rank_inc"):
            idx = cols.index("wAvg. (inc)")
            sr[idx] = f"{sr[idx]} ({r['_rank_inc']})" if sr[idx] != "-" else f"- ({r['_rank_inc']})"
        if r.get("_rank_dec"):
            idx = cols.index("wAvg. (dec)")
            sr[idx] = f"{sr[idx]} ({r['_rank_dec']})" if sr[idx] != "-" else f"- ({r['_rank_dec']})"
        str_rows.append(sr)

    # Column widths
    widths = [max(len(str_rows[i][j]) for i in range(len(str_rows))) for j in range(len(cols))]
    widths = [max(w, len(cols[j])) for j, w in enumerate(widths)]

    sep = " | "
    header = sep.join(cols[j].ljust(widths[j]) for j in range(len(cols)))
    print(header)
    print("-" * len(header))
    for sr in str_rows:
        print(sep.join(sr[j].ljust(widths[j]) for j in range(len(cols))))


def write_csv(rows, path: Path) -> None:
    """Write table to CSV (without internal rank keys)."""
    cols = ["Models", "Claimed Length", "Effective Length"] + LENGTH_LABELS + ["Avg.", "wAvg. (inc)", "wAvg. (dec)"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {c: r.get(c) for c in cols}
            if r.get("_rank_inc"):
                v = r.get("wAvg. (inc)")
                out["wAvg. (inc)"] = f"{v} ({r['_rank_inc']})" if v is not None else f"- ({r['_rank_inc']})"
            if r.get("_rank_dec"):
                v = r.get("wAvg. (dec)")
                out["wAvg. (dec)"] = f"{v} ({r['_rank_dec']})" if v is not None else f"- ({r['_rank_dec']})"
            w.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Report RULER results as a table")
    parser.add_argument(
        "root",
        type=str,
        help="Root directory (e.g. outputs/0326/ruler or .../Llama-3.2-3B-Instruct/synthetic)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output CSV path (default: analysis/output/ruler_table.csv)",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Do not add Claimed/Effective Length columns",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="worst_single",
        choices=["mean_all", "worst_single"],
        help=(
            "How to compute each summary.csv score: "
            "'mean_all' (default) or 'worst_single' "
            "(min across single-needle tasks)."
        ),
    )
    parser.add_argument(
        "--single-task-prefix",
        type=str,
        default="niah_single_",
        help=(
            "Task name prefix used when --score-mode=worst_single "
            "(default: niah_single_)."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: directory not found: {root}")
        return 1

    results = collect_results(
        root,
        score_mode=args.score_mode,
        single_task_prefix=args.single_task_prefix,
    )
    if not results:
        print("No summary.csv files found under", root)
        return 1

    rows = build_table(results, include_claimed_effective=not args.no_config)

    print("\nRULER Results Table\n")
    print_table(rows)

    out_path = args.output
    if out_path is None:
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ruler_table.csv"
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_path)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
