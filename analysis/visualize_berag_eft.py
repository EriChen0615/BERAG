#!/usr/bin/env python3
"""
Visualize BERAG ESF beam log: context length, number of chunks, final response by segment,
and belief heatmap with ground-truth state highlighted.

Before running, activate the inference environment, e.g.:
  source scripts/hpc_activate_env_py310_infer.sh
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_context_and_depth_from_path(beam_log_path: str) -> tuple[int | None, float | None]:
    """Parse context_length and depth_percent from beam log filename or sibling results.json."""
    beam_log_path = Path(beam_log_path)
    # Try sibling results.json first (e.g. ..._len_100000_depth_0_results.json)
    stem = beam_log_path.stem.replace(".beam_log", "")
    results_path = beam_log_path.parent / f"{stem}.json"
    if results_path.exists():
        try:
            with open(results_path, encoding="utf-8") as f:
                data = json.load(f)
            ctx = data.get("context_length")
            depth = data.get("depth_percent")
            return (ctx, depth)
        except (json.JSONDecodeError, KeyError):
            pass
    # Fallback: parse filename like *_len_100000_depth_0_* or *_len_100000_depth_10000_*
    name = beam_log_path.name
    m_len = re.search(r"_len_(\d+)_", name)
    m_depth = re.search(r"_depth_(\d+)_?", name)
    ctx = int(m_len.group(1)) if m_len else None
    depth_raw = int(m_depth.group(1)) if m_depth else None
    depth = (depth_raw / 100.0) if depth_raw is not None else None  # e.g. 10000 -> 100.0
    return (ctx, depth)


def load_beam_log(path: str) -> list[dict]:
    """Load JSONL beam log; return list of records."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get_best_beam_per_segment(records: list[dict]) -> dict[int, dict]:
    """Return dict segment_idx -> record with highest alpha for that segment."""
    by_segment: dict[int, list[dict]] = {}
    for r in records:
        seg = r["segment"]
        by_segment.setdefault(seg, []).append(r)
    return {
        seg: max(beams, key=lambda b: b["alpha"])
        for seg, beams in sorted(by_segment.items())
    }


def log_bar_b_to_probs(log_bar_b: list[float]) -> np.ndarray:
    """Convert log beliefs to normalized probabilities (softmax)."""
    a = np.array(log_bar_b, dtype=np.float64)
    a = a - a.max()
    exp_a = np.exp(a)
    return exp_a / exp_a.sum()


def get_ground_truth_chunk_index(depth_percent: float | None, num_chunks: int) -> int | None:
    """Infer ground-truth chunk index from depth percent (0 = first chunk, 100 = last)."""
    if depth_percent is None or num_chunks <= 0:
        return None
    return min(int(round(depth_percent / 100.0 * (num_chunks - 1))), num_chunks - 1)


def collect_unique_states(records: list[dict]) -> tuple[list[tuple], dict[tuple, int]]:
    """Collect all unique states (as tuples) across all segments; return ordered list and state->index map."""
    seen: set[tuple] = set()
    for r in records:
        for s in r.get("state_list_bar", r.get("state_list", [])):
            seen.add(tuple(s) if isinstance(s, (list, tuple)) else (s,))
    unique_states = sorted(seen)
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    return unique_states, state_to_idx


def build_heatmap_matrix(
    best_per_segment: dict[int, dict],
    state_to_idx: dict[tuple, int],
    num_unique_states: int,
) -> np.ndarray:
    """Build (num_unique_states, num_segments) matrix of belief probs; state axis = global unique state index."""
    segments = sorted(best_per_segment.keys())
    H = np.zeros((num_unique_states, len(segments)))
    for j, seg in enumerate(segments):
        r = best_per_segment[seg]
        state_list_bar = r.get("state_list_bar", r.get("state_list", []))
        log_b = r.get("log_bar_b")
        if not state_list_bar or not log_b or len(state_list_bar) != len(log_b):
            continue
        probs = log_bar_b_to_probs(log_b)
        for i, s in enumerate(state_list_bar):
            key = tuple(s) if isinstance(s, (list, tuple)) else (s,)
            if key in state_to_idx:
                row = state_to_idx[key]
                H[row, j] = probs[i]
    return H


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize BERAG ESF beam log: response by segment and belief heatmap."
    )
    parser.add_argument(
        "beam_log_path",
        type=str,
        help="Path to beam log JSONL (e.g. .../simple_niah_berag_esf_len_100000_depth_0_results.beam_log.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output figure path (default: same as beam log with .png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default 150).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show().",
    )
    args = parser.parse_args()

    path = Path(args.beam_log_path)
    if not path.exists():
        raise FileNotFoundError(f"Beam log not found: {path}")

    records = load_beam_log(path)
    if not records:
        raise ValueError(f"No records in beam log: {path}")

    # Context length and depth from filename or sibling results.json
    context_length, depth_percent = parse_context_and_depth_from_path(path)
    first = records[0]
    state_list_bar = first.get("state_list_bar", first.get("state_list", []))
    num_chunks = len(state_list_bar) if state_list_bar else 0
    gt_chunk = get_ground_truth_chunk_index(depth_percent, num_chunks) if num_chunks else None

    best_per_segment = get_best_beam_per_segment(records)
    segment_texts = [
        best_per_segment[seg].get("segment_text", "")
        for seg in sorted(best_per_segment.keys())
    ]
    full_response = "".join(segment_texts).strip() or best_per_segment[max(best_per_segment)].get("generated_text", "")

    # Unique states across all segments (shown as integer indices on y-axis)
    unique_states, state_to_idx = collect_unique_states(records)
    num_unique_states = len(unique_states)
    heatmap = build_heatmap_matrix(best_per_segment, state_to_idx, num_unique_states)
    num_segments = heatmap.shape[1]
    gt_state_idx = state_to_idx.get((gt_chunk,)) if gt_chunk is not None else None

    # Build word-level (segment_idx, word) for full response so we can wrap one paragraph and color by segment
    words_with_seg: list[tuple[int, str]] = []
    for seg_idx, text in enumerate(segment_texts):
        for word in (text or "").split():
            words_with_seg.append((seg_idx, word))

    max_chars_per_line = 100
    x_start, x_end = 0.02, 0.98
    lines_of_words: list[list[tuple[int, str]]] = []
    current_line: list[tuple[int, str]] = []
    current_len = 0
    for seg_idx, word in words_with_seg:
        need_space = len(current_line) > 0
        add_len = (1 if need_space else 0) + len(word)
        if current_len + add_len > max_chars_per_line and current_line:
            lines_of_words.append(current_line)
            current_line = [(seg_idx, word)]
            current_len = len(word)
        else:
            current_line.append((seg_idx, word))
            current_len += add_len
    if current_line:
        lines_of_words.append(current_line)

    n_lines = len(lines_of_words)
    top_ratio = max(1.2, 0.4 + n_lines * 0.08)
    fig, (ax_text, ax_heat) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[top_ratio, 2], sharex=False)
    fig.subplots_adjust(hspace=0.35)

    # ---- Top: full response as one paragraph, color-coded by segment ----
    ax_text.set_title("Final response (most likely beam per segment)", fontsize=11)
    ax_text.set_ylim(0, 1)
    ax_text.set_xlim(0, 1)
    ax_text.axis("off")

    fontsize = 10
    segment_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    y0 = 0.85
    line_height = (y0 - 0.05) / max(1, n_lines)
    width_per_char = (x_end - x_start) / max_chars_per_line

    y = y0
    for line_words in lines_of_words:
        x = x_start
        need_space = False
        for seg_idx, word in line_words:
            display_word = (" " + word) if need_space else word
            need_space = True
            color = segment_colors[seg_idx % len(segment_colors)]
            ax_text.text(
                x, y, display_word,
                transform=ax_text.transAxes,
                fontsize=fontsize,
                verticalalignment="center",
                color=color,
            )
            x += len(display_word) * width_per_char
        y -= line_height
    ax_text.set_ylim(0, y0 + 0.05)

    # Info text: context length, number of chunks, ground-truth
    info_parts = []
    if context_length is not None:
        info_parts.append(f"Context length: {context_length:,}")
    info_parts.append(f"Number of chunks: {num_chunks}")
    if depth_percent is not None:
        info_parts.append(f"Depth: {depth_percent}%")
    if gt_chunk is not None:
        info_parts.append(f"Ground-truth chunk: {gt_chunk}")
    ax_text.text(0.02, 0.98, "  |  ".join(info_parts), transform=ax_text.transAxes, fontsize=9, verticalalignment="top")

    # ---- Bottom: heatmap ----
    # X = segment, Y = state index (integer), value = belief prob; clear grid at cell boundaries
    im = ax_heat.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap="plasma",
        norm=mcolors.LogNorm(vmin=1e-6, vmax=1.0),
        extent=[-0.5, num_segments - 0.5, num_unique_states - 0.5, -0.5],
    )
    ax_heat.set_xlabel("Segment")
    ax_heat.set_ylabel("State (index)")
    ax_heat.set_title("Belief weight (normalized b̄) for chosen beam per segment")
    ax_heat.set_xticks(range(num_segments))
    ax_heat.set_yticks(range(num_unique_states))
    ax_heat.set_xticks(np.arange(num_segments + 1) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(num_unique_states + 1) - 0.5, minor=True)
    ax_heat.grid(True, which="minor", color="white", linewidth=0.8)
    ax_heat.tick_params(which="minor", size=0)
    plt.colorbar(im, ax=ax_heat, label="Probability", shrink=0.8)

    # Highlight ground-truth state row (horizontal line at that state index)
    if gt_state_idx is not None:
        ax_heat.axhline(y=gt_state_idx, color="red", linewidth=2, linestyle="--", alpha=0.9, label="Ground-truth state")
        ax_heat.legend(loc="upper right", fontsize=8)

    out_path = args.output or str(path.with_suffix(".png"))
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if not args.no_show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
