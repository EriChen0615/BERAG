import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk


DEFAULT_OUTPUT_DIR = "analysis/output/positional_invariance"
DEFAULT_RETRIEVAL_DB = "outputs/0jingbiao_mei/EVQA-testfull-with-retrieval_post_reranked"
DEFAULT_BASE_CSV = "outputs/0326/EVQA/Qwen2-VL-7B-Instruct-retrieved_passage-Top20/marked_inference_results.csv"
DEFAULT_SFT_CSV = "outputs/1125/VLLM/Qwen2-VL-2B-Instruct-7B-EVQA-VLLM-SFT-RAG-K=5-Top20-Retrieve-TakeN=0/marked_inference_results.csv"
DEFAULT_DPO_CSV = "outputs/1125/VLLM/Qwen2-VL-2B-Instruct-EVQA-VLLM-DPO-RAG-K=5-Top20-Retrieve-TakeN=0/marked_inference_results.csv"
DEFAULT_BEFT_CSV = "outputs/1125-v3/BAPE/7B-EVQA-BAPE-BEFT[K*=2]-prior=mlp_lr1e-6-l0h4-lora_r64_bs8-epoch1-K=20-h4-prior=prior_head-retrieved_passage-TakeN=0/marked_inference_results.csv"

BIN_LABELS = ["1-4", "5-8", "9-12", "13-16", "17-20"]
BIN_EDGES = [1, 5, 9, 13, 17, 21]  # right-open ranges [1,5), [5,9), ...


def extract_passage_id(passage_obj) -> Optional[str]:
    """Extract passage_id from one retrieval_passage entry."""
    if isinstance(passage_obj, dict):
        pid = passage_obj.get("passage_id")
        if pid is not None:
            return str(pid)
    return None


def _qid_aliases(qid_raw) -> List[str]:
    """
    Build alias keys for robust joins across formats:
      - EVQA_123
      - 123
      - raw string
    """
    s = str(qid_raw).strip()
    aliases = {s}
    m = re.match(r"^EVQA_(\d+)$", s, flags=re.IGNORECASE)
    if m:
        num = m.group(1)
        aliases.add(num)
        aliases.add(f"EVQA_{num}")
    elif s.isdigit():
        aliases.add(f"EVQA_{s}")
    return list(aliases)


def compute_gt_rank_map(retrieval_db_path: str, topk: int = 20) -> Tuple[Dict[str, Optional[int]], Dict[str, int]]:
    """
    Build question_id -> GT rank map from retrieval dataset.
    GT rank is 1-based within retrieval_passage[:topk], or None if not found.
    """
    ds = load_from_disk(retrieval_db_path)

    qid_to_rank: Dict[str, Optional[int]] = {}
    stats = {
        "total_retrieval_examples": 0,
        "alias_keys_generated": 0,
        "missing_question_id": 0,
        "missing_gt_pos_item_ids": 0,
        "missing_retrieval_passage": 0,
        "used_field_retrieval_passage": 0,
        "used_field_retrieved_passage": 0,
        "used_field_reranked_passage": 0,
        "gt_not_in_topk": 0,
        "gt_found_in_topk": 0,
    }

    for row in ds:
        stats["total_retrieval_examples"] += 1
        qid = row.get("question_id", None)
        if qid is None:
            stats["missing_question_id"] += 1
            continue
        qid = str(qid).strip()

        pos_item_ids = row.get("pos_item_ids", None)
        if not isinstance(pos_item_ids, list) or len(pos_item_ids) == 0:
            stats["missing_gt_pos_item_ids"] += 1
            aliases = _qid_aliases(qid)
            stats["alias_keys_generated"] += len(aliases)
            for k in aliases:
                qid_to_rank[k] = None
            continue
        gt_pid = str(pos_item_ids[0])

        retrieval_passage = row.get("retrieval_passage", None)
        source_field = "retrieval_passage"
        if not isinstance(retrieval_passage, list) or len(retrieval_passage) == 0:
            retrieval_passage = row.get("retrieved_passage", None)
            source_field = "retrieved_passage"
        if not isinstance(retrieval_passage, list) or len(retrieval_passage) == 0:
            retrieval_passage = row.get("reranked_passage", None)
            source_field = "reranked_passage"

        if not isinstance(retrieval_passage, list) or len(retrieval_passage) == 0:
            stats["missing_retrieval_passage"] += 1
            aliases = _qid_aliases(qid)
            stats["alias_keys_generated"] += len(aliases)
            for k in aliases:
                qid_to_rank[k] = None
            continue
        if source_field == "retrieval_passage":
            stats["used_field_retrieval_passage"] += 1
        elif source_field == "retrieved_passage":
            stats["used_field_retrieved_passage"] += 1
        elif source_field == "reranked_passage":
            stats["used_field_reranked_passage"] += 1

        ranked_ids: List[str] = []
        for p in retrieval_passage[:topk]:
            pid = extract_passage_id(p)
            if pid is not None:
                ranked_ids.append(pid)

        if gt_pid in ranked_ids:
            rank = ranked_ids.index(gt_pid) + 1
            aliases = _qid_aliases(qid)
            stats["alias_keys_generated"] += len(aliases)
            for k in aliases:
                qid_to_rank[k] = rank
            stats["gt_found_in_topk"] += 1
        else:
            aliases = _qid_aliases(qid)
            stats["alias_keys_generated"] += len(aliases)
            for k in aliases:
                qid_to_rank[k] = None
            stats["gt_not_in_topk"] += 1

    return qid_to_rank, stats


def bin_rank(rank: Optional[int]) -> Optional[str]:
    if rank is None:
        return None
    if 1 <= rank <= 4:
        return "1-4"
    if 5 <= rank <= 8:
        return "5-8"
    if 9 <= rank <= 12:
        return "9-12"
    if 13 <= rank <= 16:
        return "13-16"
    if 17 <= rank <= 20:
        return "17-20"
    return None


def process_system_csv(system_name: str, csv_path: str, qid_to_rank: Dict[str, Optional[int]]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    required = {"question_id", "score"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{system_name}: missing required columns in {csv_path}: {sorted(missing)}")

    local = df[["question_id", "score"]].copy()
    local["question_id"] = local["question_id"].astype(str).str.strip()
    local["gt_rank"] = local["question_id"].map(qid_to_rank)
    local["rank_bin"] = local["gt_rank"].apply(bin_rank)
    local["system"] = system_name

    stats = {
        "rows_in_csv": len(local),
        "rows_with_rank": int(local["gt_rank"].notna().sum()),
        "rows_without_rank": int(local["gt_rank"].isna().sum()),
        "rows_in_bins": int(local["rank_bin"].notna().sum()),
    }

    binned = (
        local[local["rank_bin"].notna()]
        .groupby(["system", "rank_bin"], as_index=False)
        .agg(mean_score=("score", "mean"), count=("score", "count"))
    )

    # Ensure every bin exists for plotting consistency
    complete_rows = []
    for label in BIN_LABELS:
        matched = binned[binned["rank_bin"] == label]
        if len(matched) == 0:
            complete_rows.append(
                {
                    "system": system_name,
                    "rank_bin": label,
                    "mean_score": float("nan"),
                    "count": 0,
                }
            )
        else:
            complete_rows.append(matched.iloc[0].to_dict())

    return pd.DataFrame(complete_rows), stats


def make_plot(stats_df: pd.DataFrame, output_png: Path):
    pivot = stats_df.pivot(index="rank_bin", columns="system", values="mean_score")
    pivot = pivot.reindex(BIN_LABELS)

    systems = list(pivot.columns)
    x = range(len(BIN_LABELS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 7))
    offsets = [((i - (len(systems) - 1) / 2) * width) for i in range(len(systems))]

    for idx, system in enumerate(systems):
        y = pivot[system].tolist()
        positions = [v + offsets[idx] for v in x]
        ax.bar(positions, y, width=width, label=system)

    ax.set_xlabel("GT Document Position Bin", fontsize=16)
    ax.set_ylabel("Mean VQA Score", fontsize=16)
    ax.set_title("EVQA: VQA Score vs GT Document Position (Top-20 Retrieval)", fontsize=18)
    ax.set_xticks(list(x))
    ax.set_xticklabels(BIN_LABELS, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="System", fontsize=13, title_fontsize=14)
    plt.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def make_base_diff_plot(stats_df: pd.DataFrame, output_png: Path) -> pd.DataFrame:
    """
    Plot per-bin gains versus Base: (model_mean_score - base_mean_score).
    Only includes SFT/DPO/BEFT.
    Returns a tidy DataFrame of gain values for export.
    """
    pivot = stats_df.pivot(index="rank_bin", columns="system", values="mean_score")
    pivot = pivot.reindex(BIN_LABELS)

    if "Base" not in pivot.columns:
        raise ValueError("Base system not found in stats table; cannot compute gains.")

    compare_systems = [s for s in ["SFT", "DPO", "BEFT"] if s in pivot.columns]
    # Report gain in percentage points for readability in paper figures
    gain_pivot = pivot[compare_systems].subtract(pivot["Base"], axis=0) * 100.0

    x = range(len(BIN_LABELS))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 7))
    offsets = [((i - (len(compare_systems) - 1) / 2) * width) for i in range(len(compare_systems))]

    for idx, system in enumerate(compare_systems):
        y = gain_pivot[system].tolist()
        positions = [v + offsets[idx] for v in x]
        ax.bar(positions, y, width=width, label=system)

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("GT Document Position Bin", fontsize=16)
    ax.set_ylabel("VQA Gain vs Base (percentage points)", fontsize=16)
    ax.set_title("EVQA: VQA Gain vs Base by GT Document Position", fontsize=18)
    ax.set_xticks(list(x))
    ax.set_xticklabels(BIN_LABELS, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="System", fontsize=13, title_fontsize=14)
    plt.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    # Tidy export table
    gain_df = gain_pivot.reset_index().melt(
        id_vars="rank_bin",
        value_vars=compare_systems,
        var_name="system",
        value_name="gain_vs_base",
    )
    gain_df["rank_bin"] = pd.Categorical(gain_df["rank_bin"], categories=BIN_LABELS, ordered=True)
    gain_df = gain_df.sort_values(["rank_bin", "system"]).reset_index(drop=True)
    return gain_df


def main():
    parser = argparse.ArgumentParser(description="Analyze EVQA score versus GT doc position bins.")
    parser.add_argument("--retrieval_db", type=str, default=DEFAULT_RETRIEVAL_DB, help="Retrieval dataset path.")
    parser.add_argument("--base_csv", type=str, default=DEFAULT_BASE_CSV, help="Base marked_inference_results.csv path.")
    parser.add_argument("--sft_csv", type=str, default=DEFAULT_SFT_CSV, help="SFT marked_inference_results.csv path.")
    parser.add_argument("--dpo_csv", type=str, default=DEFAULT_DPO_CSV, help="DPO marked_inference_results.csv path.")
    parser.add_argument("--beft_csv", type=str, default=DEFAULT_BEFT_CSV, help="BEFT marked_inference_results.csv path.")
    parser.add_argument("--topk", type=int, default=20, help="Top-K retrieval depth for GT position (default: 20).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for outputs.")
    args = parser.parse_args()

    retrieval_db = args.retrieval_db
    system_paths = {
        "Base": args.base_csv,
        "SFT": args.sft_csv,
        "DPO": args.dpo_csv,
        "BEFT": args.beft_csv,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading retrieval dataset: {retrieval_db}")
    qid_to_rank, retrieval_stats = compute_gt_rank_map(retrieval_db, topk=args.topk)
    print("Retrieval DB stats:")
    print(json.dumps(retrieval_stats, indent=2))

    all_stats_rows = []
    per_system_summary = {}
    for system_name, csv_path in system_paths.items():
        print(f"\nProcessing {system_name}: {csv_path}")
        stats_df, system_stat = process_system_csv(system_name, csv_path, qid_to_rank)
        all_stats_rows.append(stats_df)
        per_system_summary[system_name] = system_stat
        print(json.dumps(system_stat, indent=2))

    final_stats_df = pd.concat(all_stats_rows, ignore_index=True)
    final_stats_df["rank_bin"] = pd.Categorical(final_stats_df["rank_bin"], categories=BIN_LABELS, ordered=True)
    final_stats_df = final_stats_df.sort_values(["rank_bin", "system"]).reset_index(drop=True)

    stats_csv = out_dir / "positional_invariance_stats.csv"
    plot_png = out_dir / "positional_invariance_barplot.png"
    basediff_stats_csv = out_dir / "positional_invariance_basediff_stats.csv"
    basediff_plot_png = out_dir / "positional_invariance_barplot_basediff.png"
    summary_json = out_dir / "positional_invariance_summary.json"

    final_stats_df.to_csv(stats_csv, index=False)
    make_plot(final_stats_df, plot_png)
    basediff_df = make_base_diff_plot(final_stats_df, basediff_plot_png)
    basediff_df.to_csv(basediff_stats_csv, index=False)

    summary = {
        "retrieval_db": retrieval_db,
        "system_csv_paths": system_paths,
        "topk": args.topk,
        "bins": BIN_LABELS,
        "retrieval_stats": retrieval_stats,
        "per_system_summary": per_system_summary,
        "output_stats_csv": str(stats_csv),
        "output_plot_png": str(plot_png),
        "output_basediff_stats_csv": str(basediff_stats_csv),
        "output_basediff_plot_png": str(basediff_plot_png),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved files:")
    print(f"  {stats_csv}")
    print(f"  {plot_png}")
    print(f"  {basediff_stats_csv}")
    print(f"  {basediff_plot_png}")
    print(f"  {summary_json}")


if __name__ == "__main__":
    main()

