import argparse
import ast
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# Hard-coded CSV path (requested)
# CSV_PATH = (
#     "/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA/outputs/1225/BAPE/"
#     "SlideVQA/SlideVQA-BAPE-BEFT[K=4*]-prior=mlp-lr1e-6-l1h4-r64-epoch1-h4-"
#     "prior=prior_head-K=20-TakeN=0/marked_inference_results.csv"
# )
CSV_PATH = (
"outputs/1225/VLLM/SlideVQA-Qwen2-VL-7B-Instruct-SlideVQA-VLLM-SFT-TakeN=0-Split=test/marked_inference_results.csv"
)

# SlideVQA dataset on HuggingFace
HF_DATASET_NAME = "NTT-hil-insight/SlideVQA"


def _parse_list_like(val):
    """
    Parse fields that are sometimes stored as:
      - Python list
      - string like "[1, 7]" or "['page_1', 'page_2']"
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def normalize_evidence_pages(evidence_pages):
    """
    Normalize evidence page identifiers to integers.
    Accepts ints (e.g. 5) or strings (e.g. "page_5").
    """
    pages = []
    for p in evidence_pages if isinstance(evidence_pages, list) else [evidence_pages]:
        if p is None:
            continue
        if isinstance(p, int):
            pages.append(p)
        elif isinstance(p, str):
            s = p.strip()
            if not s:
                continue
            if s.startswith("page_"):
                try:
                    pages.append(int(s.split("page_", 1)[1]))
                except Exception:
                    continue
            else:
                try:
                    pages.append(int(s))
                except Exception:
                    continue
        else:
            try:
                pages.append(int(p))
            except Exception:
                continue

    # De-duplicate while preserving first-seen order
    return list(dict.fromkeys(pages))


def main():
    parser = argparse.ArgumentParser(description="Analyze SlideVQA multi-GT error cases.")
    parser.add_argument("--split", type=str, default="test", help="SlideVQA dataset split (default: test).")
    parser.add_argument(
        "--max_error_examples",
        type=int,
        default=30,
        help="How many multi-GT error examples to preview/export (default: 30).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output" / "0326" / "slidevqa_multigt_analysis"),
        help="Output directory (default: analysis/output/slidevqa_multigt_analysis).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"CSV_PATH does not exist: {CSV_PATH}")

    print(f"Loading marked results CSV:\n  {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if "qa_id" not in df.columns:
        raise KeyError("Expected column `qa_id` in CSV.")

    # Ensure consistent type for joining with dataset
    df["qa_id"] = df["qa_id"].astype(int)

    print(f"Loading SlideVQA from HuggingFace ({HF_DATASET_NAME}, split={args.split})...")
    ds = load_dataset(HF_DATASET_NAME, split=args.split)

    # Map qa_id -> normalized evidence page ints (ground-truth)
    ds_ev_map = {}
    for item in ds:
        qa_id = int(item["qa_id"])
        ev_pages = normalize_evidence_pages(item.get("evidence_pages", []))
        ds_ev_map[qa_id] = ev_pages

    # Label each CSV row by number of GT evidence pages
    gt_counts = []
    gt_groups = []
    for qa_id in df["qa_id"].tolist():
        ev_pages = ds_ev_map.get(qa_id, [])
        gt_count = len(ev_pages)
        gt_counts.append(gt_count)
        if gt_count == 1:
            gt_groups.append("1_gt")
        elif gt_count >= 2:
            gt_groups.append("multi_gt")
        else:
            gt_groups.append("0_gt")

    df["gt_slide_count"] = gt_counts
    df["gt_group"] = gt_groups

    # Optional sanity check: compare dataset evidence_pages to CSV's evidence_page_nums
    mismatch_info = None
    if "evidence_page_nums" in df.columns:
        csv_ev_counts = []
        csv_ev_pages_preview = []
        for v in df["evidence_page_nums"].tolist():
            csv_pages = normalize_evidence_pages(_parse_list_like(v))
            csv_ev_counts.append(len(csv_pages))
            csv_ev_pages_preview.append(csv_pages)
        df["csv_evidence_page_count"] = csv_ev_counts
        mismatch_mask = df["csv_evidence_page_count"] != df["gt_slide_count"]
        mismatch_n = int(mismatch_mask.sum())
        mismatch_info = (mismatch_n, mismatch_mask.sum() > 0, df.loc[mismatch_mask, ["qa_id", "gt_slide_count", "csv_evidence_page_count"]].head(10))
        print(f"Evidence page count sanity check: mismatches={mismatch_n}/{len(df)}")
        if mismatch_n > 0:
            print("Sample mismatches (first 10):")
            print(mismatch_info[2].to_string(index=False))

    # Summary: does answer quality drop on multi-GT?
    required_cols = ["exact_match", "gt_group"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Expected column `{c}` in CSV.")

    def _safe_mean(col):
        if col not in df.columns:
            return None
        return float(df[col].mean())

    metrics = []
    for group in ["1_gt", "multi_gt"]:
        gdf = df[df["gt_group"] == group]
        if len(gdf) == 0:
            continue
        row = {"gt_group": group, "n_examples": len(gdf)}
        # Answer correctness
        row["answer_exact_match"] = float(gdf["exact_match"].mean()) if "exact_match" in gdf.columns else None
        # Evidence quality signals (these exist in the marked CSV from this repo)
        for col in [
            "prior_evidence_em",
            "prior_evidence_em_at_topk",
            "dominant_passage_is_gt",
            "prior_passage_is_gt",
        ]:
            if col in gdf.columns:
                row[col] = float(gdf[col].mean())
        metrics.append(row)

    summary_df = pd.DataFrame(metrics).sort_values("gt_group")
    print("\nSummary by GT evidence-page bucket:")
    if len(summary_df) > 0:
        print(summary_df.to_string(index=False))
    else:
        print("No rows found for groups `1_gt` / `multi_gt`.")

    # Error analysis for multi-GT: where exact_match==0
    if "exact_match" not in df.columns:
        print("Column `exact_match` not found; skipping error-case export.")
        return

    error_df = df[df["exact_match"] == 0].copy()
    multi_error_df = error_df[error_df["gt_group"] == "multi_gt"].copy()
    sort_cols = [c for c in ["prior_evidence_em_at_topk", "prior_evidence_em", "prior_passage_is_gt"] if c in multi_error_df.columns]
    if sort_cols:
        multi_error_df = multi_error_df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    print(f"\nMulti-GT error cases (exact_match==0): {len(multi_error_df)}")

    # Among multi-GT errors, how often did evidence retrieval succeed?
    for evidence_col in ["prior_evidence_em_at_topk", "prior_evidence_em", "prior_passage_is_gt", "dominant_passage_is_gt"]:
        if evidence_col in multi_error_df.columns:
            rate = float(multi_error_df[evidence_col].mean())
            print(f"  Mean({evidence_col}) among multi-GT errors: {rate:.4f}")

    preview_n = max(0, int(args.max_error_examples))
    if preview_n > 0 and len(multi_error_df) > 0:
        cols_for_export = [
            "qa_id",
            "question",
            "img_path",
            "gold_answer",
            "generated_answer",
            "evidence_page_nums",
            "gt_slide_count",
            "prior_evidence_em",
            "prior_evidence_em_at_topk",
            "dominant_passage_is_gt",
            "prior_passage_is_gt",
            "exact_match",
        ]
        cols_for_export = [c for c in cols_for_export if c in multi_error_df.columns]
        export_df = multi_error_df.head(preview_n)[cols_for_export]

        export_path = out_dir / "multi_gt_error_cases_preview.csv"
        export_df.to_csv(export_path, index=False)
        print(f"\nExported preview CSV:\n  {export_path}")

        # Also print a small text preview for quick eyeballing
        print("\nTop preview rows (first few):")
        print(export_df.head(min(10, len(export_df))).to_string(index=False))
    else:
        print("No multi-GT error examples to export/preview.")


if __name__ == "__main__":
    main()

