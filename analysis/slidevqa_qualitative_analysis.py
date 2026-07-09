import argparse
import ast
import math
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DEFAULT_CSV_PATH = (
    "outputs/1225/BAPE/SlideVQA/"
    "SlideVQA-BAPE-BEFT[K=4*]-prior=mlp-lr1e-6-l1h4-r64-epoch1-h4-"
    "prior=prior_head-K=20-TakeN=0/marked_inference_results.csv"
)
DEFAULT_OUTPUT_DIR = "analysis/output/slidevqa_qualitative"


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


def normalize_int_list(values: Iterable) -> List[int]:
    out = []
    for item in values:
        try:
            out.append(int(item))
        except Exception:
            continue
    # De-duplicate while preserving order.
    return list(dict.fromkeys(out))


def softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    denom = sum(exps)
    if denom == 0:
        return [0.0 for _ in exps]
    return [v / denom for v in exps]


def entropy_from_probs(probs: List[float]) -> float:
    e = 0.0
    for p in probs:
        if p > 0.0:
            e -= p * math.log(p)
    return e


def to_float_list(val) -> List[float]:
    arr = parse_list_like(val)
    out = []
    for x in arr:
        try:
            out.append(float(x))
        except Exception:
            continue
    return out


def count_words(answer) -> int:
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        return 0
    text = str(answer).strip()
    if not text:
        return 0
    return len(text.split())


def compute_gt_slide_paths(base_img_path: str, gt_pages: List[int], repo_root: Path) -> List[str]:
    if not base_img_path:
        return []

    img_path = Path(base_img_path)
    if img_path.is_absolute():
        abs_img = img_path
    else:
        abs_img = (repo_root / img_path).resolve()

    matched = re.search(r"_page_\d+(\.[^.]+)$", abs_img.name)
    if not matched:
        return [str(abs_img)]

    suffix = matched.group(1)
    prefix = abs_img.name[: matched.start()]
    paths = []
    for p in gt_pages:
        paths.append(str((abs_img.parent / f"{prefix}_page_{p}{suffix}").resolve()))
    return paths


def write_category_report(output_path: Path, category_name: str, rows: pd.DataFrame, repo_root: Path) -> None:
    lines = []
    lines.append(f"Category: {category_name}")
    lines.append(f"Num examples: {len(rows)}")
    lines.append("")

    for example_idx, (_, row) in enumerate(rows.iterrows(), start=1):
        qa_id = int(row["qa_id"])
        question = str(row.get("question", ""))
        gt_answer = str(row.get("gold_answer", ""))
        gen_answer = str(row.get("generated_answer", ""))
        gt_pages = row.get("gt_pages", [])
        prior_logits = row.get("prior_logits_list", [])
        prior_probs = row.get("prior_probs", [])
        passage_pages = row.get("passage_page_nums_list", [])
        prior_entropy = float(row.get("prior_entropy", 0.0))
        answer_word_count = int(row.get("answer_word_count", 0))

        gt_slide_paths = compute_gt_slide_paths(str(row.get("img_path", "")), gt_pages, repo_root=repo_root)

        lines.append(f"[Example {example_idx}]")
        lines.append(f"Question ID: {qa_id}")
        lines.append(f"Question: {question}")
        lines.append("GT Slide Images:")
        if gt_slide_paths:
            for p in gt_slide_paths:
                lines.append(f"- {p}")
        else:
            lines.append("- <none>")
        lines.append(f"GT answer: {gt_answer}")
        lines.append(f"Generated answer: {gen_answer}")
        lines.append(f"Answer word count: {answer_word_count}")
        lines.append(f"Prior entropy: {prior_entropy:.6f}")
        lines.append("Prior Distribution:")
        if prior_logits and prior_probs:
            if passage_pages and len(passage_pages) == len(prior_probs):
                for page, logit, prob in zip(passage_pages, prior_logits, prior_probs):
                    lines.append(f"- page_{page}: logit={logit:.6f}, prob={prob:.6f}")
            else:
                for i, (logit, prob) in enumerate(zip(prior_logits, prior_probs), start=1):
                    lines.append(f"- idx_{i}: logit={logit:.6f}, prob={prob:.6f}")
        else:
            lines.append("- <invalid prior logits>")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SlideVQA qualitative analysis report generator.")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Reserved for deterministic tie-breaking.")
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top_n must be positive.")

    repo_root = Path.cwd().resolve()
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = (repo_root / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV: {csv_path}")
    print(f"Seed: {args.seed}")
    df = pd.read_csv(csv_path)

    required_cols = ["qa_id", "question", "img_path", "gold_answer", "generated_answer", "exact_match", "evidence_page_nums", "prior_logits"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["qa_id"] = df["qa_id"].astype(int)
    df["gt_pages"] = df["evidence_page_nums"].apply(lambda v: normalize_int_list(parse_list_like(v)))
    df["gt_count"] = df["gt_pages"].apply(len)
    df["is_multi_gt"] = df["gt_count"] >= 2
    df["answer_word_count"] = df["generated_answer"].apply(count_words)
    df["prior_logits_list"] = df["prior_logits"].apply(to_float_list)
    df["prior_probs"] = df["prior_logits_list"].apply(softmax)
    df["prior_entropy"] = df["prior_probs"].apply(entropy_from_probs)
    if "passage_page_nums" in df.columns:
        df["passage_page_nums_list"] = df["passage_page_nums"].apply(lambda v: normalize_int_list(parse_list_like(v)))
    else:
        df["passage_page_nums_list"] = [[] for _ in range(len(df))]

    categories = {}
    categories["long_answer"] = (
        df.sort_values(by=["answer_word_count", "qa_id"], ascending=[False, True]).head(args.top_n).copy()
    )
    multi_gt_correct_pool = df[(df["is_multi_gt"]) & (df["exact_match"] == 1)].copy()
    categories["multi_gt_correct"] = multi_gt_correct_pool.sort_values(by=["qa_id"], ascending=[True]).head(args.top_n).copy()
    single_gt_correct_pool = df[(~df["is_multi_gt"]) & (df["exact_match"] == 1)].copy()
    categories["high_entropy_single_gt_correct"] = (
        single_gt_correct_pool.sort_values(by=["prior_entropy", "qa_id"], ascending=[False, True]).head(args.top_n).copy()
    )
    categories["high_entropy_multi_gt_correct"] = (
        multi_gt_correct_pool.sort_values(by=["prior_entropy", "qa_id"], ascending=[False, True]).head(args.top_n).copy()
    )

    report_paths = {
        "long_answer": output_dir / "long_answer.txt",
        "multi_gt_correct": output_dir / "multi_gt_correct.txt",
        "high_entropy_single_gt_correct": output_dir / "high_entropy_single_gt_correct.txt",
        "high_entropy_multi_gt_correct": output_dir / "high_entropy_multi_gt_correct.txt",
    }

    for category_name, out_path in report_paths.items():
        rows = categories[category_name]
        write_category_report(out_path, category_name, rows, repo_root=repo_root)

    print("\nSelection summary")
    print("-----------------")
    print(f"long_answer: candidates={len(df)}, selected={len(categories['long_answer'])}")
    print(f"multi_gt_correct: candidates={len(multi_gt_correct_pool)}, selected={len(categories['multi_gt_correct'])}")
    print(
        "high_entropy_single_gt_correct: "
        f"candidates={len(single_gt_correct_pool)}, selected={len(categories['high_entropy_single_gt_correct'])}"
    )
    print(
        "high_entropy_multi_gt_correct: "
        f"candidates={len(multi_gt_correct_pool)}, selected={len(categories['high_entropy_multi_gt_correct'])}"
    )

    for category_name in ["high_entropy_single_gt_correct", "high_entropy_multi_gt_correct"]:
        vals = categories[category_name]["prior_entropy"].tolist()
        print(f"{category_name} top entropy values: {[round(v, 6) for v in vals]}")

    print("\nGenerated reports:")
    for name, path in report_paths.items():
        print(f"- {name}: {path}")

    for name, rows in categories.items():
        if len(rows) < args.top_n:
            print(f"WARNING: {name} has only {len(rows)} examples (< top_n={args.top_n}).")


if __name__ == "__main__":
    main()
