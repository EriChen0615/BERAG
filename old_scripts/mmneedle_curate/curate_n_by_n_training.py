#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


PROMPT_TEMPLATE = (
    "<<<EVIDENCE>>>\n"
    "Given M images indexed from 1 to M, each divided into NxN sub-images, identify the sub-image that best matches the provided caption. "
    "Respond with \"index, row, column\" and nothing else. For example, \"1, 2, 3\" indicates the sub-image in the first image, "
    "second row, and third column. If no match is found, respond only with \"-1\".\n"
    "Caption: <<<QUESTION>>>"
)


def load_caption_map(captions_json: Path) -> Dict[str, str]:
    with captions_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    filename_to_caption = {}
    for ann in data["annotations"]:
        fname = id_to_filename.get(ann["image_id"])
        if fname is None:
            continue
        if fname not in filename_to_caption:
            filename_to_caption[fname] = ann["caption"]
    return filename_to_caption


def assistant_target(index: int, row: int, col: int) -> str:
    if index < 0:
        return "-1"
    return f"{index + 1}, {row + 1}, {col + 1}"


def _select_k_indices(image_ids: List[str], gt_idx: int, k: int) -> List[int]:
    total = len(image_ids)
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > total:
        raise ValueError(f"k={k} is larger than available sequence length {total}")

    if gt_idx >= 0:
        negative_candidates = [i for i in range(total) if i != gt_idx]
        chosen_negatives = random.sample(negative_candidates, k - 1)
        chosen = [gt_idx] + chosen_negatives
    else:
        chosen = random.sample(list(range(total)), k)

    # Keep the passage order consistent with original sequence order.
    chosen.sort()
    return chosen


def build_example(row: Dict, caption_map: Dict[str, str], k: int, add_z0: bool) -> Dict:
    target_path = str(row["target"])
    target_name = Path(target_path).name
    caption = caption_map.get(target_name, f"Describe image file: {target_name}")

    all_image_ids = [str(x) for x in row["image_ids"]]
    orig_idx = int(row["index"])
    r = int(row["row"])
    c = int(row["col"])

    chosen_indices = _select_k_indices(all_image_ids, orig_idx, k)
    image_ids = [all_image_ids[i] for i in chosen_indices]
    # Keep answer index in the original sequence coordinate system.
    answer = assistant_target(orig_idx, r, c)

    passages = []
    for original_i, img_path in zip(chosen_indices, image_ids):
        passages.append(
            {
                "images": [img_path],
                "text": f"<image> [IMAGE_INDEX] {original_i + 1}",
            }
        )

    # gt_passage_idx is local to the sampled passages and must stay aligned with passage_scores.
    local_gt_idx = chosen_indices.index(orig_idx) if orig_idx >= 0 else -1

    if add_z0:
        passages.append({"images": [], "text": "[IMAGE_INDEX] -1"})
        z0_idx = len(passages) - 1
    else:
        z0_idx = -1

    # For negative examples with add_z0 enabled, z0 is the ground truth passage.
    if local_gt_idx < 0 and add_z0:
        local_gt_idx = z0_idx

    gt_passage_idx = [local_gt_idx] if local_gt_idx >= 0 else [-1]
    passage_scores = [0.0] * len(passages)
    if local_gt_idx >= 0 and local_gt_idx < len(passage_scores):
        passage_scores[local_gt_idx] = 1.0

    user_msg = PROMPT_TEMPLATE.replace("<<<QUESTION>>>", caption)

    return {
        "messages": [
            {"content": user_msg, "role": "user"},
            {"content": answer, "role": "assistant"},
        ],
        "images": [],
        "gt_passage_idx": gt_passage_idx,
        "passages": passages,
        "passage_scores": passage_scores,
        "meta": {
            "sample_id": int(row["id"]),
            "caption_file": target_name,
            "caption": caption,
            "target": target_path,
            "index": orig_idx,
            "local_gt_passage_idx": local_gt_idx,
            "z0_idx": z0_idx,
            "has_needle": orig_idx >= 0,
            "row": r,
            "col": c,
        },
    }


def summarize(data: List[Dict]) -> Dict:
    n = len(data)
    pos = sum(1 for x in data if bool(x.get("meta", {}).get("has_needle", False)))
    neg = n - pos
    avg_passages = sum(len(x["passages"]) for x in data) / n if n > 0 else 0.0
    return {
        "num_samples": n,
        "num_positive": pos,
        "num_negative": neg,
        "avg_passages_per_sample": avg_passages,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MMNeedle annotations into BEFT train_sharegpt.json format.")
    parser.add_argument("--n_grid", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--k", type=int, default=2, help="Number of passages/images per training sample.")
    parser.add_argument("--add_z0", action="store_true", help="Append null document z0 ('[IMAGE_INDEX] -1') to all samples.")
    parser.add_argument("--take_n", type=int, default=0, help="0 means use all.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metadata_root", type=str, default="../vqa_data/MMNeedle/train/metadata_stitched")
    parser.add_argument("--captions_json", type=str, default="../vqa_data/MSCOCO2014/annotations/captions_train2014.json")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If unset, writes to third_party/LLaMA-Factory-2502/data/jinghong_chen/mmneedle/...",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    metadata_root = Path(args.metadata_root).resolve()
    ann_path = metadata_root / f"annotations_{args.sequence_length}_{args.n_grid}_{args.n_grid}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {ann_path}")

    caption_map = load_caption_map(Path(args.captions_json).resolve())
    with ann_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if args.offset > 0:
        rows = rows[args.offset :]
    if args.take_n > 0:
        rows = rows[: args.take_n]

    if args.k > args.sequence_length:
        raise ValueError(f"k={args.k} cannot be larger than sequence_length={args.sequence_length}")

    examples = []
    for r in tqdm(rows, desc=f"Curating BEFT samples N={args.n_grid}, L={args.sequence_length}"):
        examples.append(build_example(r, caption_map, args.k, args.add_z0))

    z0_tag = "-z0" if args.add_z0 else ""
    default_out = (
        Path("../vqa_data/MMNeedle/train/curated")
        / (
            f"rag{args.sequence_length}-mmneedle-n{args.n_grid}x{args.n_grid}"
            f"-k={args.k}{z0_tag}-beft-size={args.take_n}-offset={args.offset}"
        )
    )
    out_dir = Path(args.output_dir).resolve() if args.output_dir else default_out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "train_sharegpt.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    stats = summarize(examples)
    stats.update(
        {
            "n_grid": args.n_grid,
            "sequence_length": args.sequence_length,
            "k": args.k,
            "add_z0": args.add_z0,
            "take_n": args.take_n,
            "offset": args.offset,
            "annotations_path": str(ann_path),
            "output_json": str(out_json),
        }
    )
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved training data: {out_json}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

