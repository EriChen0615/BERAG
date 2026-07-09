#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


def load_stitched_images(stitched_dir: Path) -> List[Path]:
    return sorted([p for p in stitched_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def flatten_meta_values(meta_for_one_file: Dict[str, str]) -> List[str]:
    return [str(v) for _, v in sorted(meta_for_one_file.items(), key=lambda kv: kv[0])]


def main():
    parser = argparse.ArgumentParser(description="Generate MMNeedle-style single-needle annotations for NxN stitched images.")
    parser.add_argument("--n_grid", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--positive_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--images_root", type=str, default="../vqa_data/MMNeedle/train/images_stitched")
    parser.add_argument("--metadata_root", type=str, default="../vqa_data/MMNeedle/train/metadata_stitched")
    parser.add_argument("--coco_train_dir", type=str, required=True, help="COCO train2014 directory for negative target sampling.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output path; default uses MMNeedle naming.")
    args = parser.parse_args()

    random.seed(args.seed)

    images_root = Path(args.images_root).resolve()
    metadata_root = Path(args.metadata_root).resolve()
    stitched_dir = images_root / f"{args.n_grid}_{args.n_grid}"
    meta_path = metadata_root / f"{args.n_grid}_{args.n_grid}.json"
    coco_train_dir = Path(args.coco_train_dir).resolve()

    if not stitched_dir.exists():
        raise FileNotFoundError(f"Missing stitched dir: {stitched_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata json: {meta_path}")

    stitched_paths = load_stitched_images(stitched_dir)
    if len(stitched_paths) == 0:
        raise ValueError(f"No stitched images found in {stitched_dir}")
    if len(stitched_paths) < args.sequence_length:
        raise ValueError(f"Need at least {args.sequence_length} stitched images, found {len(stitched_paths)}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta_data = json.load(f)

    coco_paths = sorted([p for p in coco_train_dir.iterdir() if p.is_file()])
    if not coco_paths:
        raise ValueError(f"No images found in {coco_train_dir}")

    annotations = []
    n_pos = int(args.num_sequences * args.positive_ratio)
    n_cells = args.n_grid * args.n_grid

    for i in tqdm(range(args.num_sequences), desc=f"Sampling annotations N={args.n_grid}, L={args.sequence_length}"):
        sampled_stitched = random.sample(stitched_paths, args.sequence_length)
        sampled_stitched_str = [str(p) for p in sampled_stitched]

        if i < n_pos:
            j = random.randint(0, args.sequence_length * n_cells - 1)
            idx, loc = divmod(j, n_cells)
            row, col = divmod(loc, args.n_grid)
            stitched_name = sampled_stitched[idx].name
            target = meta_data[stitched_name][f"{row}_{col}"]
        else:
            idx = -1
            row = -1
            col = -1
            exclude = []
            for p in sampled_stitched:
                exclude += flatten_meta_values(meta_data[p.name])
            exclude_set = set(exclude)
            candidates = [str(p) for p in coco_paths if str(p) not in exclude_set]
            target = random.choice(candidates) if candidates else str(random.choice(coco_paths))

        annotations.append(
            {
                "id": i,
                "image_ids": sampled_stitched_str,
                "index": idx,
                "row": row,
                "col": col,
                "target": target,
            }
        )

    if args.output_json:
        out_path = Path(args.output_json).resolve()
    else:
        out_path = metadata_root / f"annotations_{args.sequence_length}_{args.n_grid}_{args.n_grid}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print(f"Saved annotations: {out_path}")
    print(f"Total sequences: {len(annotations)} (positives={n_pos}, negatives={len(annotations)-n_pos})")


if __name__ == "__main__":
    main()

