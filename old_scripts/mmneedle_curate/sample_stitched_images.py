#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm


def list_images(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def build_stitched(paths: List[Path], n_grid: int, subimage_res: int) -> Image.Image:
    stitched = Image.new("RGB", (n_grid * subimage_res, n_grid * subimage_res))
    idx = 0
    for r in range(n_grid):
        for c in range(n_grid):
            img = Image.open(paths[idx]).convert("RGB").resize((subimage_res, subimage_res))
            stitched.paste(img, (c * subimage_res, r * subimage_res))
            idx += 1
    return stitched


def main():
    parser = argparse.ArgumentParser(description="Create MMNeedle-style stitched images from COCO train2014.")
    parser.add_argument("--coco_train_dir", type=str, required=True, help="Path to COCO train2014 image directory.")
    parser.add_argument("--n_grid", type=int, choices=[1, 2, 4, 8], required=True, help="Grid size N for NxN stitched image.")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of stitched images to generate.")
    parser.add_argument("--subimage_res", type=int, default=256, help="Resolution per sub-image.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--images_root", type=str, default="../vqa_data/MMNeedle/train/images_stitched")
    parser.add_argument("--metadata_root", type=str, default="../vqa_data/MMNeedle/train/metadata_stitched")
    args = parser.parse_args()

    random.seed(args.seed)

    coco_dir = Path(args.coco_train_dir).resolve()
    images_root = Path(args.images_root).resolve()
    metadata_root = Path(args.metadata_root).resolve()

    all_images = list_images(coco_dir)
    need_per_stitched = args.n_grid * args.n_grid
    if len(all_images) < need_per_stitched:
        raise ValueError(f"Need at least {need_per_stitched} images, found {len(all_images)} in {coco_dir}")

    out_dir = images_root / f"{args.n_grid}_{args.n_grid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_root / f"{args.n_grid}_{args.n_grid}.json"

    metadata = {}
    for i in tqdm(range(args.num_images), desc=f"Stitching {args.n_grid}x{args.n_grid} images"):
        sampled = random.sample(all_images, need_per_stitched)  # unique sub-images within one stitched image
        stitched = build_stitched(sampled, args.n_grid, args.subimage_res)

        filename = f"COCO_train2014_stitched_{args.n_grid}x{args.n_grid}_{i:06d}.jpg"
        stitched.save(out_dir / filename, "JPEG")

        entry = {}
        idx = 0
        for r in range(args.n_grid):
            for c in range(args.n_grid):
                entry[f"{r}_{c}"] = str(sampled[idx])
                idx += 1
        metadata[filename] = entry

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved stitched images: {out_dir}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Generated {args.num_images} stitched images with N={args.n_grid}.")


if __name__ == "__main__":
    main()

