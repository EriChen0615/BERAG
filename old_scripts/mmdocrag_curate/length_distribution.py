#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict
import math
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def load_records(path: Path) -> Tuple[List[dict], str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], "json"
    if text[0] == "[":
        return json.loads(text), "json"
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return records, "jsonl"


def save_records(path: Path, records: List[dict], fmt: str) -> None:
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    try:
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception:
        return None


def get_processor(name: str):
    try:
        from transformers import AutoProcessor
    except Exception:
        return None
    try:
        return AutoProcessor.from_pretrained(name, trust_remote_code=True)
    except Exception:
        return None


def normalize_image_tokens(text: str) -> str:
    if "<image>" not in text:
        return text
    parts = text.split("<image>")
    # Keep a single <image> token to represent the largest image.
    return "<image>".join([parts[0], ""] + parts[1:]).replace("<image><image>", "<image>")


def count_tokens_text(text: str, tokenizer) -> int:
    text = normalize_image_tokens(text)
    if tokenizer is None:
        # Fallback: whitespace tokens + count explicit <image> markers
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def iter_contents(messages: Iterable[dict], role_filter: str) -> Iterable[str]:
    for msg in messages:
        if role_filter and msg.get("role") != role_filter:
            continue
        yield msg.get("content", "")


def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _resize_to_pixel_constraints(width: int, height: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    pixels = width * height
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / float(pixels))
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / float(pixels))
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
    return width, height


def image_token_len(
    image_path: str,
    processor,
    cache: Dict[str, int],
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    use_processor: bool,
) -> int:
    if image_path in cache:
        return cache[image_path]
    image = _load_image(image_path)
    if use_processor and processor is not None:
        mm_inputs = processor(images=[image], return_tensors="pt")
        image_grid_thw = mm_inputs.get("image_grid_thw")
        if image_grid_thw is None or len(image_grid_thw) == 0:
            cache[image_path] = 0
            return 0
        grid = image_grid_thw[0]
        merge_length = int(merge_size) ** 2
        tokens = int(grid.prod().item()) // merge_length
        cache[image_path] = tokens
        return tokens

    width, height = image.size
    width, height = _resize_to_pixel_constraints(width, height, min_pixels, max_pixels)
    grid_h = max(1, height // patch_size)
    grid_w = max(1, width // patch_size)
    merge_length = int(merge_size) ** 2
    tokens = (grid_h * grid_w) // merge_length
    tokens = max(1, tokens)
    cache[image_path] = tokens
    return tokens


def summarize(lengths: List[int]) -> str:
    if not lengths:
        return "No samples found."
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    def pct(p):
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return lengths_sorted[idx]
    avg = sum(lengths_sorted) / n
    return (
        f"count={n} min={lengths_sorted[0]} max={lengths_sorted[-1]} "
        f"mean={avg:.2f} p50={pct(0.50)} p90={pct(0.90)} p95={pct(0.95)} p99={pct(0.99)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compute length distribution for ShareGPT-style data.")
    parser.add_argument("--input", required=True, help="Path to train_sharegpt.json/jsonl")
    parser.add_argument("--output", help="Optional output path for filtered records")
    parser.add_argument("--max-len", type=int, default=2048, help="Filter samples exceeding this length")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-VL-3B-Instruct", help="HF tokenizer name")
    parser.add_argument("--processor", default="Qwen/Qwen2.5-VL-3B-Instruct", help="HF processor name")
    parser.add_argument("--role", default="", help="Only count tokens for messages with this role")
    parser.add_argument("--use-processor", action="store_true", help="Use HF processor for exact image_grid_thw")
    parser.add_argument("--patch-size", type=int, default=14, help="Patch size for image token estimate")
    parser.add_argument("--merge-size", type=int, default=2, help="Merge size for image token estimate")
    parser.add_argument("--min-pixels", type=int, default=32 * 32, help="Min pixels for resizing")
    parser.add_argument("--max-pixels", type=int, default=768 * 768, help="Max pixels for resizing")
    args = parser.parse_args()

    records, fmt = load_records(Path(args.input))
    tokenizer = get_tokenizer(args.tokenizer)
    processor = get_processor(args.processor)
    if tokenizer is None:
        print("Warning: transformers tokenizer not available; using whitespace tokenization.")
    if processor is None and args.use_processor:
        print("Warning: transformers processor not available; falling back to patch-size estimate.")

    image_token_cache: Dict[str, int] = {}

    lengths = []
    kept = []
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Computing lengths", unit="ex")
    for rec in iterator:
        messages = rec.get("messages", [])
        passages = rec.get("passages", []) or []
        top_images = rec.get("images", []) or []

        def example_length_for_passage(passage: Optional[dict]) -> int:
            prompt = messages[:-1]
            response = messages[-1:]
            content_list = [m.get("content", "") for m in prompt]
            passage_text = ""
            passage_images = []
            if passage is not None and isinstance(passage, dict):
                passage_text = passage.get("text", "") or ""
                passage_images = passage.get("images", []) or []
                if passage_images:
                    passage_text = (" ".join(["<image>"] * len(passage_images)) + " " + passage_text).strip()
            if content_list:
                content_list[-1] = content_list[-1].replace("<<<EVIDENCE>>>", passage_text)
            full_text = "".join(content_list + [m.get("content", "") for m in response])

            all_images = list(top_images) + list(passage_images)
            if all_images and "<image>" not in full_text:
                full_text = "<image> " + full_text

            text_tokens = count_tokens_text(full_text, tokenizer)
            max_image_tokens = 0
            if processor is not None and all_images:
                for img_path in all_images:
                    try:
                        max_image_tokens = max(
                            max_image_tokens,
                            image_token_len(
                                img_path,
                                processor,
                                image_token_cache,
                                patch_size=args.patch_size,
                                merge_size=args.merge_size,
                                min_pixels=args.min_pixels,
                                max_pixels=args.max_pixels,
                                use_processor=args.use_processor,
                            ),
                        )
                    except Exception:
                        continue
            return text_tokens + max_image_tokens

        if passages:
            length = max(example_length_for_passage(p) for p in passages)
        else:
            length = example_length_for_passage(None)

        lengths.append(length)
        if length <= args.max_len:
            kept.append(rec)

    print(summarize(lengths))
    over = len(records) - len(kept)
    print(f"over_max_len={over} max_len={args.max_len}")

    if args.output:
        save_records(Path(args.output), kept, fmt)
        print(f"saved_filtered={args.output}")


if __name__ == "__main__":
    main()
