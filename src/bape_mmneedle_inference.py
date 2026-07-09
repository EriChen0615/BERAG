import sys
sys.path.append("./src")

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from bape_inference_engine import BAPEInferenceEngine
from hf_backend import HFQwen2VLBackend

_SINGLE_PATTERN = re.compile(r"^annotations_(?P<seq>\d+)_(?P<rows>\d+)_(?P<cols>\d+)\.json$")
_MULTI_PATTERN = re.compile(r"^(?P<needles>\d+)_annotations_(?P<seq>\d+)_(?P<rows>\d+)_(?P<cols>\d+)\.json$")


MMNEEDLE_PROMPT = (
    "<<<EVIDENCE>>>\n"
    "Given M images indexed from 1 to M, each divided into NxN sub-images, identify the sub-image that best matches the provided caption. "
    "Respond with \"index, row, column\" and nothing else. For example, \"1, 2, 3\" indicates the sub-image in the first image, "
    "second row, and third column. If no match is found, respond only with \"-1\".\n"
    "Caption: <<<QUESTION>>>"
)


def _get_prompt_template(path: Optional[str]) -> str:
    if path is None or path == "":
        return MMNEEDLE_PROMPT
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_prediction(text: str) -> Tuple[int, int, int]:
    if text is None:
        return -1, -1, -1
    cleaned = str(text).strip().replace("\n", " ")
    cleaned = cleaned.strip(". ").strip()

    if cleaned == "-1":
        return -1, -1, -1

    # Try exact "a, b, c" first.
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) >= 3:
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except Exception:
            pass

    # Fallback: first 3 integers anywhere in response.
    nums = re.findall(r"-?\d+", cleaned)
    if len(nums) >= 3:
        return int(nums[0]), int(nums[1]), int(nums[2])
    if len(nums) >= 1 and int(nums[0]) == -1:
        return -1, -1, -1
    return -1, -1, -1


def _needle_to_gt(needle_locations: List[Dict]) -> Tuple[int, int, int]:
    """
    Convert first needle location to 1-based (index,row,col), or -1 triplet if absent.
    """
    if not needle_locations:
        return -1, -1, -1
    loc = needle_locations[0]
    idx0 = int(loc.get("image_index", -1))
    row0 = int(loc.get("row", -1))
    col0 = int(loc.get("col", -1))
    if idx0 < 0:
        return -1, -1, -1
    return idx0 + 1, row0 + 1, col0 + 1


def _build_passages(haystack_images: List, sequence_length: int, add_z0: bool = False) -> List[Dict]:
    passages = []
    usable = haystack_images[:sequence_length]
    for i, img in enumerate(usable):
        passages.append(
            {
                "images": [img],
                # In inference, avoid literal "<image>" in text because images are
                # already provided through structured multimodal inputs.
                "text": f"[IMAGE_INDEX] {i + 1}",
            }
        )
    if add_z0:
        # Match training z0 format: no image and no <image> token prefix.
        passages.append({"images": [], "text": "[IMAGE_INDEX] -1"})
    return passages


def _compute_metrics(df: pd.DataFrame) -> Dict:
    valid_pred = df["pred_index"].notna()
    has_gt = df["gt_index"] != -1
    pred_has = df["pred_index"] != -1

    existence_match = ((~has_gt) & (~pred_has)) | (has_gt & pred_has)
    index_match = has_gt & (df["pred_index"] == df["gt_index"])
    exact_match_pos = has_gt & (df["pred_index"] == df["gt_index"]) & (df["pred_row"] == df["gt_row"]) & (df["pred_col"] == df["gt_col"])
    exact_match_all = exact_match_pos | ((~has_gt) & (~pred_has))

    df["existence_match"] = existence_match.astype(int)
    df["index_match"] = index_match.astype(int)
    # Keep exact_match for backward compatibility; it refers to all-example exact metric.
    df["exact_match"] = exact_match_all.astype(int)
    df["exact_match_pos"] = exact_match_pos.astype(int)
    df["has_gt"] = has_gt.astype(int)

    # Index and exact_pos are meaningful only on GT-present examples.
    index_denom = max(int(has_gt.sum()), 1)
    scores = {
        "existence_accuracy": float(df["existence_match"].mean()) if len(df) > 0 else 0.0,
        "index_accuracy": float(df.loc[has_gt, "index_match"].sum() / index_denom),
        "exact_accuracy_pos": float(df.loc[has_gt, "exact_match_pos"].sum() / index_denom),
        "exact_accuracy_all": float(df["exact_match"].mean()) if len(df) > 0 else 0.0,
        # Backward-compatible alias.
        "exact_accuracy": float(df["exact_match"].mean()) if len(df) > 0 else 0.0,
        "num_examples": int(len(df)),
        "num_gt_present": int(has_gt.sum()),
        "num_gt_absent": int((~has_gt).sum()),
    }
    return scores


def _parse_metadata_name(fname: str) -> Optional[Dict[str, int]]:
    match = _SINGLE_PATTERN.match(fname)
    if match:
        return {
            "needles": 1,
            "seq": int(match.group("seq")),
            "rows": int(match.group("rows")),
            "cols": int(match.group("cols")),
        }
    match = _MULTI_PATTERN.match(fname)
    if match:
        return {
            "needles": int(match.group("needles")),
            "seq": int(match.group("seq")),
            "rows": int(match.group("rows")),
            "cols": int(match.group("cols")),
        }
    return None


def _load_local_mmneedle_dataset(data_root: str, caption_path: Optional[str]):
    metadata_root = os.path.join(data_root, "metadata_stitched")
    if not os.path.isdir(metadata_root):
        raise FileNotFoundError(f"Missing local metadata directory: {metadata_root}")

    # Support either:
    # - local_root/images_stitched/** (current curated layout), or
    # - local_root/train/images_stitched/** passed accidentally.
    direct_images_root = os.path.join(data_root, "images_stitched")
    if os.path.isdir(direct_images_root):
        images_root = direct_images_root
    else:
        fallback_images_root = os.path.join(os.path.dirname(data_root), "images_stitched")
        if not os.path.isdir(fallback_images_root):
            raise FileNotFoundError(
                f"Cannot find images_stitched under {data_root} or {os.path.dirname(data_root)}"
            )
        images_root = fallback_images_root

    captions = {}
    if caption_path:
        if not os.path.isfile(caption_path):
            raise FileNotFoundError(f"caption file not found: {caption_path}")
        with open(caption_path, "r", encoding="utf-8") as f:
            captions = json.load(f)

    metadata_files = sorted([x for x in os.listdir(metadata_root) if x.endswith(".json")])
    all_records = []

    def _resolve_image_path(p: str) -> str:
        p = str(p)
        if os.path.isabs(p):
            return p
        # Official MMNeedle metadata stores relative paths like:
        # "images_stitched/1_1/COCO_val2014_stitched_5808.jpg"
        cand = os.path.join(data_root, p)
        if os.path.isfile(cand):
            return cand
        cand2 = os.path.join(images_root, p)
        if os.path.isfile(cand2):
            return cand2
        # Keep best-effort absolute path for clearer downstream errors.
        return cand

    for fname in metadata_files:
        spec = _parse_metadata_name(fname)
        if spec is None:
            continue
        fpath = os.path.join(metadata_root, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            target_field = entry.get("target", [])
            if isinstance(target_field, str):
                target_list = [target_field]
            else:
                target_list = list(target_field)

            idx_field = entry.get("index", [])
            if isinstance(idx_field, int):
                index_list = [idx_field]
            else:
                index_list = list(idx_field)

            row_field = entry.get("row", [])
            if isinstance(row_field, int):
                row_list = [row_field]
            else:
                row_list = list(row_field)

            col_field = entry.get("col", [])
            if isinstance(col_field, int):
                col_list = [col_field]
            else:
                col_list = list(col_field)

            needle_locations = []
            has_needle = False
            for idx, row, col in zip(index_list, row_list, col_list):
                idx = int(idx)
                has_needle = has_needle or idx != -1
                needle_locations.append(
                    {
                        "image_index": idx,
                        "row": int(row),
                        "col": int(col),
                    }
                )

            needle_captions = []
            for tgt in target_list:
                base = os.path.basename(str(tgt))
                needle_captions.append(captions.get(base, ""))

            record = {
                "id": f"{spec['needles']}n_{spec['seq']}seq_{spec['rows']}x{spec['cols']}_{entry.get('id', 0)}",
                "sequence_length": len(entry.get("image_ids", [])),
                "grid_rows": spec["rows"],
                "grid_cols": spec["cols"],
                "needles_per_query": spec["needles"],
                "haystack_images": [_resolve_image_path(x) for x in list(entry.get("image_ids", []))],
                "needle_locations": needle_locations,
                "needle_image_ids": target_list,
                "needle_captions": needle_captions,
                "has_needle": bool(has_needle),
            }
            all_records.append(record)
    return all_records


def _filter_mmneedle(ds, n_grid: int, haystack_m: int, needles_per_query: int, take_n: int, offset: int, seed: int):
    if isinstance(ds, list):
        ds = [
            x
            for x in ds
            if x["grid_rows"] == n_grid
            and x["grid_cols"] == n_grid
            and x["sequence_length"] == haystack_m
            and x["needles_per_query"] == needles_per_query
        ]
        if take_n > 0:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(ds)).tolist()
            ds = [ds[i] for i in perm]
            end = min(offset + take_n, len(ds))
            ds = ds[offset:end]
        elif offset > 0:
            ds = ds[offset:]
        return ds

    ds = ds.filter(
        lambda x: x["grid_rows"] == n_grid
        and x["grid_cols"] == n_grid
        and x["sequence_length"] == haystack_m
        and x["needles_per_query"] == needles_per_query
    )
    if take_n > 0:
        ds = ds.shuffle(seed=seed)
        end = min(offset + take_n, len(ds))
        ds = ds.select(range(offset, end))
    elif offset > 0:
        ds = ds.select(range(offset, len(ds)))
    return ds


def run_inference(args) -> pd.DataFrame:
    prompt_template = _get_prompt_template(args.prompt_template)

    local_loader_requested = (
        args.force_local_mmneedle
        or args.hf_dataset_path.endswith(".py")
        or os.path.isdir(args.hf_dataset_path)
    )
    if local_loader_requested:
        local_root = args.mmneedle_data_root if args.mmneedle_data_root else args.hf_dataset_path
        if local_root.endswith(".py"):
            # Default to the official benchmark data directory (COCO val-based),
            # not the training curation directory.
            local_root = args.mmneedle_data_root if args.mmneedle_data_root else args.official_mmneedle_data_root
        records = _load_local_mmneedle_dataset(local_root, args.mmneedle_caption_path)
        dataset = records
    else:
        dataset = load_dataset(args.hf_dataset_path, split=args.split)
    dataset = _filter_mmneedle(
        dataset,
        n_grid=args.n_grid,
        haystack_m=args.haystack_m,
        needles_per_query=args.needles_per_query,
        take_n=args.take_n,
        offset=args.offset,
        seed=args.seed,
    )
    print(f"Loaded {len(dataset)} MMNeedle examples after filtering.")

    backend = HFQwen2VLBackend(
        model_path=args.model_path,
        processor_path=args.processor_path if args.processor_path else args.model_path,
        adapter_name_or_path=args.adapter_name_or_path,
        max_batch_size_per_forward=args.max_batch_size_per_forward,
    )
    prior_head_config = {
        "modeling": args.prior_head_modeling,
        "num_layers": args.prior_head_num_layers,
        "proj_dim": args.prior_head_proj_dim,
    }
    engine = BAPEInferenceEngine(
        backend,
        prior_head_path=args.prior_head_path,
        prior_head_config=prior_head_config,
        dynamic_k_top_p=args.dynamic_k_top_p,
        hidden_state_offset=args.hidden_state_offset,
        num_beams=args.num_beams,
    )

    all_rows = []
    for item in tqdm(dataset, total=len(dataset), desc="MMNeedle BAPE inference"):
        caption = ""
        if item.get("needle_captions"):
            caption = str(item["needle_captions"][0])
        prompt_text = prompt_template.replace("<<<QUESTION>>>", caption)
        x = {"text": prompt_text, "image": None}

        passages = _build_passages(item["haystack_images"], args.haystack_m, add_z0=args.add_z0)
        if len(passages) == 0:
            continue

        if args.passage_prior != "prior_head":
            raise NotImplementedError("Only passage_prior=prior_head is supported for MMNeedle BAPE.")

        generated_token_ids, log_all_tokens_llk, posterior_max_idx, prior_max_idx, prior_logits = engine.generate(
            x, passages, log_passage_prior=None, max_new_tokens=args.max_new_tokens
        )
        response = engine.backend.processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        pred_index, pred_row, pred_col = _parse_prediction(response)
        gt_index, gt_row, gt_col = _needle_to_gt(item.get("needle_locations", []))

        all_rows.append(
            {
                "id": item.get("id"),
                "sequence_length": int(item.get("sequence_length", -1)),
                "grid_rows": int(item.get("grid_rows", -1)),
                "grid_cols": int(item.get("grid_cols", -1)),
                "needles_per_query": int(item.get("needles_per_query", -1)),
                "caption": caption,
                "prompt_text": prompt_text,
                "response": response,
                "pred_index": pred_index,
                "pred_row": pred_row,
                "pred_col": pred_col,
                "gt_index": gt_index,
                "gt_row": gt_row,
                "gt_col": gt_col,
                "posterior_max_idx": int(posterior_max_idx) if posterior_max_idx is not None else -1,
                "prior_max_idx": int(prior_max_idx) if prior_max_idx is not None else -1,
                "log_all_tokens_llk": log_all_tokens_llk,
                "prior_logits": prior_logits if prior_logits is not None else [],
                "passages": passages,
            }
        )

    del engine
    torch.cuda.empty_cache()
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_path", type=str, default="third_party/mmneedle/huggingface/mmneedle.py")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mmneedle_data_root", type=str, default=None)
    parser.add_argument(
        "--mmneedle_caption_path",
        type=str,
        default="/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/MMNeedle/data/file_to_caption.json",
    )
    parser.add_argument("--force_local_mmneedle", action="store_true")
    # Prefer official benchmark data root over train curation root.
    parser.add_argument(
        "--official_mmneedle_data_root",
        type=str,
        default="/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/MMNeedle/data",
        help="Official MMNeedle benchmark root (contains metadata_stitched and images_stitched).",
    )
    parser.add_argument("--n_grid", type=int, default=1)
    parser.add_argument("--haystack_m", type=int, default=10)
    parser.add_argument("--needles_per_query", type=int, default=1)
    parser.add_argument("--add_z0", action="store_true", help="Append z0 null passage: [IMAGE_INDEX] -1 with no image.")
    parser.add_argument("--take_n", type=int, default=-1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_template", type=str, default=None)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--passage_prior", type=str, default="prior_head", choices=["prior_head"])
    parser.add_argument("--prior_head_path", type=str, default=None)
    parser.add_argument("--prior_head_modeling", type=str, default="mlp_head", choices=["mlp_head", "linear_head"])
    parser.add_argument("--prior_head_num_layers", type=int, default=2)
    parser.add_argument("--prior_head_proj_dim", type=int, default=1024)
    parser.add_argument("--dynamic_k_top_p", type=float, default=None)
    parser.add_argument("--max_batch_size_per_forward", type=int, default=5)
    parser.add_argument("--hidden_state_offset", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--exp_name", type=str, required=True)

    args = parser.parse_args()
    os.makedirs(args.exp_name, exist_ok=True)

    inference_path = os.path.join(args.exp_name, "inference_results.csv")
    marked_path = os.path.join(args.exp_name, "marked_inference_results.csv")
    score_path = os.path.join(args.exp_name, "scores.json")

    if args.use_cache and os.path.exists(inference_path):
        df = pd.read_csv(inference_path)
        print(f"Loaded cache: {inference_path}")
    else:
        df = run_inference(args)
        df.to_csv(inference_path, index=False)
        print(f"Saved inference results: {inference_path}")

    if args.do_eval:
        scores = _compute_metrics(df)
        df.to_csv(marked_path, index=False)
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        print("Evaluation results:")
        print(json.dumps(scores, indent=2))
        print(f"Saved marked results: {marked_path}")
        print(f"Saved scores: {score_path}")


if __name__ == "__main__":
    main()

