#!/usr/bin/env python3
"""
Curate EVQA examples with gold passage in PreFLMR-L top-20, then build five
position-controlled variants by swapping the GT passage into rank bins.

Ground-truth policy: use only ``pos_item_ids[0]`` as the gold passage id (ignore
additional ids if present). Compare to ``passage_id`` in retrieved passages after
normalizing both to strings.

Default source (override with ``--source``):
  .../outputs/0jingbiao_mei/EVQA-testfull-with-retrieval

Outputs under ``--out_root`` (default ``analysis/EVQA-gtdoc-position-datasets/``):
  One ``save_to_disk`` folder per variant + ``manifest.json``.

How to run (after activating your env, e.g. ``scripts/hpc_activate_env_py310_infer.sh``):

  sbatch scripts/analysis/slurm_curate_evqa_gtdoc_position.sh

Sanity checks (one-liners after curation):

  python -c "from datasets import load_from_disk; d=load_from_disk('analysis/EVQA-gtdoc-position-datasets/EVQA-256-gtdoc_at_1-4'); print(len(d))"
  # expect 256 (or lower if the pool was smaller than --n_sample)

  python - <<'PY'
  from datasets import load_from_disk
  def rank(ex):
      gt = str(ex["pos_item_ids"][0])
      for i, p in enumerate(ex["retrieved_passage"][:20]):
          if str(p["passage_id"]) == gt: return i + 1
      return None
  d = load_from_disk("analysis/EVQA-gtdoc-position-datasets/EVQA-256-gtdoc_at_1-4")
  assert all(1 <= rank(ex) <= 4 for ex in d)
  print("ok")
  PY
"""

import argparse
import json
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_from_disk

DEFAULT_REPO = "/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA"
DEFAULT_SOURCE = (
    f"{DEFAULT_REPO}/outputs/0jingbiao_mei/EVQA-testfull-with-retrieval"
)

BIN_SPECS: List[Tuple[str, int, int]] = [
    ("gtdoc_at_1-4", 1, 4),
    ("gtdoc_at_5-8", 5, 8),
    ("gtdoc_at_9-12", 9, 12),
    ("gtdoc_at_13-16", 13, 16),
    ("gtdoc_at_17-20", 17, 20),
]


def _norm_pid(x: Any) -> str:
    return str(x)


def _gt_passage_id(example: Dict[str, Any]) -> str:
    ids = example["pos_item_ids"]
    if ids is None or len(ids) == 0:
        raise ValueError("pos_item_ids empty")
    return _norm_pid(ids[0])


def gt_rank_in_top20(example: Dict[str, Any]) -> Optional[int]:
    """1-based rank in first 20 passages, or None if missing / wrong length."""
    passages = example.get("retrieved_passage")
    if not passages or len(passages) < 20:
        return None
    top20 = passages[:20]
    gt = _gt_passage_id(example)
    for i, p in enumerate(top20):
        if _norm_pid(p.get("passage_id")) == gt:
            return i + 1
    return None


def _swap_top20_prefix(
    retrieved: List[Dict[str, Any]],
    orig_rank_1based: int,
    target_rank_1based: int,
) -> List[Dict[str, Any]]:
    out = deepcopy(retrieved)
    top = out[:20]
    a, b = orig_rank_1based - 1, target_rank_1based - 1
    top[a], top[b] = top[b], top[a]
    out[:20] = top
    return out


def _build_variant_rows(
    examples: List[Dict[str, Any]],
    L: int,
    R: int,
    seed: int,
    bin_tag: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (mapped rows, audit rows)."""
    rng_base = seed + sum(ord(c) for c in bin_tag) * 10007
    rows: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        orig_r = gt_rank_in_top20(ex)
        if orig_r is None:
            raise RuntimeError("pool row must have GT in top-20")
        rng = random.Random(rng_base + idx)
        ex2 = deepcopy(ex)
        if L <= orig_r <= R:
            new_passages = deepcopy(ex["retrieved_passage"])
            final_r = orig_r
        else:
            i = rng.randint(L, R)
            new_passages = _swap_top20_prefix(ex["retrieved_passage"], orig_r, i)
            final_r = i
        ex2["retrieved_passage"] = new_passages
        rows.append(ex2)
        audits.append(
            {
                "original_gt_rank": orig_r,
                "final_gt_rank": final_r,
            }
        )
    return rows, audits


def _assert_bin(rows: List[Dict[str, Any]], L: int, R: int) -> None:
    for ex in rows:
        r = gt_rank_in_top20(ex)
        assert r is not None and L <= r <= R, (r, L, R)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate EVQA top-20 GT position variants (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Directory for save_to_disk outputs (default: <repo>/analysis/EVQA-gtdoc-position-datasets)",
    )
    parser.add_argument("--repo_root", type=str, default=DEFAULT_REPO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_sample", type=int, default=256)
    args = parser.parse_args()

    repo = os.path.abspath(args.repo_root)
    source = os.path.abspath(args.source)
    if args.out_root is None:
        out_root = os.path.join(repo, "analysis", "EVQA-gtdoc-position-datasets")
    else:
        out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    print(f"Loading dataset from {source}")
    ds = load_from_disk(source)
    n_total = len(ds)

    def pool_ok(ex: Dict[str, Any]) -> bool:
        return gt_rank_in_top20(ex) is not None

    pool_indices = [i for i in range(n_total) if pool_ok(ds[i])]
    n_pool = len(pool_indices)
    frac = n_pool / n_total if n_total else 0.0
    print(f"Total rows: {n_total}; GT in top-20 (len>=20): {n_pool} ({frac:.4%})")

    rng = random.Random(args.seed)
    rng.shuffle(pool_indices)
    take = min(args.n_sample, len(pool_indices))
    if take < args.n_sample:
        print(
            f"WARNING: only {take} examples available (requested {args.n_sample})."
        )
    chosen = pool_indices[:take]
    base_rows = [deepcopy(ds[i]) for i in chosen]

    for tag, L, R in BIN_SPECS:
        rows, audits = _build_variant_rows(base_rows, L, R, args.seed, tag)
        _assert_bin(rows, L, R)
        out_ds = Dataset.from_list(rows)
        subdir = os.path.join(out_root, f"EVQA-{take}-{tag}")
        print(f"Saving {len(out_ds)} rows -> {subdir}")
        out_ds.save_to_disk(subdir)
        manifest: Dict[str, Any] = {
            "seed": args.seed,
            "source": source,
            "repo_root": repo,
            "n_total_source": n_total,
            "n_with_gt_in_top20": n_pool,
            "fraction_gt_in_top20": frac,
            "n_sample_requested": args.n_sample,
            "n_sample_actual": take,
            "variant": tag,
            "bin_inclusive": [L, R],
            "per_row_audit": audits,
        }
        with open(os.path.join(subdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Assert OK: all GT ranks in [{L}, {R}]")

    print("Done.")


if __name__ == "__main__":
    main()
