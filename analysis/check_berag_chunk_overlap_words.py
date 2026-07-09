#!/usr/bin/env python3
"""
Sanity-check BERAG/RULER-side passage chunking overlap (word-based).

It imports `_chunk_context_to_passages` from `src/simple_niah_esf_provider.py`
and verifies that adjacent passages share the expected overlapping region
(`overlap_words`) at the boundary.

Usage examples:
  python analysis/check_berag_chunk_overlap_words.py --context-file /path/to/context.txt
  python analysis/check_berag_chunk_overlap_words.py --context "w1 w2 w3 ..." --chunk-size 16 --num-chunks 4 --overlap-words 4
"""

import argparse
from pathlib import Path
import sys


def _import_chunker():
    # analysis/ -> repo root, then repo root/src
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))
    from simple_niah_esf_provider import _chunk_context_to_passages

    return _chunk_context_to_passages


def _check_pair(p1: str, p2: str, overlap_words: int) -> bool:
    w1 = (p1 or "").split()
    w2 = (p2 or "").split()
    if not w1 or not w2 or overlap_words <= 0:
        return True
    k = min(overlap_words, len(w1), len(w2))
    return w1[-k:] == w2[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, default=None, help="Context string.")
    parser.add_argument("--context-file", type=str, default=None, help="Path to file containing context.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Passage size in words.")
    parser.add_argument("--num-chunks", type=int, default=None, help="Derive chunk size from total word count.")
    parser.add_argument("--overlap-words", type=int, default=100, help="Expected overlap in words.")
    parser.add_argument("--show", type=int, default=3, help="How many adjacent pairs to print.")
    args = parser.parse_args()

    if not args.context and not args.context_file:
        raise ValueError("Provide --context or --context-file")

    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    else:
        context = args.context

    chunker = _import_chunker()
    passages = chunker(
        context,
        tokenizer=None,  # tokenizer-free (word-based) in this chunker
        chunk_size=args.chunk_size,
        num_of_chunks=args.num_chunks,
        overlap_words=args.overlap_words,
    )

    if not passages:
        print("No passages produced.")
        return 1

    ok_all = True
    print(f"Produced {len(passages)} passages.")
    for i in range(min(len(passages) - 1, args.show)):
        p1 = passages[i]
        p2 = passages[i + 1]
        ok = _check_pair(p1, p2, args.overlap_words)
        ok_all = ok_all and ok

        if args.show > 0:
            w1 = p1.split()
            w2 = p2.split()
            k = min(args.overlap_words, len(w1), len(w2), max(1, args.overlap_words))
            suffix = " ".join(w1[-k:]) if w1 else ""
            prefix = " ".join(w2[:k]) if w2 else ""

            print(f"[pair {i}->{i+1}] overlap_match={ok}")
            print(f"  suffix_last_{k}: {suffix[:120]}{'...' if len(suffix) > 120 else ''}")
            print(f"  prefix_first_{k}: {prefix[:120]}{'...' if len(prefix) > 120 else ''}")

    print(f"Overall overlap check passed: {ok_all}")
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())

