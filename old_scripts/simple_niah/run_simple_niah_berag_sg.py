#!/usr/bin/env python3
"""
Run simple_niah with BERAG-SG inference engine (segment-level beam search,
marginalized chunk scoring, optional composite-state rounds).

Uses the NIAH tester from third_party/NeedleInAHaystack with
SimpleNiahBERAGSGProvider (BERAG engine in BERAG-SG mode + HF backend).

Example (from repo root):
  PYTHONPATH=".:third_party/NeedleInAHaystack" python scripts/simple_niah/run_simple_niah_berag_sg.py \\
    --model_path /path/to/Qwen2.5-VL-7B-Instruct \\
    --needle "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day." \\
    --retrieval_question "What is the best thing to do in San Francisco?"
"""

import argparse
import json
import os
import re
import sys


def _build_decode_suffix(args) -> str:
    """Build a short, filesystem-safe suffix from BERAG-SG parameters so different configs write to different dirs."""
    parts = []
    parts.append(f"s{getattr(args, 'segment_length', 4)}")
    parts.append(f"b{getattr(args, 'berag_sg_beam_size', 2)}")
    parts.append(f"m{getattr(args, 'max_composite_size', 2)}")
    top_p = getattr(args, "berag_sg_top_p", 0.9)
    if top_p > 0:
        parts.append(f"p{top_p}")
    else:
        parts.append("p0")
    if getattr(args, "num_of_chunks", None) is not None:
        parts.append(f"n{args.num_of_chunks}")
    else:
        parts.append(f"c{getattr(args, 'chunk_size', 512)}")
    temp = getattr(args, "berag_sg_temperature", 0.5)
    parts.append(f"t{temp}")
    prune = getattr(args, "berag_sg_beam_prune", "diverse_beam_search")
    parts.append("diverse" if prune == "diverse_beam_search" else "topb")
    suffix = "_".join(str(p) for p in parts)
    suffix = re.sub(r"[^a-zA-Z0-9_.-]", "_", suffix)
    return suffix or "default"


def _setup_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = scripts/simple_niah -> repo_root = parent of scripts
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    niah_root = os.path.join(repo_root, "third_party", "NeedleInAHaystack")
    if niah_root not in sys.path:
        sys.path.insert(0, niah_root)


def main():
    parser = argparse.ArgumentParser(
        description="Run simple_niah with BERAG-SG inference engine (segment beam + composite states)."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model.")
    parser.add_argument("--processor_path", type=str, default=None, help="Defaults to model_path.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional LoRA adapter.")
    parser.add_argument(
        "--needle",
        type=str,
        default="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        help="Needle text to hide in the haystack.",
    )
    parser.add_argument(
        "--retrieval_question",
        type=str,
        default="What is the best thing to do in San Francisco?",
        help="Question to ask the model.",
    )
    parser.add_argument(
        "--haystack_dir",
        type=str,
        default="PaulGrahamEssays",
        help="Directory name under needlehaystack/.",
    )
    parser.add_argument("--context_lengths_min", type=int, default=1000)
    parser.add_argument("--context_lengths_max", type=int, default=16000)
    parser.add_argument("--context_lengths_num_intervals", type=int, default=35)
    parser.add_argument("--document_depth_percent_min", type=int, default=0)
    parser.add_argument("--document_depth_percent_max", type=int, default=100)
    parser.add_argument("--document_depth_percent_intervals", type=int, default=35)
    parser.add_argument(
        "--document_depth_percent_interval_type",
        type=str,
        default="linear",
        choices=["linear", "sigmoid"],
    )
    parser.add_argument("--save_results", type=lambda x: x.lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--save_contexts", type=lambda x: x.lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--results_version", type=int, default=1)
    parser.add_argument("--final_context_length_buffer", type=int, default=200)
    parser.add_argument("--print_ongoing_status", type=lambda x: x.lower() in ("true", "1", "yes"), default=True)
    # BERAG-SG
    parser.add_argument("--segment_length", type=int, default=4, help="Segment length m for BERAG-SG.")
    parser.add_argument("--berag_sg_beam_size", type=int, default=2, help="Beam size B for BERAG-SG.")
    parser.add_argument(
        "--max_composite_size",
        type=int,
        default=2,
        help="Max number of original chunks in a composite state (stop when reached).",
    )
    parser.add_argument("--chunk_size", type=int, default=512, help="Token chunk size when num_of_chunks unset.")
    parser.add_argument("--num_of_chunks", type=int, default=None, help="If set, split context into this many chunks.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--berag_sg_top_p",
        type=float,
        default=0.9,
        help="TopP nucleus over chunks when expanding beams (skip low-prob chunks). Use 0 to disable. Default 0.9.",
    )
    parser.add_argument(
        "--berag_sg_temperature",
        type=float,
        default=0.5,
        help="Temperature for multinomial sampling when generating segments. Default 0.5; use 0 for greedy.",
    )
    parser.add_argument(
        "--berag_sg_beam_prune",
        type=str,
        default="diverse_beam_search",
        choices=["diverse_beam_search", "top_b"],
        help="Beam prune: diverse_beam_search (best from each conditioning chunk, round-robin) or top_b (top B by score). Default diverse_beam_search.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for BERAG-SG beam/posterior logs; defaults to output_dir when use_output_dir_for_results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/0226/SimpleNIAH_BERAGSG",
        help="Directory for results/contexts and optionally logs.",
    )
    parser.add_argument(
        "--use_output_dir_for_results",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Write results/contexts under output_dir (with decode_suffix subdir when set).",
    )
    parser.add_argument(
        "--decode_suffix",
        type=str,
        default=None,
        help="Subdir name under output_dir for this config. If not set, auto-generated from segment_length, beam_size, top_p, etc.",
    )
    parser.add_argument("--debug", type=lambda x: x.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--model_name", type=str, default="simple_niah_berag_sg")
    parser.add_argument("--inference_timeout", type=float, default=3600.0)
    parser.add_argument("--max_batch_size_per_forward", type=int, default=5)
    # Prior (optional)
    parser.add_argument("--prior_head_path", type=str, default=None, help="Optional path to passage prior head.")
    parser.add_argument("--prior_head_config", type=str, default=None, help="JSON or empty; optional prior head config.")
    # Multi-needle
    parser.add_argument("--multi_needle", type=lambda x: x.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--needles", type=str, default=None, help='JSON list of needles, e.g. \'["n1", "n2"]\'.')
    args = parser.parse_args()

    if args.multi_needle:
        if not args.needles:
            parser.error("--multi_needle requires --needles (JSON list)")
        try:
            needles_list = json.loads(args.needles)
        except json.JSONDecodeError as e:
            parser.error(f"--needles must be valid JSON: {e}")
        if not isinstance(needles_list, list) or not all(isinstance(n, str) for n in needles_list):
            parser.error("--needles must be a JSON list of strings")
        if len(needles_list) < 1:
            parser.error("--needles must contain at least one needle")
    else:
        needles_list = None

    _setup_paths()

    decode_suffix = args.decode_suffix if args.decode_suffix else _build_decode_suffix(args)
    effective_output_dir = (
        os.path.join(args.output_dir, decode_suffix)
        if (args.use_output_dir_for_results and args.output_dir)
        else args.output_dir
    )
    log_dir = args.log_dir if args.log_dir else effective_output_dir
    results_dir = os.path.join(effective_output_dir, "results") if effective_output_dir else None
    contexts_dir = os.path.join(effective_output_dir, "contexts") if effective_output_dir else None
    if effective_output_dir:
        print(f"[BERAG-SG] decode_suffix={decode_suffix} -> output_dir={effective_output_dir} log_dir={log_dir}")

    from needlehaystack import LLMMultiNeedleHaystackTester, LLMNeedleHaystackTester
    from src.hf_backend import HFQwen2VLBackend
    from src.berag_inference_engine import BAPEInferenceEngine
    from src.simple_niah_berag_sg_provider import SimpleNiahBERAGSGProvider
    from src.simple_niah_esf_provider import SimpleNiahLocalEvaluator, SimpleNiahMultiNeedleEvaluator

    processor_path = args.processor_path or args.model_path
    backend = HFQwen2VLBackend(
        model_path=args.model_path,
        processor_path=processor_path,
        adapter_name_or_path=args.adapter_path,
        max_batch_size_per_forward=args.max_batch_size_per_forward,
    )

    prior_head_config = None
    if args.prior_head_config:
        try:
            prior_head_config = json.loads(args.prior_head_config)
        except json.JSONDecodeError:
            prior_head_config = {}

    engine = BAPEInferenceEngine(
        backend=backend,
        prior_head_path=args.prior_head_path,
        prior_head_config=prior_head_config,
        num_beams=0,
        debug=args.debug,
        segment_length=args.segment_length,
        berag_sg_beam_size=args.berag_sg_beam_size,
        max_composite_size=args.max_composite_size,
        log_dir=log_dir,
        berag_sg_top_p=args.berag_sg_top_p if args.berag_sg_top_p > 0 else None,
        berag_sg_temperature=args.berag_sg_temperature,
        berag_sg_beam_prune=args.berag_sg_beam_prune,
    )
    needle_for_provider = needles_list[0] if needles_list else args.needle
    provider = SimpleNiahBERAGSGProvider(
        engine=engine,
        backend=backend,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        num_of_chunks=args.num_of_chunks,
        max_new_tokens=args.max_new_tokens,
        needle=needle_for_provider,
        inference_timeout=args.inference_timeout,
        needles=needles_list,
        segment_length=args.segment_length,
        berag_sg_beam_size=args.berag_sg_beam_size,
        max_composite_size=args.max_composite_size,
        log_dir=log_dir,
        berag_sg_top_p=args.berag_sg_top_p if args.berag_sg_top_p > 0 else None,
        berag_sg_temperature=args.berag_sg_temperature,
        berag_sg_beam_prune=args.berag_sg_beam_prune,
    )

    if args.multi_needle:
        evaluator = SimpleNiahMultiNeedleEvaluator(needles=needles_list, question_asked=args.retrieval_question)
    else:
        evaluator = SimpleNiahLocalEvaluator(needle=args.needle, question_asked=args.retrieval_question)

    if args.multi_needle:
        tester = LLMMultiNeedleHaystackTester(
            model_to_test=provider,
            evaluator=evaluator,
            needle=needles_list[0],
            needles=needles_list,
            haystack_dir=args.haystack_dir,
            retrieval_question=args.retrieval_question,
            results_version=args.results_version,
            context_lengths_min=args.context_lengths_min,
            context_lengths_max=args.context_lengths_max,
            context_lengths_num_intervals=args.context_lengths_num_intervals,
            document_depth_percent_min=args.document_depth_percent_min,
            document_depth_percent_max=args.document_depth_percent_max,
            document_depth_percent_intervals=args.document_depth_percent_intervals,
            document_depth_percent_interval_type=args.document_depth_percent_interval_type,
            save_results=args.save_results,
            save_contexts=args.save_contexts,
            final_context_length_buffer=args.final_context_length_buffer,
            print_ongoing_status=args.print_ongoing_status,
            results_dir=results_dir,
            contexts_dir=contexts_dir,
        )
    else:
        tester = LLMNeedleHaystackTester(
            model_to_test=provider,
            evaluator=evaluator,
            needle=args.needle,
            haystack_dir=args.haystack_dir,
            retrieval_question=args.retrieval_question,
            results_version=args.results_version,
            context_lengths_min=args.context_lengths_min,
            context_lengths_max=args.context_lengths_max,
            context_lengths_num_intervals=args.context_lengths_num_intervals,
            document_depth_percent_min=args.document_depth_percent_min,
            document_depth_percent_max=args.document_depth_percent_max,
            document_depth_percent_intervals=args.document_depth_percent_intervals,
            document_depth_percent_interval_type=args.document_depth_percent_interval_type,
            save_results=args.save_results,
            save_contexts=args.save_contexts,
            final_context_length_buffer=args.final_context_length_buffer,
            print_ongoing_status=args.print_ongoing_status,
            results_dir=results_dir,
            contexts_dir=contexts_dir,
        )
    tester.start_test()


if __name__ == "__main__":
    main()
