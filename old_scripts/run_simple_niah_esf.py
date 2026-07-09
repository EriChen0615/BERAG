#!/usr/bin/env python3
"""
Run single-needle simple_niah with BERAG ESF inference engine.

Uses the NIAH tester from third_party/NeedleInAHaystack with a local
SimpleNiahESFProvider (ESF engine + HF backend). Haystack data must live under
third_party/NeedleInAHaystack/needlehaystack/<haystack_dir>/ (e.g. PaulGrahamEssays).

Example:
  PYTHONPATH=".:third_party/NeedleInAHaystack" python scripts/run_simple_niah_esf.py \\
    --model_path /path/to/Qwen2.5-VL-7B-Instruct \\
    --needle "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day." \\
    --retrieval_question "What is the best thing to do in San Francisco?"
"""

import argparse
import json
import os
import re
import sys


def _setup_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    niah_root = os.path.join(repo_root, "third_party", "NeedleInAHaystack")
    if niah_root not in sys.path:
        sys.path.insert(0, niah_root)


def _build_decode_suffix(args) -> str:
    """Build a short, filesystem-safe suffix from decoding parameters so different configs write to different dirs."""
    parts = []
    k = getattr(args, "transition_kernel", "identity")
    if k == "log_linear":
        parts.append("loglin")
    elif k == "multiplicative_threshold":
        parts.append("mult")
        tw = getattr(args, "threshold_combined", 0.5)
        if tw != 0.5:
            parts.append(f"tw{tw}")
    elif k == "mass_threshold_kernel":
        parts.append("mass")
        eps = getattr(args, "mass_threshold_epsilon", 0.0)
        if eps != 0.0:
            parts.append(f"eps{eps}")
    else:
        parts.append("id")
    parts.append(f"b{getattr(args, 'beam_width', 4)}")
    parts.append(f"s{getattr(args, 'segment_size', 10)}")
    max_state = getattr(args, "max_state_size", None)
    if max_state is not None:
        parts.append(f"m{max_state}")
    topk = getattr(args, "state_explore_TopK", 10)
    parts.append(f"k{topk}")
    decode_mode = getattr(args, "decode_mode", "segment_beam")
    if decode_mode != "segment_beam":
        parts.append(decode_mode[:4])
    suffix = "_".join(str(p) for p in parts)
    suffix = re.sub(r"[^a-zA-Z0-9_.-]", "_", suffix)
    return suffix or "default"


def main():
    parser = argparse.ArgumentParser(
        description="Run simple_niah (single-needle) with ESF or standard generation and local model."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="esf",
        choices=["esf", "standard"],
        help="Provider: esf (BERAG ESF engine) or standard (full-context model.generate()). Default: esf.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HF model (e.g. Qwen2.5-VL-7B-Instruct).",
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Path to processor; defaults to model_path.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional LoRA adapter path.",
    )
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
        help="Directory name under needlehaystack/ (e.g. PaulGrahamEssays).",
    )
    parser.add_argument(
        "--context_lengths_min",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--context_lengths_max",
        type=int,
        default=16000,
    )
    parser.add_argument(
        "--context_lengths_num_intervals",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--document_depth_percent_min",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--document_depth_percent_max",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--document_depth_percent_intervals",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--document_depth_percent_interval_type",
        type=str,
        default="linear",
        choices=["linear", "sigmoid"],
    )
    parser.add_argument(
        "--save_results",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )
    parser.add_argument(
        "--save_contexts",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )
    parser.add_argument(
        "--results_version",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--final_context_length_buffer",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--print_ongoing_status",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )
    # ESF / provider options
    parser.add_argument(
        "--segment_size",
        type=int,
        default=10,
        help="ESF segment size.",
    )
    parser.add_argument(
        "--debug",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="When True, print engine progress at each beam update (segment tokens and scores).",
    )
    parser.add_argument(
        "--state_explore_TopK",
        type=int,
        default=10,
        help="Max states to expand per beam (and for belief support pruning).",
    )
    parser.add_argument(
        "--state_explore_mode",
        type=str,
        default="TopP_capped",
        choices=["TopK", "TopP", "TopP_capped"],
        help="State expansion: TopK, TopP (nucleus), or TopP_capped (nucleus capped by state_explore_TopK). Default TopP_capped.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus probability for TopP / TopP_capped state selection (default 0.95).",
    )
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="segment_beam",
        choices=["mixture", "map", "segment_beam"],
        help="Decode mode: mixture (token-level), map, or segment_beam (default).",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=4,
        help="Number of beams for segment_beam decode mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/0226/SimpleNIAH",
        help="Directory for inference logs and (when --use_output_dir_for_results) results/contexts. Default: outputs/0226/SimpleNIAH.",
    )
    parser.add_argument(
        "--decode_suffix",
        type=str,
        default=None,
        help="Subdir name under output_dir for this decode config. If not set, auto-generated from transition_kernel, beam_width, segment_size, etc.",
    )
    parser.add_argument(
        "--use_output_dir_for_results",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="When True (default), write results and contexts under output_dir/results and output_dir/contexts.",
    )
    parser.add_argument(
        "--transition_kernel",
        type=str,
        default="identity",
        choices=["identity", "log_linear", "multiplicative_threshold", "mass_threshold_kernel"],
    )
    parser.add_argument(
        "--threshold_single",
        type=float,
        default=None,
        help="Per-state threshold t_s for multiplicative_threshold kernel; if None, engine uses 2/K.",
    )
    parser.add_argument(
        "--threshold_combined",
        type=float,
        default=0.5,
        help="Combined mass threshold t_w for multiplicative_threshold kernel (default 0.5).",
    )
    parser.add_argument(
        "--mass_threshold_epsilon",
        type=float,
        default=0.0,
        help="Epsilon for mass_threshold_kernel: state eligible if b(s) > 1/K + epsilon (default 0.0).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Token chunk size for splitting context into passages (used when --num_of_chunks is not set).",
    )
    parser.add_argument(
        "--num_of_chunks",
        type=int,
        default=None,
        help="If set, context is split into this many chunks (chunk_size derived per context). Overrides --chunk_size.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="simple_niah_berag_esf",
        help="Name used in result files.",
    )
    parser.add_argument(
        "--inference_timeout",
        type=float,
        default=3600.0,
        help="Max seconds per evaluate_model call (default 3600). Raises asyncio.TimeoutError if exceeded.",
    )
    parser.add_argument(
        "--max_batch_size_per_forward",
        type=int,
        default=5,
        help="Max batch size per forward pass; larger batches are split (default 5). Reduces peak GPU memory.",
    )
    # Multi-needle
    parser.add_argument(
        "--multi_needle",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="If true, run multi-needle NIAH (use LLMMultiNeedleHaystackTester, require --needles).",
    )
    parser.add_argument(
        "--needles",
        type=str,
        default=None,
        help="JSON list of needle strings for multi-needle, e.g. '[\"needle1\", \"needle2\"]'. Required when --multi_needle.",
    )
    parser.add_argument(
        "--max_state_size",
        type=int,
        default=None,
        help="Max allowed composite state size (e.g. 2 for two needles). Passed to BERAG ESF engine when set.",
    )
    args = parser.parse_args()

    if args.multi_needle:
        if not args.needles:
            parser.error("--multi_needle requires --needles (JSON list of needle strings)")
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
    if effective_output_dir:
        print(f"[decode config] suffix={decode_suffix} -> output_dir={effective_output_dir}")

    from needlehaystack import LLMNeedleHaystackTester, LLMMultiNeedleHaystackTester  # noqa: E402
    from src.hf_backend import HFQwen2VLBackend  # noqa: E402
    from src.simple_niah_esf_provider import (  # noqa: E402
        SimpleNiahLocalEvaluator,
        SimpleNiahMultiNeedleEvaluator,
    )

    processor_path = args.processor_path or args.model_path
    backend = HFQwen2VLBackend(
        model_path=args.model_path,
        processor_path=processor_path,
        adapter_name_or_path=args.adapter_path,
        max_batch_size_per_forward=args.max_batch_size_per_forward,
    )

    if args.provider == "esf":
        from src.berag_esf_inference_engine import BERAGESFInferenceEngine  # noqa: E402
        from src.simple_niah_esf_provider import SimpleNiahESFProvider  # noqa: E402

        engine = BERAGESFInferenceEngine(
            backend=backend,
            segment_size=args.segment_size,
            state_explore_TopK=args.state_explore_TopK,
            decode_mode=args.decode_mode,
            beam_width=args.beam_width,
            log_dir=effective_output_dir if effective_output_dir else None,
            debug=args.debug,
            transition_kernel=args.transition_kernel,
            state_explore_mode=args.state_explore_mode,
            top_p=args.top_p,
            max_state_size=args.max_state_size,
            threshold_single=args.threshold_single,
            threshold_combined=args.threshold_combined,
            mass_threshold_epsilon=args.mass_threshold_epsilon,
        )
        needle_for_provider = needles_list[0] if needles_list else args.needle
        provider = SimpleNiahESFProvider(
            engine=engine,
            backend=backend,
            model_name=args.model_name,
            chunk_size=args.chunk_size,
            num_of_chunks=args.num_of_chunks,
            max_new_tokens=args.max_new_tokens,
            needle=needle_for_provider,
            inference_timeout=args.inference_timeout,
            needles=needles_list,
        )
    else:
        from src.simple_niah_standard_provider import SimpleNiahStandardProvider  # noqa: E402

        model_name = (
            args.model_name
            if args.model_name != "simple_niah_berag_esf"
            else "simple_niah_standard"
        )
        provider = SimpleNiahStandardProvider(
            backend=backend,
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
            needle=args.needle,
        )

    if args.multi_needle:
        evaluator = SimpleNiahMultiNeedleEvaluator(
            needles=needles_list,
            question_asked=args.retrieval_question,
        )
    else:
        evaluator = SimpleNiahLocalEvaluator(
            needle=args.needle,
            question_asked=args.retrieval_question,
        )

    results_dir = None
    contexts_dir = None
    if args.use_output_dir_for_results and args.output_dir:
        results_dir = os.path.join(effective_output_dir, "results")
        contexts_dir = os.path.join(effective_output_dir, "contexts")

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
