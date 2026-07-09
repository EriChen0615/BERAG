#!/usr/bin/env python3
"""
Profile script to identify bottlenecks in BAPE inference engine.
"""
import time
import torch
import argparse
from bape_inference_engine import BAPEInferenceEngine
from hf_backend import HFQwen2VLBackend
import cProfile
import pstats
from io import StringIO

def profile_inference():
    """Profile the BAPE inference process."""
    
    # Initialize backend
    print("Initializing backend...")
    start_time = time.time()
    backend = HFQwen2VLBackend(
        model_path="Qwen/Qwen2-VL-2B-Instruct",
        processor_path="Qwen/Qwen2-VL-2B-Instruct",
        adapter_name_or_path="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[joint]-size=64000-max=2048"
    )
    init_time = time.time() - start_time
    print(f"Backend initialization time: {init_time:.2f}s")
    
    # Initialize inference engine
    print("Initializing inference engine...")
    start_time = time.time()
    engine = BAPEInferenceEngine(backend)
    engine_init_time = time.time() - start_time
    print(f"Inference engine initialization time: {engine_init_time:.2f}s")
    
    # Test data
    input_context = {
        "text": "Based on the image and the following evidence: <<<EVIDENCE>>> Please answer the question: What is shown in the image?",
        "image": "../vqa_data/KBVQA_data/EVQA/images/inat/val/00006_Animalia_Arthropoda_Arachnida_Araneae_Araneidae_Aculepeira_ceropegia/083b38d0-2c65-4e82-9111-4a7ef467ba84.jpg"  # Use a real image path
    }
    
    passages = [
        "The image shows a person riding a bicycle on a street.",
        "A cyclist is visible in the image, riding down a road.",
        "There is a bicycle rider in the photograph.",
        "The picture contains a person on a bike.",
        "A bicycle can be seen with a person riding it."
    ]
    
    print(f"Testing with {len(passages)} passages...")
    
    # Profile the generation process
    def run_generation():
        return engine.generate(
            input_context=input_context,
            passages=passages,
            passage_prior=None,
            max_new_tokens=50
        )
    
    # Run with profiling
    print("Running generation with profiling...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    generated_tokens, log_tokens_llk, log_passage_posterior = run_generation()
    total_time = time.time() - start_time
    
    profiler.disable()
    
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Generated {len(generated_tokens)} tokens")
    print(f"Generated text: {backend.processor.tokenizer.decode(generated_tokens)}")
    
    # Print profiling results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())
    
    # Detailed timing analysis
    print("\n" + "="*50)
    print("DETAILED TIMING ANALYSIS")
    print("="*50)
    
    # Test individual components
    print("\n1. Testing input preparation...")
    start_time = time.time()
    batched_inputs = backend.prepare_batched_input(input_context, [], passages)
    prep_time = time.time() - start_time
    print(f"   Batched input preparation: {prep_time:.4f}s")
    
    print("\n2. Testing forward pass...")
    start_time = time.time()
    log_probs, past_key_values = backend.forward(batched_inputs)
    forward_time = time.time() - start_time
    print(f"   Forward pass: {forward_time:.4f}s")
    print(f"   Output shape: {log_probs.shape}")
    
    print("\n3. Testing token sampling...")
    start_time = time.time()
    token_idx, log_token_llk = engine._sample(log_probs[0])  # Use first passage for sampling
    sample_time = time.time() - start_time
    print(f"   Token sampling: {sample_time:.4f}s")
    print(f"   Sampled token: {token_idx}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\n4. GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Performance metrics
    print(f"\n5. Performance Metrics:")
    print(f"   Tokens per second: {len(generated_tokens) / total_time:.2f}")
    print(f"   Passages per second: {len(passages) / total_time:.2f}")
    print(f"   Time per passage: {total_time / len(passages):.4f}s")
    
    return {
        'total_time': total_time,
        'init_time': init_time,
        'prep_time': prep_time,
        'forward_time': forward_time,
        'sample_time': sample_time,
        'tokens_generated': len(generated_tokens),
        'passages': len(passages)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--take_n", type=int, default=5, help="Number of examples to process")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    print("BAPE Inference Profiling")
    print("=" * 50)
    
    try:
        results = profile_inference()
        print(f"\nProfiling completed successfully!")
        print(f"Total time: {results['total_time']:.2f}s")
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
