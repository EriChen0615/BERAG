#!/usr/bin/env python3
"""
Compare batched vs sequential processing performance.
"""
import time
import torch
from bape_inference_engine import BAPEInferenceEngine
from hf_backend import HFQwen2VLBackend

def compare_batching_performance():
    """Compare batched vs sequential processing."""
    
    # Initialize backend
    print("Initializing backend...")
    backend = HFQwen2VLBackend(
        model_path="Qwen/Qwen2-VL-2B-Instruct",
        processor_path="Qwen/Qwen2-VL-2B-Instruct",
        adapter_name_or_path="third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/evqa/ppl/rag2-answer-ppl[joint]-size=64000-max=2048"
    )
    
    # Test data
    input_context = {
        "text": "Based on the image and the following evidence: <<<EVIDENCE>>> Please answer the question: What is shown in the image?",
        "image": "data/evqa/images/val2014/COCO_val2014_000000000042.jpg"
    }
    
    passages = [
        "The image shows a person riding a bicycle on a street.",
        "A cyclist is visible in the image, riding down a road.",
        "There is a bicycle rider in the photograph.",
        "A picture contains a person on a bike.",
        "A bicycle can be seen with a person riding it."
    ]
    
    print(f"Testing with {len(passages)} passages...")
    
    # Test 1: Sequential processing (original method)
    print("\n" + "="*50)
    print("SEQUENTIAL PROCESSING")
    print("="*50)
    
    def sequential_forward():
        """Simulate sequential processing."""
        log_probs_list = []
        for i, passage in enumerate(passages):
            # Prepare input for single passage
            inputs = backend.prepare_input(input_context, [], passage)
            # Forward pass
            log_probs, _ = backend.forward_single(inputs)
            log_probs_list.append(log_probs)
        return torch.stack(log_probs_list)
    
    # Warm up
    print("Warming up...")
    for _ in range(3):
        sequential_forward()
    
    # Time sequential processing
    print("Timing sequential processing...")
    start_time = time.time()
    for _ in range(10):  # Run 10 iterations
        seq_log_probs = sequential_forward()
    seq_time = time.time() - start_time
    seq_avg_time = seq_time / 10
    
    print(f"Sequential processing time: {seq_avg_time:.4f}s per iteration")
    print(f"Sequential output shape: {seq_log_probs.shape}")
    
    # Test 2: Batched processing (new method)
    print("\n" + "="*50)
    print("BATCHED PROCESSING")
    print("="*50)
    
    def batched_forward():
        """Simulate batched processing."""
        # Prepare batched input
        batched_inputs = backend.prepare_batched_input(input_context, [], passages)
        # Forward pass
        log_probs, _ = backend.forward(batched_inputs)
        return log_probs
    
    # Warm up
    print("Warming up...")
    for _ in range(3):
        batched_forward()
    
    # Time batched processing
    print("Timing batched processing...")
    start_time = time.time()
    for _ in range(10):  # Run 10 iterations
        batch_log_probs = batched_forward()
    batch_time = time.time() - start_time
    batch_avg_time = batch_time / 10
    
    print(f"Batched processing time: {batch_avg_time:.4f}s per iteration")
    print(f"Batched output shape: {batch_log_probs.shape}")
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    speedup = seq_avg_time / batch_avg_time
    print(f"Sequential time: {seq_avg_time:.4f}s")
    print(f"Batched time: {batch_avg_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Check if outputs are similar
    max_diff = torch.max(torch.abs(seq_log_probs - batch_log_probs)).item()
    print(f"Max difference between outputs: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✓ Outputs are identical (within numerical precision)")
    else:
        print("⚠ Outputs differ - check implementation")
    
    # Memory usage comparison
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Detailed breakdown
    print(f"\nDetailed Breakdown:")
    print(f"  Sequential: {seq_avg_time:.4f}s")
    print(f"  Batched: {batch_avg_time:.4f}s")
    print(f"  Improvement: {((seq_avg_time - batch_avg_time) / seq_avg_time * 100):.1f}%")
    
    return {
        'sequential_time': seq_avg_time,
        'batched_time': batch_avg_time,
        'speedup': speedup,
        'max_diff': max_diff
    }

if __name__ == "__main__":
    print("Batching Performance Comparison")
    print("=" * 50)
    
    try:
        results = compare_batching_performance()
        print(f"\nComparison completed!")
        print(f"Speedup: {results['speedup']:.2f}x")
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
