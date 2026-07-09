#!/usr/bin/env python3
"""
Script to compare two prior_head.pt checkpoint files.
Usage: python compare_prior_heads.py <path_to_prior_head1.pt> <path_to_prior_head2.pt>
"""

import argparse
import torch
import sys


def compare_prior_heads(path1, path2):
    """Compare two prior_head.pt files and report differences."""
    
    print(f"Loading prior_head 1 from: {path1}")
    try:
        state_dict1 = torch.load(path1, map_location='cpu')
    except Exception as e:
        print(f"Error loading {path1}: {e}")
        sys.exit(1)
    
    print(f"Loading prior_head 2 from: {path2}")
    try:
        state_dict2 = torch.load(path2, map_location='cpu')
    except Exception as e:
        print(f"Error loading {path2}: {e}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Check if keys match
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    if keys1 != keys2:
        print("\n⚠️  WARNING: State dict keys don't match!")
        print(f"Keys only in prior_head 1: {keys1 - keys2}")
        print(f"Keys only in prior_head 2: {keys2 - keys1}")
        print()
    else:
        print(f"\n✓ Both prior_heads have the same {len(keys1)} keys")
    
    # Compare each parameter
    common_keys = keys1 & keys2
    all_identical = True
    
    for key in sorted(common_keys):
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        print(f"\n{key}:")
        print(f"  Shape: {param1.shape}")
        
        if param1.shape != param2.shape:
            print(f"  ⚠️  Shape mismatch! prior_head 2 shape: {param2.shape}")
            all_identical = False
            continue
        
        # Check if identical
        are_identical = torch.equal(param1, param2)
        
        if are_identical:
            print(f"  ✓ Identical")
        else:
            all_identical = False
            print(f"  ✗ Different")
            
            # Calculate statistics
            diff = param1 - param2
            abs_diff = torch.abs(diff)
            rel_diff = abs_diff / (torch.abs(param1) + 1e-10)
            
            print(f"  Absolute difference:")
            print(f"    Max:  {abs_diff.max().item():.6e}")
            print(f"    Mean: {abs_diff.mean().item():.6e}")
            print(f"    Min:  {abs_diff.min().item():.6e}")
            
            print(f"  Relative difference:")
            print(f"    Max:  {rel_diff.max().item():.6e}")
            print(f"    Mean: {rel_diff.mean().item():.6e}")
            
            # L2 norm
            l2_norm = torch.norm(diff).item()
            l2_norm_param1 = torch.norm(param1).item()
            print(f"  L2 norm of difference: {l2_norm:.6e}")
            print(f"  Relative L2 norm: {l2_norm / (l2_norm_param1 + 1e-10):.6e}")
            
            # Show some sample values
            print(f"  Sample values (first 5):")
            flat1 = param1.flatten()[:5]
            flat2 = param2.flatten()[:5]
            for i, (v1, v2) in enumerate(zip(flat1, flat2)):
                print(f"    [{i}] prior_head1: {v1.item():.6e}, prior_head2: {v2.item():.6e}")
    
    print("\n" + "="*80)
    if all_identical:
        print("✓ RESULT: All parameters are IDENTICAL")
    else:
        print("✗ RESULT: Parameters are DIFFERENT")
    print("="*80 + "\n")
    
    return all_identical


def main():
    parser = argparse.ArgumentParser(
        description="Compare two prior_head.pt checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "prior_head1",
        type=str,
        help="Path to first prior_head.pt file"
    )
    parser.add_argument(
        "prior_head2", 
        type=str,
        help="Path to second prior_head.pt file"
    )
    
    args = parser.parse_args()
    
    are_identical = compare_prior_heads(args.prior_head1, args.prior_head2)
    
    # Exit code: 0 if identical, 1 if different
    sys.exit(0 if are_identical else 1)


if __name__ == "__main__":
    main()

