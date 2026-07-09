#!/usr/bin/env python3
"""
Script to remove main images from BEFT training data JSON file.
Sets all "images" fields (main images) to empty arrays, keeping only passage-specific images.
"""

import json
import os
import sys
from pathlib import Path


def remove_main_images(input_file: str, output_file: str = None):
    """
    Remove main images from BEFT training data.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input file)
    """
    # Read input file
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} examples")
    
    # Remove main images from each example
    modified_count = 0
    for i, example in enumerate(data):
        if "images" in example:
            original_images = example["images"]
            if original_images and len(original_images) > 0:
                example["images"] = []
                modified_count += 1
                if i < 5:  # Print first few examples for verification
                    print(f"  Example {i}: Removed {len(original_images)} main image(s): {original_images}")
    
    print(f"\nModified {modified_count} examples (removed main images)")
    
    # Determine output file
    if output_file is None:
        output_file = input_file
        # Create backup
        backup_file = input_file + ".backup"
        print(f"\nCreating backup: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Backup created successfully")
    
    # Write output file
    print(f"\nWriting output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully removed main images from {output_file}")
    
    # Verify a few examples
    print("\nVerifying first 3 examples:")
    for i in range(min(3, len(data))):
        example = data[i]
        print(f"  Example {i}:")
        print(f"    images: {example.get('images', [])}")
        if 'passages' in example and len(example['passages']) > 0:
            print(f"    passage 0 images: {example['passages'][0].get('images', [])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_main_images.py <input_json_file> [output_json_file]")
        print("  If output_file is not specified, input file will be overwritten (with backup)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    remove_main_images(input_file, output_file)

