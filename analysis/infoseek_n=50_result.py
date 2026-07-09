#!/usr/bin/env python3
"""
Script to compute average scores from multiple InfoseekNew BAPE result files.
"""

import json
from pathlib import Path
from collections import defaultdict

# Define the paths to the scores files
base_dir = Path("/home/jc2124/rds/rds-cvnlp-hirYTW1FQIw/shared_space/A-RAVQA")
scores_files = [
    base_dir / "outputs/1125-v2/BAPE/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=50-h4-prior=prior_head-retrieved_passage-TakeN=1000-Offset=2000/scores.json",
    base_dir / "outputs/1125-v2/BAPE/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=50-h4-prior=prior_head-retrieved_passage-TakeN=1000-Offset=1000/scores.json",
    base_dir / "outputs/1125-v2/BAPE/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=50-h4-prior=prior_head-retrieved_passage-TakeN=1000-Offset=3000/scores.json",
    base_dir / "outputs/1125-v2/BAPE/InfoseekNew-BAPE-BEFT[K=2]-data=64000-K=50-h4-prior=prior_head-retrieved_passage-TakeN=708-Offset=4000/scores.json",
]

def compute_averages(scores_files):
    """Compute average scores across multiple JSON files."""
    # Simple numeric fields to average
    numeric_fields = [
        "score",
        "unseen_question_score",
        "unseen_entity_score",
        "posterior_passage_hit_rate",
        "retrieval_hit_rate",
        "prior_passage_hit_rate",
        "correct_ignore_rate",
    ]
    
    # Collect all values
    all_data = []
    for file_path in scores_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.append(data)
    
    # Compute averages for simple numeric fields
    averages = {}
    for field in numeric_fields:
        values = [data[field] for data in all_data]
        averages[field] = sum(values) / len(values)
    
    # Compute averages for prior_recall_at_k
    averages["prior_recall_at_k"] = {}
    k_values = sorted([int(k) for k in all_data[0]["prior_recall_at_k"].keys()])
    
    for k in k_values:
        k_str = str(k)
        values = [data["prior_recall_at_k"][k_str] for data in all_data]
        averages["prior_recall_at_k"][k_str] = sum(values) / len(values)
    
    return averages

def main():
    """Main function to compute and print averages."""
    print("Computing averages from the following files:")
    for file_path in scores_files:
        print(f"  - {file_path.name}")
    print()
    
    averages = compute_averages(scores_files)
    
    print("Average Scores:")
    print("=" * 50)
    print(f"Score: {averages['score']:.2f}")
    print(f"Unseen Question Score: {averages['unseen_question_score']:.2f}")
    print(f"Unseen Entity Score: {averages['unseen_entity_score']:.2f}")
    print(f"Posterior Passage Hit Rate: {averages['posterior_passage_hit_rate']:.4f}")
    print(f"Retrieval Hit Rate: {averages['retrieval_hit_rate']:.4f}")
    print(f"Prior Passage Hit Rate: {averages['prior_passage_hit_rate']:.4f}")
    print(f"Correct Ignore Rate: {averages['correct_ignore_rate']:.4f}")
    print()
    print("Prior Recall at K (selected values):")
    print("-" * 50)
    for k in [1, 5, 10, 20, 30, 40, 50]:
        k_str = str(k)
        print(f"  K={k}: {averages['prior_recall_at_k'][k_str]:.4f}")
    
    # Save to JSON file
    output_file = base_dir / "analysis/infoseek_n=50_average_scores.json"
    with open(output_file, 'w') as f:
        json.dump(averages, f, indent=2)
    
    print()
    print(f"Full results saved to: {output_file}")

if __name__ == "__main__":
    main()

