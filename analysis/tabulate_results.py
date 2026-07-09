"""
This script is used to tabulate the results of experiments.
For each directory, it checks that the directory name matches all MUST_MATCH keywords and does not contain any of the MUST_NOT_MATCH keywords.
If it does, it adds the directory to the list of matching directories.

The scores can be found under the `scores.json` file in the directory.
"""
import os
import json
import pandas as pd
from tabulate import tabulate
import re
from pathlib import Path

# "BAPE-RAG2_PPL[Ensemble]-wPrior-FullEPL1"
RESULT_DIR = "outputs/1125-v2/BAPE"
# RESULT_DIR = "outputs/1025/VLLM"
# MUST_MATCH = ["DPO-RAG-K=5"]
# MUST_MATCH = ["RAG2_PPL[Ensemble]", "wPrior", "prior=prior_head", "retrieved_passage", "FullEPL1"]
# MUST_MATCH = ["RAG2_PPL[Ensemble]", "wPrior10", "prior=prior_head", "retrieved_passage", "FullEPL1"]
# MUST_MATCH = ["l1h4", "cont"]
# MUST_MATCH = ["l0h4", "step14000", "bs8"]
MUST_MATCH = ["7B", "reranked_passage", "TakeN=0", "epoch1"]
# MUST_MATCH = ["s1h4"]
# MUST_MATCH = ["7B-BAPE", "RAG2_PPL[Ensemble]", "wPrior", "prior=prior_head", "retrieved_passage"]
# MUST_MATCH = ["wPrior5", "FullEPL1", "retrieved_passage", "prior=prior_head"]
# MUST_NOT_MATCH = ["DynamicKTopP", "7B"]
# MUST_NOT_MATCH = ["hasGTdoc", "wPrior_dyn_token", 'wPrior5', '7B', "DynamicKTopP"]
# MUST_NOT_MATCH = ["hasGTdoc", "wPrior10", "wPrior_", '7B']
MUST_NOT_MATCH = []

def matches_criteria(dirname, must_match, must_not_match):
    """Check if directory name matches all must_match and none of must_not_match."""
    # Check all must_match keywords are present
    for keyword in must_match:
        if keyword not in dirname:
            return False
    
    # Check none of must_not_match keywords are present
    for keyword in must_not_match:
        if keyword in dirname:
            return False
    
    return True

def extract_params_from_dirname(dirname):
    """Extract experiment parameters from directory name."""
    params = {}
    
    # Extract K value
    k_match = re.search(r'K=(\d+)', dirname)
    if k_match:
        params['K'] = int(k_match.group(1))

    topk_match = re.search(r'Top(\d+)', dirname)
    if topk_match:
        params['TopK'] = int(topk_match.group(1))
    
    # Extract passage type
    if 'retrieved_passage' in dirname:
        params['passage_type'] = 'retrieved'
    elif 'reranked_passage' in dirname:
        params['passage_type'] = 'reranked'
    else:
        params['passage_type'] = 'unknown'
    
    # Extract TakeN value
    taken_match = re.search(r'TakeN=(\d+)', dirname)
    if taken_match:
        params['TakeN'] = int(taken_match.group(1))
    
    # Extract DynamicKTopP if present
    topp_match = re.search(r'DynamicKTopP=([\d.]+)', dirname)
    if topp_match:
        params['TopP'] = float(topp_match.group(1))
    else:
        params['TopP'] = 1.0
    
    # Extract prior type
    prior_match = re.search(r'prior=([^-]+)', dirname)
    if prior_match:
        params['prior'] = prior_match.group(1)

    topk_match = re.search(r'K=(\d+)', dirname)
    if topk_match:
        params['K'] = int(topk_match.group(1))
    
    return params

if __name__ == "__main__":
    matching_dirs = []
    
    # Scan through result directory
    for subdir in os.listdir(RESULT_DIR):
        subdir_path = os.path.join(RESULT_DIR, subdir)
        
        # Check if it's a directory
        if not os.path.isdir(subdir_path):
            continue
        
        # Check if it matches criteria
        if not matches_criteria(subdir, MUST_MATCH, MUST_NOT_MATCH):
            continue
        
        # Check if scores.json exists
        scores_file = os.path.join(subdir_path, "scores.json")
        if not os.path.exists(scores_file):
            print(f"Warning: {subdir} matches criteria but has no scores.json")
            continue
        
        # Load scores
        try:
            with open(scores_file, 'r') as f:
                scores = json.load(f)
        except Exception as e:
            print(f"Error loading {scores_file}: {e}")
            continue
        
        # Extract parameters from directory name
        params = extract_params_from_dirname(subdir)
        
        # Combine parameters and scores
        result = {**params, **scores}
        result['dirname'] = subdir
        
        matching_dirs.append(result)
    
    # Create DataFrame
    if not matching_dirs:
        print("No matching directories found!")
    else:
        df = pd.DataFrame(matching_dirs)
        
        # Sort by K and TopP for better readability
        sort_cols = []
        if 'K' in df.columns:
            sort_cols.append('K')
        if 'TopK' in df.columns:
            sort_cols.append('TopK')
        if 'TopP' in df.columns:
            sort_cols.append('TopP')
        
        if sort_cols:
            df = df.sort_values(by=sort_cols)
        
        # Select columns to display (customize as needed)
        display_cols = ['K', 'passage_type', 'TopP', 'prior']
        # Add score columns (assuming common metric names)
        score_cols = [col for col in df.columns if col not in display_cols + ['dirname', 'TakeN']]
        display_cols.extend(score_cols)
        
        # Filter to existing columns
        display_cols = [col for col in display_cols if col in df.columns]
        
        # Display table
        print(f"\nFound {len(df)} matching experiments:")
        print("=" * 80)
        print(tabulate(df[display_cols], headers='keys', tablefmt='grid', showindex=False, floatfmt='.4f'))
        
        # Save to CSV
        output_file = os.path.join(RESULT_DIR, "results_summary.csv")
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
