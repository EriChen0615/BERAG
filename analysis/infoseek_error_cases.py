import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset
import sys 
sys.path.append("./src/")
from vqa_datasets import load_passages

def load_and_analyze_results(csv_path, score_threshold=60.0):
    """
    Load results and analyze error cases below the score threshold.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Sort by score ascending (worst cases first)
    df_sorted = df.sort_values('score')
    
    # Separate into error cases and good cases
    error_cases = df_sorted[df_sorted['score'] < score_threshold]
    good_cases = df_sorted[df_sorted['score'] >= score_threshold]
    
    return error_cases, good_cases

def create_question_docs_map(dataset, pid_to_content_map):
    """
    Create a mapping from question_id to its ground truth documents.
    """
    question_docs_map = {}
    for item in dataset:
        if 'pos_item_ids' in item:
            docs = []
            for pid in item['pos_item_ids']:
                if pid in pid_to_content_map:
                    docs.append({
                        'id': pid,
                        'content': pid_to_content_map[pid]
                    })
            question_docs_map[item['question_id']] = docs
    return question_docs_map

def print_case_analysis(case_df, case_type="Error"):
    """
    Print detailed analysis of cases.
    """
    print(f"\n{case_type} Cases Analysis ({len(case_df)} cases)")
    print("=" * 80)
    
    # Load InfoseekNew dataset and passages
    dataset = load_dataset("Jingbiao/aravqa", "Infoseek_data", split='valid_m2kr')
    passage_ds, pid_to_content_map = load_passages("InfoseekNew_FullPassage", split='valid')
    
    # Create question to documents map once
    question_docs_map = create_question_docs_map(dataset, pid_to_content_map)
    
    for idx, row in case_df.iterrows():
        print(f"\nCase {idx + 1} (Score: {row['score']:.1f})")
        print("-" * 80)
        print(f"Question ID: {row['question_id']}")
        print(f"Image Path: {row['img_path']}")
        print(f"Question: {row['question']}")
        print(f"Ground Truth: {row['gt_answers']}")
        print(f"All answers: {row['all_answers']}")
        print(f"Model Answer: {row['answer']}")
        
        # Use the pre-computed map to get ground truth documents
        if row['question_id'] in question_docs_map:
            print("\nGround Truth Documents:")
            for doc in question_docs_map[row['question_id']]:
                print(f"\nDocument ID: {doc['id']}")
                print(f"Content: {doc['content']}")
        
        print("-" * 80)

def plot_score_distribution(df, output_dir):
    """
    Create a histogram of scores.
    """
    plt.figure(figsize=(10, 6))
    plt.title('Distribution of GPT-4 Judgment Scores')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.savefig(Path(output_dir) / 'score_distribution.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze GPT-4 judgment results')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file with GPT-4 judgments')
    parser.add_argument('--threshold', type=float, default=60.0,
                       help='Score threshold below which cases are considered errors')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directory to save analysis outputs')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load and analyze results
    error_cases, good_cases = load_and_analyze_results(args.csv_path, args.threshold)
    
    # Print summary statistics
    total_cases = len(error_cases) + len(good_cases)
    print(f"\nAnalysis Summary")
    print(f"Total cases: {total_cases}")
    print(f"Error cases: {len(error_cases)} ({len(error_cases)/total_cases*100:.1f}%)")
    print(f"Good cases: {len(good_cases)} ({len(good_cases)/total_cases*100:.1f}%)")
    print(f"Average score: {(error_cases['score'].sum() + good_cases['score'].sum())/total_cases:.1f}")
    
    # Print detailed analysis of error cases
    print_case_analysis(error_cases, "Error")
    
    # Create and save visualizations
    plot_score_distribution(pd.concat([error_cases, good_cases]), output_dir)
    
    # Save error cases to a separate CSV for further analysis
    error_cases.to_csv(output_dir / 'error_cases.csv', index=False)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()