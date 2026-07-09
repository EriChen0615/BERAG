"""Script to evaluate InfoseekNew predictions using BEM."""

import argparse
import json
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import multiprocessing
from evaluation_utils import evaluate_example

def load_predictions(pred_path: str) -> List[Dict]:
    """Load predictions from jsonl file."""
    predictions = []
    with open(pred_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions

def load_references(split: str = 'valid') -> Dict:
    """Load reference answers for InfoseekNew."""
    if split in ['test', 'valid', 'valid_m2kr']:
        ref_path = "third_party/infoseek_eval/infoseek/infoseek_val.jsonl"
    elif split in ['train']:
        ref_path = "third_party/infoseek_eval/infoseek/infoseek_train.jsonl"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    references = {}
    with open(ref_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            references[data['data_id']] = {
                'question': data['question'],
                'answers': data['answer_eval']
            }
    return references

def process_single_example(args: Tuple[str, List[str], str, str]) -> float:
    """Process a single example for multiprocessing."""
    question, reference_list, candidate, question_type = args
    try:
        score = evaluate_example(
            question=question,
            reference_list=reference_list,
            candidate=candidate,
            question_type=question_type
        )
    except:
        score = 0.0
    return score

def evaluate_predictions(predictions: List[Dict], references: Dict, num_processes: int = 8) -> Dict:
    """Evaluate predictions using BEM with multiprocessing."""
    # Prepare arguments for multiprocessing
    eval_args = []
    for pred in predictions:
        data_id = pred['data_id']
        if data_id not in references:
            continue
            
        ref_data = references[data_id]
        question = ref_data['question']
        reference_list = ref_data['answers']
        candidate = pred['prediction']
        question_type = 'automatic'  # placeholder
        
        eval_args.append((question, reference_list, candidate, question_type))
    
    # Use multiprocessing to evaluate
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # no GPU for multiprocessing
    with multiprocessing.Pool(processes=num_processes) as pool:
        scores = list(tqdm(
            pool.imap(process_single_example, eval_args, chunksize=1),
            total=len(eval_args),
            desc="Evaluating with BEM"
        ))
    
    # Calculate average score
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        'bem_score': avg_score,
        'num_examples': len(scores)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate InfoseekNew predictions using BEM')
    parser.add_argument('--eval_dir', type=str, required=True,
                       help='Directory containing predictions.jsonl')
    parser.add_argument('--split', type=str, default='valid',
                       choices=['train', 'valid', 'test', 'valid_m2kr'],
                       help='Dataset split to evaluate against')
    parser.add_argument('--num_processes', type=int, default=8,
                       help='Number of processes to use for evaluation')
    args = parser.parse_args()
    
    # Load predictions
    pred_path = os.path.join(args.eval_dir, 'predictions.jsonl')
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Could not find predictions at {pred_path}")
    
    predictions = load_predictions(pred_path)
    references = load_references(args.split)
    
    # Evaluate
    results = evaluate_predictions(
        predictions, 
        references,
        num_processes=args.num_processes
    )
    
    # Save results
    output_path = os.path.join(args.eval_dir, 'bem_scores.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"BEM Score: {results['bem_score']:.4f}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main() 