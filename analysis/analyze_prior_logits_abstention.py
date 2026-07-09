import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import json
import sys
import os
from tqdm import tqdm
import argparse
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Infoseek evaluation imports (only needed for Infoseek dataset)
# Will be imported later if needed

def parse_prior_logits(logits_str):
    """Parse prior_logits string to list of floats"""
    if pd.isna(logits_str):
        return None
    try:
        # Try to parse as Python literal
        logits = ast.literal_eval(logits_str)
        if isinstance(logits, list):
            return [float(x) for x in logits]
        return None
    except:
        return None

def calculate_entropy(logits_list):
    """Calculate entropy from prior logits after softmax"""
    if logits_list is None or len(logits_list) == 0:
        return None
    try:
        # Convert to numpy array and apply softmax
        logits = np.array(logits_list)
        # Subtract max for numerical stability
        logits_stable = logits - np.max(logits)
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / np.sum(exp_logits)
        # Calculate entropy: H = -sum(p_i * log(p_i))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probs * np.log(probs + epsilon))
        return entropy
    except:
        return None

def calculate_weighted_deflection_probability(logits_list):
    """Calculate weighted deflection probability from prior logits
    
    Formula: P(d|x) = Σ_k (1 - σ(s_k)) · P(z_k|x)
    where:
    - σ(s_k) = sigmoid(s_k) = relevance probability for passage k
    - P(z_k|x) = softmax(s_k) = passage prior for passage k
    - s_k = logit for passage k
    
    Returns deflection probability (0-1 scale) or None if invalid
    """
    if logits_list is None or len(logits_list) == 0:
        return None
    try:
        # Convert to numpy array
        logits = np.array(logits_list)
        
        # Calculate passage prior P(z_k|x) using softmax
        logits_stable = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits_stable)
        passage_prior = exp_logits / np.sum(exp_logits)
        
        # Calculate relevance probability σ(s_k) using sigmoid
        # σ(s_k) = 1 / (1 + exp(-s_k))
        relevance_prob = 1.0 / (1.0 + np.exp(-logits))
        
        # Calculate deflection probability for each passage: (1 - σ(s_k)) · P(z_k|x)
        deflection_per_passage = (1.0 - relevance_prob) * passage_prior
        
        # Overall deflection probability: sum over all passages
        deflection_prob = np.sum(deflection_per_passage)
        
        return float(deflection_prob)
    except:
        return None

def calculate_z0_logit_difference(logits_list):
    """Calculate z0 logit difference from max of other logits
    
    z0 is assumed to be the last passage in the list.
    If z0_logit is the maximum, it means the model is not using passages to answer.
    
    Returns: z0_logit - max(other_logits)
    - Positive value: z0_logit is maximum, should deflect
    - Negative value: z0_logit is not maximum, should not deflect
    - None if invalid
    """
    if logits_list is None or len(logits_list) < 2:
        return None
    try:
        logits = np.array(logits_list)
        z0_logit = logits[-1]  # Last logit is z0
        other_logits = logits[:-1]  # All logits except z0
        max_other_logit = np.max(other_logits) if len(other_logits) > 0 else z0_logit
        difference = z0_logit - max_other_logit
        return float(difference)
    except:
        return None


def get_answered_deflected_masks(deflection_scores, deflection_mechanism, threshold):
    """
    Return answered/deflected masks for the configured deflection mechanism.
    """
    if deflection_mechanism == 'max_prior_logits':
        answered_mask = deflection_scores >= threshold
        deflected_mask = deflection_scores < threshold
    elif deflection_mechanism == 'weighted_prior_logits':
        answered_mask = deflection_scores <= threshold  # lower deflection prob => answer
        deflected_mask = deflection_scores > threshold
    elif deflection_mechanism == 'max_z0_logits':
        answered_mask = deflection_scores <= threshold  # z0 not max => answer
        deflected_mask = deflection_scores > threshold
    else:
        raise ValueError(f"Unknown deflection_mechanism: {deflection_mechanism}")
    return answered_mask, deflected_mask
    
def process_experiment(csv_path, K, dataset_type=None, reference=None, reference_qtype=None, qid2example=None, deflection_mechanism='max_prior_logits'):
    """Process a single experiment and return metrics
    
    Args:
        csv_path: Path to CSV file
        K: K value for recall@K
        dataset_type: 'EVQA' or 'Infoseek' (auto-detected if None)
        reference: Infoseek reference data (if needed)
        reference_qtype: Infoseek reference qtype data (if needed)
        qid2example: Infoseek qid2example mapping (if needed)
        deflection_mechanism: 'max_prior_logits', 'weighted_prior_logits', or 'max_z0_logits'
    """
    df = pd.read_csv(csv_path)
    
    # Detect dataset type if not provided
    if dataset_type is None:
        has_score_column = 'score' in df.columns
        dataset_type = 'EVQA' if has_score_column else 'Infoseek'
    
    # Parse prior_logits and calculate deflection scores based on mechanism
    if 'prior_logits' in df.columns:
        df['prior_logits_parsed'] = df['prior_logits'].apply(parse_prior_logits)
        df['max_prior_logit'] = df['prior_logits_parsed'].apply(
            lambda x: max(x) if x is not None and len(x) > 0 else None
        )
        if deflection_mechanism == 'weighted_prior_logits':
            df['weighted_deflection_prob'] = df['prior_logits_parsed'].apply(calculate_weighted_deflection_probability)
        elif deflection_mechanism == 'max_z0_logits':
            df['z0_logit_difference'] = df['prior_logits_parsed'].apply(calculate_z0_logit_difference)
    df['prior_entropy'] = df['prior_logits_parsed'].apply(calculate_entropy)

    # Calculate scores and determine correctness based on dataset type
    if dataset_type == 'EVQA':
        if 'score' not in df.columns:
            raise ValueError("EVQA dataset should have 'score' column")
        df['is_correct'] = (df['score'] >= 0.6).astype(bool)
    else:
        # Infoseek: calculate scores using evaluate_by_example
        # Always ensure evaluate_by_example is imported (even when reference is preloaded)
        sys.path.append("./third_party/infoseek_eval")
        from infoseek_eval import evaluate_by_example
        from infoseek_eval import load_jsonl, prepare_qid2example

        reference_path = "third_party/infoseek_eval/infoseek/infoseek_val.jsonl"
        reference_qtype_path = "third_party/infoseek_eval/infoseek/infoseek_val_qtype.jsonl"
        reference = load_jsonl(reference_path)
        reference_qtype = load_jsonl(reference_qtype_path) if reference_qtype_path is not None else None
        qid2example = prepare_qid2example(reference, reference_qtype)

        scores = []
        is_correct_list = []
        missing_pred = 0
        eval_errors = 0
        error_samples = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {os.path.basename(csv_path)}"):
            question_id = row['question_id']
            prediction = row.get('prediction', row.get('generated_answer', ''))
            
            if pd.isna(prediction) or prediction == '':
                scores.append(0.0)
                is_correct_list.append(False)
                missing_pred += 1
                continue
            
            try:
                preds = [{
                    'data_id': question_id,
                    'prediction': str(prediction)
                }]
                result, gt_answers = evaluate_by_example(preds, reference, reference_qtype, qid2example)
                score = max(
                    result["unseen_entity_score"]["score"], 
                    result["final_score"], 
                    result["unseen_question_score"]["score"]
                )
                scores.append(score)
                is_correct_list.append(score >= 60.0)
            except Exception as e:
                scores.append(0.0)
                is_correct_list.append(False)
                eval_errors += 1
                if len(error_samples) < 5:
                    error_samples.append({
                        "question_id": question_id,
                        "prediction": str(prediction),
                        "error": str(e)
                    })

        df['score'] = scores
        df['is_correct'] = is_correct_list

        if missing_pred > 0 or eval_errors > 0:
            print(f"[warn] {os.path.basename(csv_path)}: skipped {missing_pred} empty predictions, {eval_errors} eval errors")
            if error_samples:
                print("[warn] sample eval errors:")
                for sample in error_samples:
                    print(f"       qid={sample['question_id']} pred={sample['prediction']} err={sample['error']}")
    
    # Calculate overall score (mean score across all questions)
    overall_score = df['score'].mean()

    # Create binary indicator for GT passage inclusion
    df['has_gt_passage'] = (df['gt_passage_in_zidx'] != -1)

    # Compute deflection score based on mechanism
    if deflection_mechanism == 'max_prior_logits':
        df['deflection_score'] = df['max_prior_logit']
        threshold = 0.0  # Logit scale: 0.0 corresponds to probability 0.5
    elif deflection_mechanism == 'weighted_prior_logits':
        df['deflection_score'] = df['weighted_deflection_prob']
        threshold = 0.5  # Probability scale: 0.5 means 50% deflection probability
    elif deflection_mechanism == 'max_z0_logits':
        df['deflection_score'] = df['z0_logit_difference']
        threshold = 0.0  # If z0_logit_difference > 0, z0 is max, should deflect
    else:
        raise ValueError(f"Unknown deflection_mechanism: {deflection_mechanism}")

    # Remove rows where deflection_score is None
    df_valid = df[df['deflection_score'].notna()].copy()
    df_valid = df_valid[df_valid['prior_entropy'].notna()].copy()
    
    results = {}
    
    # Check if we have valid data
    if len(df_valid) == 0:
        print(f"[warn] No valid data found in {os.path.basename(csv_path)} (all deflection scores or entropy are None)")
        # Return empty results but still create the structure
        results[f'K={K}'] = {
            'real_coverage': 0.0,
            'recall_at_k': 0.0,
            'risk_at_cr': 0.0,
            'risk_at_ct_gt': 0.0,
            'risk_at_ct_abstention': 0.0,
            'jaccard_index': 0.0,
            'strict_rag_accuracy': 0.0,
            'strict_rag_risk': 1.0,
            'strict_rag_correct_count': 0,
            'strict_rag_total_count': 0,
            'deflection_precision': 0.0,
            'deflection_recall': 0.0,
            'deflection_f1': 0.0,
            'deflection_accuracy': 0.0,
            'correct_deflection': 0.0,
            'incorrect_deflection': 1.0
        }
        # Still save JSON file
        output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else "."
        gt_coverage_rate = (df['gt_passage_in_zidx'] != -1).mean()
        detailed_results = {
            "dataset_type": dataset_type,
            "deflection_mechanism": deflection_mechanism,
            "overall_score": float(overall_score),
            "gt_passage_coverage_rate": float(gt_coverage_rate),
            "total_samples": int(len(df)),
            "valid_samples": 0,
            "k_metrics": results
        }
        results_path = f"{output_dir}/prior_logits_abstention_analysis_{deflection_mechanism}.json"
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        return results, dataset_type, overall_score
    
    answered_mask, deflected_mask = get_answered_deflected_masks(
        df_valid['deflection_score'], deflection_mechanism, threshold
    )
    
    real_coverage = answered_mask.mean()
    
    # Calculate Risk@C_r: Risk at real coverage
    if answered_mask.sum() > 0:
        risk_at_cr = 1 - df_valid[answered_mask]['is_correct'].mean()
    else:
        risk_at_cr = 0.0
    
    # Deflection as GT-in-context classifier: predict "absent" when deflected
    gt_absent = df_valid['gt_passage_in_zidx'] == -1  # GT not in provided context
    gt_present = ~gt_absent

    tp = ((deflected_mask) & (gt_absent)).sum()
    fp = ((deflected_mask) & (~gt_absent)).sum()
    fn = ((~deflected_mask) & (gt_absent)).sum()
    tn = ((~deflected_mask) & (~gt_absent)).sum()
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    deflection_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    # For backward compatibility in report keys
    correct_deflection = accuracy
    incorrect_deflection = 1 - accuracy

    # StrictRAG correctness:
    # - GT absent: correct iff model deflects
    # - GT present: correct iff answer is correct AND model does NOT deflect
    strict_rag_correct_mask = (gt_absent & deflected_mask) | (gt_present & df_valid['is_correct'] & (~deflected_mask))
    strict_rag_accuracy = strict_rag_correct_mask.mean() if len(df_valid) > 0 else 0.0
    strict_rag_risk = 1 - strict_rag_accuracy
    strict_rag_correct_count = int(strict_rag_correct_mask.sum())
    strict_rag_total_count = int(len(strict_rag_correct_mask))
    
    # Calculate Recall@K (Theoretical Coverage C_t)
    # Recall@K = proportion where gt_passage_in_zidx < K and gt_passage_in_zidx != -1
    recall_at_k = ((df_valid['gt_passage_in_zidx'] >= 0) & (df_valid['gt_passage_in_zidx'] < K)).mean()
    
    # Calculate Risk@C_t(GT label): Risk at theoretical coverage based on GT passage position
    # Risk = error rate among questions where GT passage is in top-K
    top_k_mask = (df_valid['gt_passage_in_zidx'] >= 0) & (df_valid['gt_passage_in_zidx'] < K)
    if top_k_mask.sum() > 0:
        risk_at_ct_gt = 1 - df_valid[top_k_mask]['is_correct'].mean()
    else:
        risk_at_ct_gt = 0.0
    
    # Calculate Jaccard Index: |C_gt ∩ C_r| / |C_gt ∪ C_r|
    # C_gt: set where GT passage is in top-K
    # C_r: set where model doesn't deflect (answered)
    c_gt = top_k_mask
    c_r = answered_mask
    intersection = (c_gt & c_r).sum()
    union = (c_gt | c_r).sum()
    jaccard_index = intersection / union if union > 0 else 0.0
    
    # Calculate Risk@C_t(Abstention): 
    # First, tune the abstention threshold so that C_r = C_t
    # Then find the risk = 1 - accuracy at that threshold
    sorted_scores = np.sort(df_valid['deflection_score'].values)
    # Find the threshold that gives coverage = C_t
    if deflection_mechanism == 'max_prior_logits':
        # For max_prior_logits: higher score = answered, so we need (1 - recall_at_k) percentile from bottom
        # This means (1 - recall_at_k) will be deflected, and recall_at_k will be answered
        threshold_idx = int((1 - recall_at_k) * len(sorted_scores))
        threshold_idx = max(0, min(threshold_idx, len(sorted_scores) - 1))
        threshold_abstention = sorted_scores[threshold_idx]
        answered_mask_abstention, _ = get_answered_deflected_masks(
            df_valid['deflection_score'], deflection_mechanism, threshold_abstention
        )
    elif deflection_mechanism == 'weighted_prior_logits':
        # For weighted_prior_logits: lower deflection_prob = answered, so we need recall_at_k percentile from bottom
        # This means recall_at_k will be answered, and (1 - recall_at_k) will be deflected
        threshold_idx = int(recall_at_k * len(sorted_scores))
        threshold_idx = max(0, min(threshold_idx, len(sorted_scores) - 1))
        threshold_abstention = sorted_scores[threshold_idx]
        answered_mask_abstention, _ = get_answered_deflected_masks(
            df_valid['deflection_score'], deflection_mechanism, threshold_abstention
        )
    else:  # max_z0_logits
        # For max_z0_logits: lower score (z0 not max) = answered, so we need recall_at_k percentile from bottom
        # This means recall_at_k will be answered, and (1 - recall_at_k) will be deflected
        threshold_idx = int(recall_at_k * len(sorted_scores))
        threshold_idx = max(0, min(threshold_idx, len(sorted_scores) - 1))
        threshold_abstention = sorted_scores[threshold_idx]
        answered_mask_abstention, _ = get_answered_deflected_masks(
            df_valid['deflection_score'], deflection_mechanism, threshold_abstention
        )
    
    if answered_mask_abstention.sum() > 0:
        risk_at_ct_abstention = 1 - df_valid[answered_mask_abstention]['is_correct'].mean()
    else:
        risk_at_ct_abstention = 0.0
    
    results[f'K={K}'] = {
        'real_coverage': real_coverage,
        'recall_at_k': recall_at_k,
        'risk_at_cr': risk_at_cr,
        'risk_at_ct_gt': risk_at_ct_gt,
        'risk_at_ct_abstention': risk_at_ct_abstention,
        'jaccard_index': jaccard_index,
        'strict_rag_accuracy': strict_rag_accuracy,
        'strict_rag_risk': strict_rag_risk,
        'strict_rag_correct_count': strict_rag_correct_count,
        'strict_rag_total_count': strict_rag_total_count,
        # Deflection metrics: GT-in-context classifier
        'deflection_precision': precision,
        'deflection_recall': recall,
        'deflection_f1': deflection_f1,
        'deflection_accuracy': accuracy,
        # Backward-compatible fields
        'correct_deflection': correct_deflection,
        'incorrect_deflection': incorrect_deflection
    }
    
    # Save other statistics to file (but don't print)
    output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else "."
    
    # Calculate other statistics silently
    gt_coverage_rate = (df['gt_passage_in_zidx'] != -1).mean()
    
    if len(df_valid) > 0:
        y_true = df_valid['has_gt_passage'].astype(int).values
        y_scores = df_valid['deflection_score'].values

        # Calculate ROC curve and AUC
        # For weighted_prior_logits, we need to reverse the direction (lower deflection prob = more relevant)
        # For max_z0_logits, lower score (z0 not max) = more relevant
        if deflection_mechanism == 'max_prior_logits':
            sorted_indices = np.argsort(y_scores)[::-1]  # Higher score = more relevant
        elif deflection_mechanism == 'weighted_prior_logits':
            sorted_indices = np.argsort(y_scores)  # Lower deflection prob = more relevant
        else:  # max_z0_logits
            sorted_indices = np.argsort(y_scores)  # Lower score (z0 not max) = more relevant
        
        y_true_sorted = y_true[sorted_indices]
        y_scores_sorted = y_scores[sorted_indices]

        unique_scores = np.unique(y_scores_sorted)
        if deflection_mechanism == 'max_prior_logits':
            thresholds_roc = np.concatenate([[y_scores_sorted.max() + 1], unique_scores, [y_scores_sorted.min() - 1]])
        elif deflection_mechanism == 'weighted_prior_logits':
            thresholds_roc = np.concatenate([[y_scores_sorted.min() - 1], unique_scores, [y_scores_sorted.max() + 1]])
        else:  # max_z0_logits
            thresholds_roc = np.concatenate([[y_scores_sorted.min() - 1], unique_scores, [y_scores_sorted.max() + 1]])

        tpr_list = []
        fpr_list = []

        for threshold in thresholds_roc:
            if deflection_mechanism == 'max_prior_logits':
                y_pred = (y_scores >= threshold).astype(int)
            elif deflection_mechanism == 'weighted_prior_logits':
                y_pred = (y_scores <= threshold).astype(int)  # Lower deflection prob = predicted present
            else:  # max_z0_logits
                y_pred = (y_scores <= threshold).astype(int)  # Lower score (z0 not max) = predicted present
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        tpr_array = np.array(tpr_list)
        fpr_array = np.array(fpr_list)

        sort_indices = np.argsort(fpr_array)
        fpr_sorted = fpr_array[sort_indices]
        tpr_sorted = tpr_array[sort_indices]

        auc = np.trapz(tpr_sorted, fpr_sorted)

        # Calculate risk vs coverage curve
        thresholds = np.linspace(df_valid['deflection_score'].min(), df_valid['deflection_score'].max(), 1000)
        coverage_list = []
        risk_list = []

        for threshold in thresholds:
            answered_mask, _ = get_answered_deflected_masks(
                df_valid['deflection_score'], deflection_mechanism, threshold
            )
            
            coverage = answered_mask.mean()
            
            if coverage > 0:
                answered_df = df_valid[answered_mask]
                risk = 1 - answered_df['is_correct'].mean()
            else:
                risk = 0
            
            coverage_list.append(coverage)
            risk_list.append(risk)

        coverage_array = np.array(coverage_list)
        risk_array = np.array(risk_list)

        # Save detailed results to JSON
        detailed_results = {
            "dataset_type": dataset_type,
            "deflection_mechanism": deflection_mechanism,
            "overall_score": float(overall_score),
            "gt_passage_coverage_rate": float(gt_coverage_rate),
            "gt_passage_coverage_percent": float(gt_coverage_rate * 100),
            "total_samples": int(len(df)),
            "valid_samples": int(len(df_valid)),
            "roc_auc": float(auc),
            "overall_accuracy": float(df_valid['is_correct'].mean()),
            "strict_rag_overall_accuracy": float(strict_rag_accuracy),
            "strict_rag_overall_risk": float(strict_rag_risk),
            "k_metrics": results
        }

        results_path = f"{output_dir}/prior_logits_abstention_analysis_{deflection_mechanism}.json"
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
    else:
        auc = 0.0
        detailed_results = {
            "dataset_type": dataset_type,
            "deflection_mechanism": deflection_mechanism,
            "overall_score": float(overall_score),
            "gt_passage_coverage_rate": float(gt_coverage_rate),
            "total_samples": int(len(df)),
            "valid_samples": 0,
            "strict_rag_overall_accuracy": 0.0,
            "strict_rag_overall_risk": 1.0,
            "k_metrics": results
        }
        results_path = f"{output_dir}/prior_logits_abstention_analysis_{deflection_mechanism}.json"
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
    
    return results, dataset_type, overall_score

def shorten_experiment_name(exp_name, max_length=40):
    """Shorten experiment name for display"""
    if len(exp_name) <= max_length:
        return exp_name
    # Try to keep the last part which usually contains important info
    parts = exp_name.split('-')
    if len(parts) > 1:
        # Keep last few parts
        shortened = '-'.join(parts[-3:])
        if len(shortened) <= max_length:
            return shortened
    # If still too long, truncate from the end
    return exp_name[:max_length-3] + '...'

def load_infoseek_reference():
    """Load Infoseek reference data once."""
    sys.path.append("./third_party/infoseek_eval")
    from infoseek_eval import evaluate_by_example, load_jsonl, prepare_qid2example  # noqa: F401
    
    reference_path = "third_party/infoseek_eval/infoseek/infoseek_val.jsonl"
    reference_qtype_path = "third_party/infoseek_eval/infoseek/infoseek_val_qtype.jsonl"
    reference = load_jsonl(reference_path)
    reference_qtype = load_jsonl(reference_qtype_path) if reference_qtype_path is not None else None
    qid2example = prepare_qid2example(reference, reference_qtype)
    return reference, reference_qtype, qid2example


def main():
    parser = argparse.ArgumentParser(description='Analyze prior logits abstention for multiple experiments')
    parser.add_argument('--experiments', nargs='+', required=True,
                        help='List of CSV file paths for experiments')
    parser.add_argument('--Ks', nargs='+', type=int, required=True,
                        help='List of K values corresponding to each experiment (must match order of --experiments)')
    parser.add_argument('--dataset', type=str, choices=['EVQA', 'Infoseek'], default=None,
                        help='Dataset type (auto-detected if not specified)')
    parser.add_argument('--deflection-mechanism', type=str, choices=['max_prior_logits', 'weighted_prior_logits', 'max_z0_logits'],
                        default='max_prior_logits',
                        help='Deflection mechanism: max_prior_logits (default), weighted_prior_logits, or max_z0_logits')
    parser.add_argument('--report_csv_path', type=str, default=None,
                        help='Path to save CSV report (default: analysis/output/prior_logits_abstention_report_{mechanism}.csv)')
    
    args = parser.parse_args()
    
    # Validate that Ks and experiments have the same length
    if len(args.Ks) != len(args.experiments):
        raise ValueError(f"Number of K values ({len(args.Ks)}) must match number of experiments ({len(args.experiments)})")
    
    # Load Infoseek reference data once if needed
    reference = None
    reference_qtype = None
    qid2example = None
    if args.dataset == 'Infoseek':
        reference, reference_qtype, qid2example = load_infoseek_reference()
    
    # Process all experiments
    all_results = {}
    all_overall_scores = {}
    dataset_type = None
    
    for csv_path, K in zip(args.experiments, args.Ks):
        exp_name = os.path.basename(os.path.dirname(csv_path))
        print(f"Processing: {exp_name} (K={K})")
        
        results, detected_type, overall_score = process_experiment(
            csv_path, 
            K,
            dataset_type=args.dataset,
            reference=reference,
            reference_qtype=reference_qtype,
            qid2example=qid2example,
            deflection_mechanism=args.deflection_mechanism
        )
        
        if dataset_type is None:
            dataset_type = detected_type
            # Load reference data if Infoseek
            if dataset_type == 'Infoseek' and reference is None:
                reference, reference_qtype, qid2example = load_infoseek_reference()
        
        all_results[exp_name] = results
        all_overall_scores[exp_name] = overall_score
    
    # Build DataFrame for results
    table_data = []
    for exp_name, results in all_results.items():
        overall_score = all_overall_scores[exp_name]
        # Each experiment now has only one K value
        if len(results) == 0:
            print(f"[warn] No results for experiment: {exp_name}")
            continue
        # Get the K key (should be only one)
        k_key = list(results.keys())[0]
        r = results[k_key]
        try:
            K = int(k_key.split('=')[1])
        except (IndexError, ValueError) as e:
            print(f"[error] Failed to parse K from key '{k_key}' for experiment {exp_name}: {e}")
            continue
        
        table_data.append({
            'Experiment': exp_name,
            'K': K,
            'Overall Score': overall_score,
            'Real Coverage C_r (%)': r['real_coverage'] * 100,
            'Recall@K C_t (%)': r['recall_at_k'] * 100,
            'Risk@C_r (%)': r['risk_at_cr'] * 100,
            'Risk@C_t(GT) (%)': r['risk_at_ct_gt'] * 100,
            'Risk@C_t(Abstention) (%)': r['risk_at_ct_abstention'] * 100,
            'StrictRAG Risk (%)': r['strict_rag_risk'] * 100,
            'Jaccard Index (C_gt ∩ C_r)': r['jaccard_index'],
            'Deflection Precision (%)': r['deflection_precision'] * 100,
            'Deflection Recall (%)': r['deflection_recall'] * 100,
            'Deflection F1': r['deflection_f1'],
            'Deflection Accuracy (%)': r['deflection_accuracy'] * 100,
            'StrictRAG Accuracy (%)': r['strict_rag_accuracy'] * 100
        })
    
    df_report = pd.DataFrame(table_data)
    if df_report.empty:
        print("No results to display.")
        print(f"Processed {len(all_results)} experiments, but no valid results were generated.")
        print("This may indicate:")
        print("  - All experiments had empty or invalid deflection scores")
        print("  - All experiments had empty prior_logits data")
        print("  - Check the JSON files in experiment directories for detailed error information")
        return
    
    # Print table using tabulate (with shortened experiment names for display)
    print("\n" + "="*150)
    print(f"{dataset_type}")
    print("="*150)
    
    # Create display version with shortened names
    df_display = df_report.copy()
    df_display['Experiment'] = df_display['Experiment'].apply(lambda x: shorten_experiment_name(x))
    
    # Sort by K and then by Overall Score
    df_display = df_display.sort_values(['K', 'Overall Score'], ascending=[True, False])
    
    if HAS_TABULATE:
        print(tabulate(df_display, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
    else:
        print(df_display.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Save CSV report
    if args.report_csv_path is None:
        # Default path with mechanism suffix
        os.makedirs('analysis/output', exist_ok=True)
        csv_path = f'analysis/output/prior_logits_abstention_report_{args.deflection_mechanism}.csv'
    else:
        csv_path = args.report_csv_path
        # Add mechanism suffix before file extension
        base, ext = os.path.splitext(csv_path)
        csv_path = f"{base}_{args.deflection_mechanism}{ext}"
        # Create directory if it doesn't exist
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
    
    # Save full DataFrame with original experiment names
    df_report.to_csv(csv_path, index=False)
    print(f"\nReport saved to: {csv_path}")

if __name__ == '__main__':
    main()
