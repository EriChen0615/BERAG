import sys
sys.path.append('./src')
sys.path.append('./src/evaluation')
import json
import pandas as pd
from vqa_datasets import load_vqa_dataset
from pprint import pprint
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tabulate import tabulate
import os

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--history_file", type=str, default=None)

    args = parser.parse_args()

    vqa_dataset  = load_vqa_dataset("EVQA", split="test", img_basedir='data/')
    df = vqa_dataset.to_pandas()
    dict_to_report = {}
    output_dir  = "/".join(args.history_file.split('/')[:-1])

    verify_doc_data = []
    with open(args.history_file, 'r') as f:
        verify_histories = json.load(f)
        verify_histories = verify_histories[len(verify_histories)-len(df):]
        for idx, turn in enumerate(verify_histories):
            gt_doc_ids = df.iloc[idx]['pos_item_ids']
            for doc_to_verify in turn[3]:
                if doc_to_verify['pass_verification']:
                    if any([doc_to_verify['evidence_title'] in gt_doc_id for gt_doc_id in gt_doc_ids]):
                        verify_doc_data.append((1, 1)) # True Positive
                    else:
                        verify_doc_data.append((1, 0)) # False Positive
                else:
                    if any([doc_to_verify['evidence_title'] in gt_doc_id for gt_doc_id in gt_doc_ids]):
                        verify_doc_data.append((0, 1)) # False Negative
                    else:
                        verify_doc_data.append((0, 0)) # True Negative
    
    verify_doc_df = pd.DataFrame(verify_doc_data, columns=['prediction', 'label'])
    # Compute Accuracy, Recall, Precision, F1 score
    accuracy = accuracy_score(verify_doc_df['label'], verify_doc_df['prediction'])
    recall = recall_score(verify_doc_df['label'], verify_doc_df['prediction'])
    precision = precision_score(verify_doc_df['label'], verify_doc_df['prediction'])
    f1 = f1_score(verify_doc_df['label'], verify_doc_df['prediction'])

    # Prepare the report
    metrics = {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1
    }

    # Pretty print using tabulate
    report_table = [
        ["Accuracy", accuracy],
        ["Recall", recall],
        ["Precision", precision],
        ["F1 Score", f1]
    ]

    print(tabulate(report_table, headers=["Metric", "Score"], tablefmt="grid"))

    # Save the metrics to a JSON file
    # os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'verification_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {output_file}")

