#!/usr/bin/env python3
"""
Evaluate Ablation Metrics
Calculates standard DAG discovery metrics (Precision, Recall, F1) by comparing 
predicted edges to ground truth edges.
"""

import os
import json
import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_graph(predicted_csv, true_csv, stats_json=None):
    df_pred = pd.read_csv(predicted_csv)
    df_true = pd.read_csv(true_csv)
    
    # We evaluate edge existence, regardless of weight
    # Create sets of (i, j, lag) tuples
    
    # Ground truth edges
    true_set = set()
    for _, row in df_true.iterrows():
        true_set.add((int(row['i']), int(row['j']), int(row['lag'])))
        
    # Predicted edges
    # For dynamic data windowing, we may predict the same edge in multiple windows.
    # For a robust ablation of the *graph structure*, we take the union across all windows.
    pred_set = set()
    for _, row in df_pred.iterrows():
        pred_set.add((int(row['i']), int(row['j']), int(row['lag'])))
        
    # Calculate True Positives, False Positives, False Negatives
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"Metrics against Ground Truth ({len(true_set)} edges):")
    print(f" - Predicted Edges: {len(pred_set)}")
    print(f" - True Positives:  {tp}")
    print(f" - False Positives: {fp}")
    print(f" - False Negatives: {fn}")
    print(f" - Precision:       {precision:.4f}")
    print(f" - Recall:          {recall:.4f}")
    print(f" - F1 Score:        {f1:.4f}")
    
    # Update JSON with metrics if requested
    if stats_json and os.path.exists(stats_json):
        with open(stats_json, 'r') as f:
            stats = json.load(f)
            
        stats['precision'] = precision
        stats['recall'] = recall
        stats['f1'] = f1
        stats['true_edges'] = len(true_set)
        stats['predicted_edges'] = len(pred_set)
        
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=4)
            
        print(f"Updated {stats_json} with evaluation metrics.")
        return stats
        
    return {'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted_csv', type=str, required=True, help="Predicted edges from ablation wrapper")
    parser.add_argument('--true_csv', type=str, required=True, help="Ground truth edges")
    parser.add_argument('--stats_json', type=str, default=None, help="JSON stats file to append metrics into")
    args = parser.parse_args()
    
    evaluate_graph(args.predicted_csv, args.true_csv, args.stats_json)
