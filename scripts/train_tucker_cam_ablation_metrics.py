#!/usr/bin/env python3
"""
Tucker-CAM Ablation Study (Metrics-Driven)
Runs isolated training with architectural toggles and parameterized R_w and R_a, 
outputting thresholded predicted edge graphs for NeurIPS F1 evaluation.
"""

import os
import sys
import argparse
import time
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Use the duplicated, modifiable ablation scripts
from ablation_dynotears_tucker_cam import from_pandas_dynamic_tucker_cam

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_tucker_params_count(d, p, n_knots, rank_w, rank_a, disable_tucker=False):
    K = n_knots + 3  # cubic b-spline degree
    if disable_tucker:
        # Dense parameters (Tucker deactivated)
        return d * d * K + d * d * p * K
    else:
        # W Core (r,r,r) + 3 Factor Matrices
        w_params = rank_w**3 + 2 * d * rank_w + K * rank_w
        # A Core (r,r,r,r) + 4 Factor Matrices 
        a_params = rank_a**4 + 2 * d * rank_a + p * rank_a + K * rank_a
        return w_params + a_params

def run_ablation(data_np, var_names, p, window_size, stride, 
                 R_w, R_a, w_threshold,
                 disable_tucker=False, disable_smoothness=False, 
                 disable_core_sparsity=False, disable_orthogonality=False,
                 max_windows=None):
    
    n_samples = len(data_np)
    num_windows = (n_samples - window_size) // stride + 1
    if max_windows:
        num_windows = min(num_windows, max_windows)

    times = []
    
    # If Tucker is disabled, force full rank
    actual_Rw = data_np.shape[1] if disable_tucker else R_w
    actual_Ra = data_np.shape[1] if disable_tucker else R_a
    
    params = get_tucker_params_count(data_np.shape[1], p, 5, actual_Rw, actual_Ra, disable_tucker)
    edges_list = []

    logger.info(f"Running Ablation on {num_windows} windows.")
    logger.info(f" - R_w={actual_Rw}, R_a={actual_Ra}")
    logger.info(f" - Tucker Disabled: {disable_tucker}")
    logger.info(f" - Smoothness Disabled: {disable_smoothness}")
    logger.info(f" - Core Sparsity Disabled: {disable_core_sparsity}")
    logger.info(f" - Orthogonality Disabled: {disable_orthogonality}")
    
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        chunk = data_np[start_idx:end_idx]
        df = pd.DataFrame(chunk, columns=var_names)
        
        t0 = time.time()
        
        try:
            edges = from_pandas_dynamic_tucker_cam(
                df,
                p=p,
                rank_w=actual_Rw,
                rank_a=actual_Ra,
                n_knots=5,
                lambda_smooth=0.01,
                lambda_core=0.01,
                lambda_orth=0.001,
                disable_tucker=disable_tucker,
                disable_smoothness=disable_smoothness,
                disable_core_sparsity=disable_core_sparsity,
                disable_orthogonality=disable_orthogonality,
                max_iter=100,
                lr=0.01,
                w_threshold=w_threshold,
                device='cpu', # CPU is safest for rigorous timing memory benchmarks
                return_indices=True
            )
            
            t1 = time.time()
            times.append(t1 - t0)
            
            # Format edges for evaluation (only keep those above threshold)
            for edge in edges:
                src, tgt, lag, weight = edge
                edges_list.append([i, src, tgt, lag, abs(weight)])
                
            if (i+1) % 10 == 0:
                logger.info(f"Processed {i+1}/{num_windows} windows. Avg Time: {np.mean(times):.2f}s, Params: {params:,}")
                
        except Exception as e:
            logger.error(f"Window {i} failed: {e}")
            import traceback
            traceback.print_exc()
            
    return edges_list, np.mean(times), params

def main():
    parser = argparse.ArgumentParser(description="Tucker-CAM Real Metrics Ablation")
    parser.add_argument('--dataset', type=str, required=True, help="Path to data .npy")
    parser.add_argument('--columns', type=str, required=True, help="Path to columns .npy")
    parser.add_argument('--lags', type=str, required=True, help="Path to lags .npy")
    parser.add_argument('--R_w', type=int, default=20, help="Contemporaneous rank")
    parser.add_argument('--R_a', type=int, default=10, help="Time-lagged rank")
    
    # Architectural Ablations
    parser.add_argument('--disable_tucker', action='store_true', help="Dense baseline")
    parser.add_argument('--disable_smoothness', action='store_true', help="Ablate smoothness")
    parser.add_argument('--disable_core_sparsity', action='store_true', help="Ablate core sparsity")
    parser.add_argument('--disable_orthogonality', action='store_true', help="Ablate orthogonality")
    
    parser.add_argument('--w_threshold', type=float, default=0.1, help="Edge weight threshold to consider it a discovered edge")
    parser.add_argument('--max_windows', type=int, default=None, help="Subset windows")
    parser.add_argument('--output_dir', type=str, default='results/ablation_metrics')
    parser.add_argument('--identifier', type=str, default='baseline', help="Experiment name")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_np = np.load(args.dataset)
    var_names = np.load(args.columns, allow_pickle=True).tolist()
    
    lags_raw = np.load(args.lags, allow_pickle=True)
    p = 5
    if isinstance(lags_raw, dict): p = max(lags_raw.values())
    elif isinstance(lags_raw, np.ndarray) and lags_raw.shape == (): p = max(lags_raw.item().values())
    elif isinstance(lags_raw, np.ndarray) and len(lags_raw) > 0: p = 5 

    logger.info(f"Loaded dataset {args.dataset} (d={data_np.shape[1]}), using p={p}")

    edges, avg_time, param_count = run_ablation(
        data_np, var_names, p, window_size=100, stride=10,
        R_w=args.R_w, R_a=args.R_a, w_threshold=args.w_threshold,
        disable_tucker=args.disable_tucker,
        disable_smoothness=args.disable_smoothness,
        disable_core_sparsity=args.disable_core_sparsity,
        disable_orthogonality=args.disable_orthogonality,
        max_windows=args.max_windows
    )
    
    edges_csv = os.path.join(args.output_dir, f"predicted_edges_{args.identifier}.csv")
    df_edges = pd.DataFrame(edges, columns=['window_idx', 'i', 'j', 'lag', 'weight'])
    df_edges.to_csv(edges_csv, index=False)
    
    stats = {
        'identifier': args.identifier,
        'R_w': args.R_w,
        'R_a': args.R_a,
        'disable_tucker': args.disable_tucker,
        'disable_smoothness': args.disable_smoothness,
        'disable_core_sparsity': args.disable_core_sparsity,
        'disable_orthogonality': args.disable_orthogonality,
        'avg_time_sec': avg_time,
        'param_count': param_count,
        'dataset_dim': data_np.shape[1],
        'edges_file': edges_csv
    }
    
    json_path = os.path.join(args.output_dir, f"stats_{args.identifier}.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Finished {args.identifier}. Avg Time: {avg_time:.2f}s, Params: {param_count:,}")

if __name__ == "__main__":
    main()
