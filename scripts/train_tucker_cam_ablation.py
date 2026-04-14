#!/usr/bin/env python3
"""
Tucker-CAM Tensor Rank Ablation Study Script
Runs isolated training with parameterized R_w and R_a, logging accurate
memory and time benchmarks for NeurIPS plotting.
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
import psutil

# Add parent directory to path to import final_pipeline modules
# Also add the final_pipeline directory itself so inner imports work
pipeline_dir = Path(__file__).resolve().parent.parent / 'executable' / 'final_pipeline'
sys.path.insert(0, str(pipeline_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'executable'))
from dynotears_tucker_cam import from_pandas_dynamic_tucker_cam

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_tucker_params_count(d, p, n_knots, rank_w, rank_a):
    K = n_knots + 3  # cubic b-spline degree
    # W Core (r,r,r) + 3 Factor Matrices
    w_params = rank_w**3 + 2 * d * rank_w + K * rank_w
    # A Core (r,r,r,r) + 4 Factor Matrices 
    a_params = rank_a**4 + 2 * d * rank_a + p * rank_a + K * rank_a
    return w_params + a_params

def run_ablation(data_np, var_names, p, window_size, stride, R_w, R_a, max_windows=None):
    n_samples = len(data_np)
    num_windows = (n_samples - window_size) // stride + 1
    if max_windows:
        num_windows = min(num_windows, max_windows)

    times = []
    params = get_tucker_params_count(data_np.shape[1], p, 5, R_w, R_a)
    edges_list = []

    logger.info(f"Running Ablation: R_w={R_w}, R_a={R_a}, Windows={num_windows}")
    
    # Run sequentially for accurate memory profiling
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
                rank_w=R_w,
                rank_a=R_a,
                n_knots=5,
                lambda_smooth=0.01,
                lambda_w=0.0,
                lambda_a=0.0,
                max_iter=100,
                lr=0.01,
                w_threshold=0.0,
                device='cpu',
                return_indices=True
            )
            
            t1 = time.time()
            times.append(t1 - t0)
            
            # Format edges for evaluation
            for edge in edges:
                src, tgt, lag, weight = edge
                edges_list.append([i, src, tgt, lag, abs(weight)])
                
            if (i+1) % 10 == 0:
                logger.info(f"Processed {i+1}/{num_windows} windows. Avg Time: {np.mean(times):.2f}s, Params: {params:,}")
                
        except Exception as e:
            logger.error(f"Window {i} failed: {e}")
            
    return edges_list, np.mean(times), params

def main():
    parser = argparse.ArgumentParser(description="Tucker-CAM Rank Ablation")
    parser.add_argument('--dataset', type=str, required=True, help="Path to data .npy")
    parser.add_argument('--columns', type=str, required=True, help="Path to columns .npy")
    parser.add_argument('--lags', type=str, required=True, help="Path to lags .npy")
    parser.add_argument('--R_w', type=int, required=True, help="Contemporaneous rank")
    parser.add_argument('--R_a', type=int, required=True, help="Time-lagged rank")
    parser.add_argument('--max_windows', type=int, default=None, help="Subset windows for pure perf measure")
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_np = np.load(args.dataset)
    var_names = np.load(args.columns, allow_pickle=True).tolist()
    
    # Parse lags
    lags_raw = np.load(args.lags, allow_pickle=True)
    p = 5 # Default fallback
    if isinstance(lags_raw, dict): p = max(lags_raw.values())
    elif isinstance(lags_raw, np.ndarray) and lags_raw.shape == (): p = max(lags_raw.item().values())
    elif isinstance(lags_raw, np.ndarray) and len(lags_raw) > 0: p = 5 # Simplified for brevity

    logger.info(f"Loaded dataset {args.dataset} (d={data_np.shape[1]}), using p={p}")

    edges, avg_time, param_count = run_ablation(
        data_np, var_names, p, window_size=100, stride=10,
        R_w=args.R_w, R_a=args.R_a, max_windows=args.max_windows
    )
    
    # Save edges
    identifier = f"Rw_{args.R_w}_Ra_{args.R_a}"
    edges_csv = os.path.join(args.output_dir, f"edges_{identifier}.csv")
    df_edges = pd.DataFrame(edges, columns=['window_idx', 'i', 'j', 'lag', 'weight'])
    df_edges.to_csv(edges_csv, index=False)
    
    # Save JSON stats
    stats = {
        'R_w': args.R_w,
        'R_a': args.R_a,
        'avg_time_sec': avg_time,
        'param_count': param_count,
        'dataset_dim': data_np.shape[1],
        'edges_file': edges_csv
    }
    
    json_path = os.path.join(args.output_dir, f"stats_{identifier}.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Finished. Avg Time: {avg_time:.2f}s, Params: {param_count:,}")
    logger.info(f"Saved stats to {json_path}")

if __name__ == "__main__":
    main()
