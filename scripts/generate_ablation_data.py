#!/usr/bin/env python3
"""
Generates synthetic multivariate time-series data for the Tensor Rank Ablation study.
Outputs .npy files for data, columns, and an empty lags dict for compatibility.
"""
import os
import argparse
import numpy as np

def generate_data(d, t, out_prefix):
    # Generate random normally distributed data
    data = np.random.randn(t, d).astype(np.float32)
    
    # Inject synthetic causal relationships (fixed edge weight 0.8)
    # We create a random DAG for contemporaneous edges (W) 
    # and a random lagged graph (A)
    true_edges = []
    
    # 1. Random Contemporaneous DAG (upper triangular to ensure acyclicity)
    # Edge probability ~ 5%
    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < 0.05:
                # Add relationship: j depends on i
                data[:, j] += 0.8 * data[:, i]
                true_edges.append({'i': i, 'j': j, 'lag': 0, 'weight': 0.8})
                
    # 2. Random Lagged edges (Lag 1)
    # Edge probability ~ 5%
    for i in range(d):
        for j in range(d):
            if np.random.rand() < 0.05:
                # Add relationship: j(t) depends on i(t-1)
                data[1:, j] += 0.8 * data[:-1, i]
                true_edges.append({'i': i, 'j': j, 'lag': 1, 'weight': 0.8})
                
    # Add some noise back in
    data += np.random.randn(t, d) * 0.1
    
    # Generate column names
    cols = np.array([f"Sensor_{i}" for i in range(d)], dtype=object)
    
    # Generate default lags (p=5 for all)
    lags = {c: 5 for c in cols}
    
    # Save files
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    np.save(f"{out_prefix}_data.npy", data)
    np.save(f"{out_prefix}_columns.npy", cols)
    np.save(f"{out_prefix}_lags.npy", lags)
    
    import pandas as pd
    df_edges = pd.DataFrame(true_edges)
    df_edges.to_csv(f"{out_prefix}_true_edges.csv", index=False)
    
    print(f"Generated {d}-dim dataset with {t} timesteps")
    print(f"Saved {len(true_edges)} ground truth edges to {out_prefix}_true_edges.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, required=True, help="Dimensionality")
    parser.add_argument("--t", type=int, required=True, help="Timesteps")
    parser.add_argument("--output", type=str, required=True, help="Output prefix")
    args = parser.parse_args()
    generate_data(args.d, args.t, args.output)
