#!/usr/bin/env python3
"""
analyze_ablation.py - Parses ablation JSON results and generates NeurIPS-ready plots.
Creates:
1. Dual-axis line plot showing Time and Parameter scaling vs. Ranks
2. Heatmap of computational cost over R_w and R_a
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# NeurIPS styling
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn")
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})

def parse_results(results_dir):
    json_files = glob.glob(os.path.join(results_dir, "stats_*.json"))
    data = []
    
    for jf in json_files:
        with open(jf, 'r') as f:
            js = json.load(f)
            data.append(js)
            
    df = pd.DataFrame(data)
    if df.empty:
        print(f"No JSON files found in {results_dir}")
    return df

def plot_dual_axis_scaling(df, output_path, dataset_name):
    """Line plot showing Time and Parameter scaling against R_w (holding R_a fixed)"""
    if df.empty: return
    
    # Filter where R_a is fixed to standard (e.g., 10) to show R_w scaling
    baseline_Ra = 10
    df_plot = df[df['R_a'] == baseline_Ra].sort_values('R_w')
    
    if df_plot.empty:
        df_plot = df.sort_values('R_w')
    
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    
    color1 = '#1f77b4' # Blue
    color2 = '#d62728' # Red
    
    # Axis 1: Time
    ax1.set_xlabel('Contemporaneous Tensor Rank ($R_w$)')
    ax1.set_ylabel('Time per Window (s)', color=color1)
    lns1 = ax1.plot(df_plot['R_w'], df_plot['avg_time_sec'], marker='o', color=color1, linewidth=2, label='Computation Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Axis 2: Parameter Count
    ax2 = ax1.twinx()
    ax2.set_ylabel('Tucker Parameter Count', color=color2)
    lns2 = ax2.plot(df_plot['R_w'], df_plot['param_count'], marker='s', color=color2, linewidth=2, linestyle='--', label='Parameters')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Emphasize cubic scaling of Time vs linear scaling of Parameters
    ax1.set_title(f'Computational Complexity Scaling on {dataset_name}')
    
    # Combined legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def plot_heatmap(df, output_path, dataset_name):
    """Heatmap of time cost across R_w and R_a matrix"""
    if df.empty: return
    
    pivot_df = df.pivot(index='R_w', columns='R_a', values='avg_time_sec')
    
    # If not a full grid, we might have NaNs, so fill them or just let seaborn handle it
    plt.figure(figsize=(5, 4))
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Time per Window (s)'})
    
    plt.title(f'Time Cost Matrix ({dataset_name})')
    plt.xlabel('Lagged Tensor Rank ($R_a$)')
    plt.ylabel('Contemporaneous Tensor Rank ($R_w$)')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='../results/ablation', help="Base directory of ablation results")
    parser.add_argument('--output_dir', type=str, default='../paper_plots', help="Directory to save plots")
    args = parser.parse_args()
    
    # Respect absolute paths; otherwise resolve relative to CWD
    results_base = Path(args.results_dir).resolve()
    plots_dir = Path(args.output_dir).resolve()
    os.makedirs(plots_dir, exist_ok=True)
    
    for dataset in ['msl', 'highd']:
        res_dir = results_base / dataset
        df = parse_results(str(res_dir))
        
        if not df.empty:
            label = "MSL (d=55)" if dataset == 'msl' else "High-D Synthetic (d=500)"
            
            # Line plot target
            plot_dual_axis_scaling(df, str(plots_dir / f"ablation_scaling_{dataset}.pdf"), label)
            
            # Heatmap target
            plot_heatmap(df, str(plots_dir / f"ablation_heatmap_{dataset}.pdf"), label)

if __name__ == "__main__":
    main()
