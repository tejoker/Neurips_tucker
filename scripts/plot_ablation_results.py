#!/usr/bin/env python3
"""
NeurIPS-Ready Plotting Script for Ablation Study
Generates a bar chart comparing F1 metrics across architectural variants
and a Pareto Frontier scatter plot for Time vs F1 Score.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
            # Make sure eval metrics exist
            if 'f1' in js:
                data.append(js)
            else:
                print(f"Warning: No F1 metrics found in {jf}")
            
    df = pd.DataFrame(data)
    if df.empty:
        print(f"No fully evaluated JSON files found in {results_dir}")
    return df

def plot_architectural_ablation(df, output_path):
    """Bar chart comparing F1 scores across architectural variants"""
    if df.empty: return
    
    # We want to compare identifiers: baseline, no_smoothness, etc.
    if 'identifier' not in df.columns:
        return
        
    plt.figure(figsize=(6, 4))
    
    # Sort or manually order for better presentation
    order = ['baseline', 'no_smoothness', 'no_core_sparsity', 'no_orthogonality', 'dense_tucker_disabled']
    # Filter only available variants
    avail_order = [v for v in order if v in df['identifier'].values]
    
    # Friendly labels
    labels = {
        'baseline': 'Tucker-CAM (Full)',
        'no_smoothness': 'w/o Smoothness\nPenalty',
        'no_core_sparsity': 'w/o Core\nSparsity',
        'no_orthogonality': 'w/o Orthogonality',
        'dense_tucker_disabled': 'No Tucker\n(Dense)'
    }
    
    plot_df = df[df['identifier'].isin(avail_order)].copy()
    plot_df['Label'] = plot_df['identifier'].map(labels)
    
    # Determine the order in the plot
    order_labels = [labels[v] for v in avail_order]
    
    colors = ['#1f77b4' if l == 'Tucker-CAM (Full)' else '#ff7f0e' for l in order_labels]
    
    ax = sns.barplot(
        data=plot_df, 
        x='Label', 
        y='f1', 
        order=order_labels,
        palette=colors,
        edgecolor='black'
    )
    
    # Add value annotations
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Prediction Accuracy by Architectural Component')
    plt.ylabel('F1 Score (Structural Connectivity)')
    plt.xlabel('')
    plt.ylim(0, max(plot_df['f1']) * 1.2 if max(plot_df['f1']) > 0 else 1.0)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def plot_pareto_frontier(df, output_path):
    """Scatter plot showing Trade-off between F1 Score and Computation Time"""
    if df.empty or 'f1' not in df.columns or 'avg_time_sec' not in df.columns: return

    plt.figure(figsize=(5, 4))
    
    labels = {
        'baseline': 'Tucker-CAM (Full)',
        'no_smoothness': 'w/o Smoothness Penalty',
        'no_core_sparsity': 'w/o Core Sparsity',
        'no_orthogonality': 'w/o Orthogonality',
        'dense_tucker_disabled': 'No Tucker (Dense Baseline)'
    }
    
    markers = {
        'baseline': '*',
        'no_smoothness': 'o',
        'no_core_sparsity': 's',
        'no_orthogonality': 'D',
        'dense_tucker_disabled': 'X'
    }
    
    colors = {
        'baseline': '#1f77b4',
        'no_smoothness': '#ff7f0e',
        'no_core_sparsity': '#2ca02c',
        'no_orthogonality': '#d62728',
        'dense_tucker_disabled': '#9467bd'
    }

    # Plot each configuration
    for idx, row in df.iterrows():
        ident = row['identifier']
        if ident in labels:
            label = labels[ident]
            marker = markers.get(ident, 'o')
            color = colors.get(ident, '#333333')
            size = 200 if ident == 'baseline' else 80
            
            plt.scatter(
                row['avg_time_sec'], 
                row['f1'], 
                marker=marker, 
                color=color, 
                s=size, 
                edgecolor='black', 
                label=label,
                zorder=3 if ident == 'baseline' else 2
            )
            
    # Add pareto frontier line for visual guide (optional, here we just plot points)
    plt.grid(True, linestyle='--', alpha=0.6, zorder=1)
    
    plt.title('Efficiency vs. Accuracy Pareto Frontier')
    plt.xlabel('Average Computation Time (s)')
    plt.ylabel('F1 Score')
    
    # Handle legends carefully to avoid duplicates
    handles, legends = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(legends, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', fancybox=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True, help="Directory containing JSON metrics")
    parser.add_argument('--output_dir', type=str, default='paper_plots/ablation', help="Where to save PDFs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    df = parse_results(args.results_dir)
    
    if not df.empty:
        plot_architectural_ablation(df, os.path.join(args.output_dir, "ablation_metrics_bar.pdf"))
        plot_pareto_frontier(df, os.path.join(args.output_dir, "ablation_pareto_scatter.pdf"))
    else:
        print("No valid results found. Cannot plot.")

if __name__ == "__main__":
    main()
