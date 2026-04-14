#!/usr/bin/env python3
"""
Ablation Studies for Tucker-CAM
Tests different variants to validate each component's contribution.

Variants:
1. Full Tucker-CAM (baseline)
2. No Tucker (full W tensor - only for small d)
3. Linear (no P-splines)
4. Single Metric (only s_abs)
5. L1 Sparsity (instead of Top-K)
6. Fixed Threshold (no adaptive)

Usage:
    python ablation_studies.py --dataset smd --entity machine-1-1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import time
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


ABLATION_VARIANTS = {
    'full': {
        'name': 'Tucker-CAM (Full)',
        'use_tucker': True,
        'use_psplines': True,
        'use_multi_metric': True,
        'use_topk': True,
        'use_adaptive_threshold': True
    },
    'no_tucker': {
        'name': 'No Tucker Decomposition',
        'use_tucker': False,
        'use_psplines': True,
        'use_multi_metric': True,
        'use_topk': True,
        'use_adaptive_threshold': True
    },
    'linear': {
        'name': 'Linear (No P-splines)',
        'use_tucker': True,
        'use_psplines': False,
        'use_multi_metric': True,
        'use_topk': True,
        'use_adaptive_threshold': True
    },
    'single_metric': {
        'name': 'Single Metric (s_abs only)',
        'use_tucker': True,
        'use_psplines': True,
        'use_multi_metric': False,
        'use_topk': True,
        'use_adaptive_threshold': True
    },
    'l1_sparsity': {
        'name': 'L1 Sparsity (no Top-K)',
        'use_tucker': True,
        'use_psplines': True,
        'use_multi_metric': True,
        'use_topk': False,
        'use_adaptive_threshold': True
    },
    'fixed_threshold': {
        'name': 'Fixed Threshold',
        'use_tucker': True,
        'use_psplines': True,
        'use_multi_metric': True,
        'use_topk': True,
        'use_adaptive_threshold': False
    }
}


def load_data(dataset_name='smd', entity='machine-1-1'):
    """Load dataset"""
    if dataset_name.lower() == 'smap':
        data_file = 'telemanom/golden_period_dataset_clean.csv'
        logger.info(f"Loading {dataset_name} from {data_file}")
        data = pd.read_csv(data_file, index_col=0)
        n = len(data)
        split_idx = n // 2
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
    elif dataset_name.lower() == 'smd':
        train_file = f'ServerMachineDataset/train/{entity}.txt'
        test_file = f'ServerMachineDataset/test/{entity}.txt'
        logger.info(f"Loading {dataset_name} ({entity}) from {train_file} & {test_file}")
        
        # Load SMD headerless CSVs
        train_data = pd.read_csv(train_file, header=None)
        test_data = pd.read_csv(test_file, header=None)
        
        # Assign dummy column names
        train_data.columns = [f"dim_{i}" for i in range(train_data.shape[1])]
        test_data.columns = [f"dim_{i}" for i in range(test_data.shape[1])]
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    return train_data, test_data

def point_adjustment(y_score, y_true, threshold):
    y_pred = (y_score >= threshold).astype(int)
    y_pred_pa = y_pred.copy()
    
    events = []
    start = None
    for i, val in enumerate(y_true):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(y_true) - 1))
        
    for start, end in events:
        if np.sum(y_pred[start : end + 1]) > 0:
            y_pred_pa[start : end + 1] = 1
    return y_pred_pa

def find_best_f1_pa(y_score, y_true, num_steps=100):
    if len(y_score) == 0: return 0.0
    from sklearn.metrics import precision_recall_fscore_support
    min_score, max_score = np.min(y_score), np.max(y_score)
    thresholds = [min_score] if min_score == max_score else np.linspace(min_score, max_score, num_steps)
    
    best_f1 = 0.0
    for th in thresholds:
        y_pred_pa = point_adjustment(y_score, y_true, th)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred_pa, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def run_ablation_variant(train_data, test_data, variant_config, variant_name):
    """Run a specific ablation variant"""
    logger.info(f"Running variant: {variant_config['name']}")
    
    # Set environment variables based on variant
    os.environ['USE_TUCKER_CAM'] = str(variant_config['use_tucker']).lower()
    os.environ['USE_PSPLINES'] = str(variant_config['use_psplines']).lower()
    os.environ['USE_MULTI_METRIC'] = str(variant_config['use_multi_metric']).lower()
    os.environ['USE_TOPK'] = str(variant_config['use_topk']).lower()
    os.environ['USE_ADAPTIVE_THRESHOLD'] = str(variant_config['use_adaptive_threshold']).lower()
    
    # Create variant directory
    variant_dir = Path(f'results/ablations/{variant_name}')
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    train_file = variant_dir / 'train.csv'
    test_file = variant_dir / 'test.csv'
    train_data.to_csv(train_file)
    test_data.to_csv(test_file)
    
    # Track resources
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    start_time = time.time()
    
    try:
        from executable.launcher import run_pipeline
        
        train_dir = variant_dir / 'train_results'
        test_dir = variant_dir / 'test_results'
        
        # Train
        logger.info(f"  Training...")
        success = run_pipeline(str(train_file), str(train_dir), resume=False)
        if not success:
            logger.error(f"  Training failed")
            return None
        
        # Test
        logger.info(f"  Testing...")
        success = run_pipeline(str(test_file), str(test_dir), resume=False)
        if not success:
            logger.error(f"  Testing failed")
            return None
        
        # Measure resources
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        peak_memory = end_memory  # Simplified
        elapsed_hours = (end_time - start_time) / 3600
        
        # Determine actual evaluation metrics if using SMD
        f1 = None
        if 'smd_labels_path' in variant_config:
            logger.info("  Running dual-metric anomaly detection...")
            import subprocess
            golden_w = train_dir / 'weights' / 'weights_enhanced.csv'
            test_w = test_dir / 'weights' / 'weights_enhanced.csv'
            
            # Fallback to standard weights.csv if enhanced isn't found
            if not golden_w.exists(): golden_w = train_dir / 'weights' / 'weights.csv'
            if not test_w.exists(): test_w = test_dir / 'weights' / 'weights.csv'
            
            det_out = variant_dir / 'anomaly_detection.csv'
            cmd_det = [
                sys.executable, 'executable/dual_metric_anomaly_detection.py',
                '--golden', str(golden_w),
                '--test', str(test_w),
                '--output', str(det_out)
            ]
            
            res = subprocess.run(cmd_det, capture_output=True, text=True)
            if res.returncode == 0 and det_out.exists():
                df_det = pd.read_csv(det_out)
                labels_pt = np.loadtxt(variant_config['smd_labels_path'], delimiter=',')
                
                num_windows = int(df_det['window_idx'].max()) + 1 if not df_det.empty else 0
                if num_windows > 0 and 'abs_score' in df_det.columns:
                    y_true = np.zeros(num_windows)
                    y_scores = np.zeros(num_windows)
                    
                    # Convert point labels to window labels
                    for w in range(num_windows):
                        start = w * 10
                        end = start + 100
                        if end > len(labels_pt): break
                        if np.any(labels_pt[start:end] == 1):
                            y_true[w] = 1
                            
                    for _, row in df_det.iterrows():
                        idx = int(row['window_idx'])
                        if idx < num_windows:
                            y_scores[idx] = row['abs_score']
                            
                    f1 = find_best_f1_pa(y_scores, y_true)
                    logger.info(f"  PA-F1 Score: {f1:.4f}")
                else:
                    logger.warning("  Detection output empty or missing columns")
            else:
                logger.error(f"  Anomaly detection script failed. STDOUT: {res.stdout[:200]}")
        
        # No fallback: if evaluation failed, report None (never fabricate scores)
        if f1 is None:
            logger.error("  F1 could not be computed (missing labels or detection failure)")
        
        return {
            'variant': variant_name,
            'name': variant_config['name'],
            'f1': f1,
            'memory_gb': peak_memory,
            'time_hours': elapsed_hours
        }
        
    except MemoryError:
        logger.error(f"  OOM Error!")
        return {
            'variant': variant_name,
            'name': variant_config['name'],
            'f1': None,
            'memory_gb': '>125',
            'time_hours': None
        }
    except Exception as e:
        logger.error(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Ablation studies for Tucker-CAM')
    parser.add_argument('--dataset', type=str, default='smd', choices=['smap', 'msl', 'smd'])
    parser.add_argument('--entity', type=str, default='machine-1-1', help='Specific entity for SMD')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Specific variants to run (default: all)')
    parser.add_argument('--output-dir', type=str, default='results/ablations')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ABLATION STUDIES FOR TUCKER-CAM")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info("")
    
    # Load data
    train_data, test_data = load_data(args.dataset, args.entity)
    logger.info(f"Data: train={len(train_data)}, test={len(test_data)}")
    
    # Store labels path if SMD for F1 calculation
    smd_labels_path = None
    if args.dataset.lower() == 'smd':
        smd_labels_path = f'ServerMachineDataset/test_label/{args.entity}.txt'
    
    # Select variants to run
    if args.variants:
        variants_to_run = {k: v for k, v in ABLATION_VARIANTS.items() if k in args.variants}
    else:
        variants_to_run = ABLATION_VARIANTS
    
    logger.info(f"Running {len(variants_to_run)} variants")
    logger.info("")
    
    # Run ablations
    results = []
    for variant_name, variant_config in variants_to_run.items():
        if smd_labels_path:
            variant_config['smd_labels_path'] = smd_labels_path
        result = run_ablation_variant(train_data, test_data, variant_config, variant_name)
        if result:
            results.append(result)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'ablation_results.csv', index=False)
    
    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("ABLATION RESULTS")
    logger.info("="*80)
    print(results_df.to_string(index=False))
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
