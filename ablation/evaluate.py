#!/usr/bin/env python3
"""
Evaluation utilities for Tucker-CAM ablation study.

Provides:
- Point-Adjusted F1 (PA-F1)
- Standard F1
- AUC-PR
- Multi-seed aggregation with mean/std
- Paired t-test for significance
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    auc,
)
from scipy import stats
from typing import Dict, List, Tuple, Optional


def point_adjust(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Point-adjustment protocol (Xu et al., 2018).

    If any point within a contiguous anomaly segment is predicted as anomalous,
    credit the entire segment as detected.
    """
    y_pred_pa = y_pred.copy()
    events = _extract_events(y_true)
    for start, end in events:
        if np.any(y_pred[start:end + 1] == 1):
            y_pred_pa[start:end + 1] = 1
    return y_pred_pa


def _extract_events(y_true: np.ndarray) -> List[Tuple[int, int]]:
    """Extract contiguous anomaly segments from binary label array."""
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
    return events


def compute_f1_at_threshold(
    y_score: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    use_pa: bool = True,
) -> Dict[str, float]:
    """Compute F1, precision, recall at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    if use_pa:
        y_pred = point_adjust(y_pred, y_true)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"f1": f1, "precision": p, "recall": r}


def find_best_f1(
    y_score: np.ndarray,
    y_true: np.ndarray,
    num_thresholds: int = 200,
    use_pa: bool = True,
) -> Dict[str, float]:
    """
    Sweep thresholds to find the one maximizing F1 (or PA-F1).

    Returns dict with best_f1, best_threshold, precision, recall.
    """
    if len(y_score) == 0 or np.all(y_true == 0) or np.all(y_true == 1):
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}

    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    if lo == hi:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": lo}

    thresholds = np.linspace(lo, hi, num_thresholds)
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": lo}

    for th in thresholds:
        metrics = compute_f1_at_threshold(y_score, y_true, th, use_pa=use_pa)
        if metrics["f1"] > best["f1"]:
            best = {**metrics, "threshold": float(th)}

    return best


def compute_auc_pr(y_score: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    if len(y_score) == 0 or np.all(y_true == 0) or np.all(y_true == 1):
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def full_evaluation(
    y_score: np.ndarray, y_true: np.ndarray, num_thresholds: int = 200
) -> Dict[str, float]:
    """
    Complete evaluation: PA-F1, standard F1, AUC-PR.

    Returns a flat dict with all metrics.
    """
    pa = find_best_f1(y_score, y_true, num_thresholds, use_pa=True)
    std = find_best_f1(y_score, y_true, num_thresholds, use_pa=False)
    auc_pr = compute_auc_pr(y_score, y_true)

    return {
        "pa_f1": pa["f1"],
        "pa_precision": pa["precision"],
        "pa_recall": pa["recall"],
        "pa_threshold": pa["threshold"],
        "std_f1": std["f1"],
        "std_precision": std["precision"],
        "std_recall": std["recall"],
        "std_threshold": std["threshold"],
        "auc_pr": auc_pr,
    }


def aggregate_seeds(
    seed_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across seeds.

    Returns dict mapping metric_name -> {mean, std, values}.
    """
    if not seed_results:
        return {}

    keys = seed_results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in seed_results if k in r]
        ci95 = 0.0
        if len(vals) > 1:
            sem = stats.sem(vals)
            ci95 = float(1.96 * sem)
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci95": ci95,
            "n": len(vals),
            "values": vals,
        }
    return agg


def paired_t_test(
    baseline_values: List[float], variant_values: List[float]
) -> Dict[str, float]:
    """
    Paired t-test between baseline (full model) and a variant.

    Returns t-statistic, p-value, and whether the difference is significant (p<0.05).
    """
    if len(baseline_values) != len(variant_values) or len(baseline_values) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False}

    t_stat, p_value = stats.ttest_rel(baseline_values, variant_values)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def format_result_table(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    baseline_key: str = "full",
) -> str:
    """
    Format aggregated results as a LaTeX-ready table string.

    all_results: variant_name -> metric_name -> {mean, std, values}
    """
    metrics = ["pa_f1", "std_f1", "auc_pr"]
    header = "Variant | PA-F1 | Std-F1 | AUC-PR | sig (vs full)"

    lines = [header, "-" * len(header)]

    baseline = all_results.get(baseline_key, {})

    for variant_name, agg in all_results.items():
        parts = [variant_name]
        sig_parts = []
        for m in metrics:
            if m in agg:
                mean = agg[m]["mean"]
                std = agg[m]["std"]
                parts.append(f"{mean:.3f} +/- {std:.3f}")
            else:
                parts.append("N/A")

            # Significance test against baseline
            if variant_name != baseline_key and m in agg and m in baseline:
                tt = paired_t_test(baseline[m]["values"], agg[m]["values"])
                sig_parts.append("*" if tt["significant"] else "")
            else:
                sig_parts.append("")

        sig_str = ",".join(sig_parts)
        parts.append(sig_str)
        lines.append(" | ".join(parts))

    return "\n".join(lines)
