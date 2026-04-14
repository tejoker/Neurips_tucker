#!/usr/bin/env python3
"""
Unified training interface for all ablation model variants.

Provides a single fit_model() function that:
1. Creates the appropriate model variant
2. Runs augmented Lagrangian optimization (acyclicity constraint)
3. Extracts weight matrices
4. Returns edge weights for anomaly scoring

All variants share the same optimization loop -- only the model differs.
"""

import sys
import time
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

# Import ablation model variants
from ablation.models.dense_cam import DenseCAMModel
from ablation.models.cp_cam import CPCAMModel
from ablation.models.linear_model import LinearSVARModel

# Import the Tucker model from scripts/ (already ablation-specific, not main pipeline)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ablation_cam_model_tucker import TuckerCAMModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS = {
    "full": {
        "name": "Tucker-CAM (Full)",
        "model_type": "tucker",
        "description": "Full model with all components enabled",
    },
    "no_tucker": {
        "name": "No Tucker (Dense)",
        "model_type": "dense",
        "description": "Full W/A tensors, no decomposition",
    },
    "linear": {
        "name": "Linear (No P-splines)",
        "model_type": "linear",
        "description": "Linear SVAR, no nonlinear basis functions",
    },
    "cp_decomposition": {
        "name": "CP Decomposition",
        "model_type": "cp",
        "description": "CP instead of Tucker factorization",
    },
    "single_metric": {
        "name": "Single Metric (s_abs only)",
        "model_type": "tucker",
        "description": "Same model, detection uses only s_abs",
        "detection_mode": "single",
    },
    "fixed_threshold": {
        "name": "Fixed Threshold",
        "model_type": "tucker",
        "description": "Same model, mean+3*std threshold (no adaptive)",
        "threshold_mode": "fixed",
    },
    "no_core_sparsity": {
        "name": "No Core Sparsity",
        "model_type": "tucker",
        "model_kwargs": {"disable_core_sparsity": True},
        "description": "Tucker without core tensor L1 penalty",
    },
    "no_orthogonality": {
        "name": "No Orthogonality",
        "model_type": "tucker",
        "model_kwargs": {"disable_orthogonality": True},
        "description": "Tucker without factor orthogonality penalty",
    },
    "no_smoothness": {
        "name": "No Smoothness",
        "model_type": "tucker",
        "model_kwargs": {"disable_smoothness": True},
        "description": "Tucker without P-spline smoothness penalty",
    },
    "l1_sparsity": {
        "name": "L1 Sparsity (no Top-K)",
        "model_type": "tucker",
        "description": "L1 penalty on W/A instead of Top-K pruning",
        "lambda_w": 0.01,
        "lambda_a": 0.01,
    },
}


def _compute_basis_matrix(n, n_knots, K, degree=3, device="cpu"):
    """Compute B-spline basis matrix (shared across model types)."""
    knots = np.linspace(0, 1, n_knots)
    t = np.concatenate(
        [np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)]
    )
    basis_list = []
    x_vals = np.linspace(0, 1, n)
    for i in range(K):
        coef = np.zeros(K)
        coef[i] = 1.0
        bspl = BSpline(t, coef, degree)
        basis_list.append(bspl(x_vals))
    B = np.column_stack(basis_list)
    return torch.tensor(B, dtype=torch.float32, device=device)


def create_model(
    variant_name: str,
    d: int,
    p: int,
    n_knots: int = 5,
    rank_w: int = 20,
    rank_a: int = 10,
    lambda_smooth: float = 0.01,
    device: str = "cpu",
):
    """
    Create the appropriate model for a given variant.

    Returns (model, variant_config) tuple.
    """
    variant = VARIANTS[variant_name]
    model_type = variant["model_type"]
    extra_kwargs = variant.get("model_kwargs", {})

    if model_type == "tucker":
        model = TuckerCAMModel(
            d=d,
            p=p,
            n_knots=n_knots,
            rank_w=rank_w,
            rank_a=rank_a,
            lambda_smooth=lambda_smooth,
            device=device,
            **extra_kwargs,
        )
    elif model_type == "dense":
        model = DenseCAMModel(d=d, p=p, n_knots=n_knots, lambda_smooth=lambda_smooth, device=device)
    elif model_type == "cp":
        model = CPCAMModel(
            d=d,
            p=p,
            n_knots=n_knots,
            rank_w=rank_w,
            rank_a=rank_a,
            lambda_smooth=lambda_smooth,
            device=device,
        )
    elif model_type == "linear":
        model = LinearSVARModel(d=d, p=p, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, variant


def fit_single_window(
    model,
    X: torch.Tensor,
    Xlags: torch.Tensor,
    variant_config: dict,
    max_iter: int = 100,
    lr: float = 0.01,
    lambda_w: float = 0.0,
    lambda_a: float = 0.0,
    lambda_core: float = 0.01,
    lambda_orth: float = 0.001,
    rho_init: float = 1.0,
    rho_max: float = 1e10,
    h_tol: float = 1e-8,
    verbose: bool = False,
):
    """
    Fit a model on a single window of data using augmented Lagrangian.

    Shared optimization loop for all model variants.
    Returns the fitted model (weights can be extracted after).
    """
    n, d = X.shape
    device = X.device

    # Override lambda_w/lambda_a for l1_sparsity variant
    if "lambda_w" in variant_config:
        lambda_w = variant_config["lambda_w"]
    if "lambda_a" in variant_config:
        lambda_a = variant_config["lambda_a"]

    # Set up basis matrices (skip for linear model)
    model_type = variant_config["model_type"]
    if model_type != "linear":
        n_knots = model.n_knots if hasattr(model, "n_knots") else 5
        K = model.K
        B = _compute_basis_matrix(n, n_knots, K, device=str(device))
        model.set_basis_matrices(B, B)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    alpha = torch.zeros(1, device=device)
    rho = rho_init
    prev_h = float("inf")
    history_losses = []

    for it in range(max_iter):
        optimizer.zero_grad()

        pred = model.forward(X, Xlags)
        loss_fit = torch.mean((X - pred) ** 2)

        # L1 penalties on coefficients
        loss_l1 = torch.tensor(0.0, device=device)
        if lambda_w > 0 and hasattr(model, "get_W_coefs"):
            W_coefs = model.get_W_coefs()
            loss_l1 = loss_l1 + lambda_w * torch.sum(torch.abs(W_coefs))

        # Smoothness penalty
        loss_smooth = torch.tensor(0.0, device=device)
        if hasattr(model, "compute_smoothness_penalty"):
            loss_smooth = model.compute_smoothness_penalty()

        # Core sparsity (Tucker only)
        loss_core = torch.tensor(0.0, device=device)
        if hasattr(model, "compute_core_sparsity_penalty"):
            loss_core = lambda_core * model.compute_core_sparsity_penalty()

        # Orthogonality (Tucker only)
        loss_orth = torch.tensor(0.0, device=device)
        if hasattr(model, "compute_orthogonality_penalty"):
            loss_orth = lambda_orth * model.compute_orthogonality_penalty()

        # Acyclicity constraint: h(W) = tr(e^(W o W)) - d
        W = model.get_weight_matrix()
        W_sq = W * W
        h = torch.trace(torch.matrix_exp(W_sq)) - d

        loss_total = (
            loss_fit
            + loss_l1
            + loss_smooth
            + loss_core
            + loss_orth
            + alpha * h
            + 0.5 * rho * h * h
        )

        loss_total.backward()
        optimizer.step()

        h_val = h.item()

        # Early stopping: DAG constraint satisfied
        if abs(h_val) <= h_tol and it > 0:
            break

        # Early stopping: loss plateau
        history_losses.append(loss_total.item())
        if it >= 3:
            recent = history_losses[-3:]
            changes = [
                abs(recent[i] - recent[i - 1]) / (abs(recent[i - 1]) + 1e-8)
                for i in range(1, len(recent))
            ]
            if all(c < 5e-4 for c in changes):
                break

        # Update Lagrangian: increase rho only when h is not improving
        if h_val > 0.25 * prev_h:
            rho = min(rho * 10, rho_max)
        alpha = alpha + rho * h_val
        prev_h = h_val

        if verbose and it % 20 == 0:
            logger.info(
                f"  iter {it}: loss={loss_fit.item():.4f} h={h.item():.2e} rho={rho:.1e}"
            )

    return model


def extract_weight_matrix(model) -> np.ndarray:
    """Extract d x d contemporaneous weight matrix from fitted model."""
    with torch.no_grad():
        W = model.get_weight_matrix()
    return W.detach().cpu().numpy()


def extract_all_weights(model):
    """Extract W and A_lags weight matrices from fitted model."""
    with torch.no_grad():
        if hasattr(model, "get_all_weight_matrices_gpu"):
            W, A_lags = model.get_all_weight_matrices_gpu()
        elif hasattr(model, "get_all_weight_matrices"):
            W, A_lags = model.get_all_weight_matrices()
        else:
            W = model.get_weight_matrix()
            A_lags = []
    W_np = W.detach().cpu().numpy()
    A_np = [a.detach().cpu().numpy() for a in A_lags]
    return W_np, A_np


# ---------------------------------------------------------------------------
# Full pipeline: rolling window training + weight extraction
# ---------------------------------------------------------------------------


def run_rolling_windows(
    data: np.ndarray,
    variant_name: str,
    p: int = 5,
    window_size: int = 100,
    stride: int = 10,
    rank_w: int = 20,
    rank_a: int = 10,
    n_knots: int = 5,
    max_iter: int = 100,
    lr: float = 0.01,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = False,
):
    """
    Run rolling-window causal discovery on a time series.

    Args:
        data: (T, d) numpy array
        variant_name: key from VARIANTS dict
        p: lag order
        window_size: samples per window
        stride: step between windows

    Returns:
        List of (window_idx, W_matrix) tuples where W is (d,d) numpy array.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    T, d = data.shape
    results = []

    n_windows = (T - window_size - p) // stride + 1
    if n_windows <= 0:
        logger.warning(f"Not enough data for rolling windows (T={T}, ws={window_size})")
        return results

    if verbose:
        logger.info(
            f"Rolling windows: T={T}, d={d}, windows={n_windows}, variant={variant_name}"
        )

    for w_idx in range(n_windows):
        start = w_idx * stride
        end = start + window_size

        if end + p > T:
            break

        # Extract window data
        window_data = data[start : end + p].astype(np.float32)

        # Build X and Xlags
        X_np = window_data[p:]  # (window_size, d)
        Xlags_list = []
        for lag in range(1, p + 1):
            Xlags_list.append(window_data[p - lag : -lag if lag < len(window_data) else None][:window_size])
        Xlags_np = np.concatenate(Xlags_list, axis=1)  # (window_size, d*p)

        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        Xlags = torch.tensor(Xlags_np, dtype=torch.float32, device=device)

        # Create fresh model for each window (same as main pipeline)
        model, variant_config = create_model(
            variant_name, d, p,
            n_knots=n_knots, rank_w=rank_w, rank_a=rank_a,
            device=device,
        )

        # Re-seed per window for reproducibility (seed + window_idx)
        torch.manual_seed(seed + w_idx)

        start_time = time.time()
        model = fit_single_window(
            model, X, Xlags, variant_config,
            max_iter=max_iter, lr=lr, verbose=verbose,
        )
        elapsed = time.time() - start_time

        W_np = extract_weight_matrix(model)
        results.append({
            "window_idx": w_idx,
            "W": W_np,
            "time_s": elapsed,
        })

        # Memory cleanup
        del model, X, Xlags
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


def compute_anomaly_scores(
    golden_weights: list,
    test_weights: list,
    detection_mode: str = "multi",
    threshold_mode: str = "adaptive",
) -> dict:
    """
    Compute per-window anomaly scores from weight matrices.

    Args:
        golden_weights: list of W matrices from training (normal) data
        test_weights: list of W matrices from test data
        detection_mode: "multi" (s_abs + s_change + s_trend) or "single" (s_abs only)
        threshold_mode: "adaptive" (MAD-based) or "fixed" (mean + 3*std)

    Returns:
        dict with 'scores', 'predictions', 'threshold' arrays
    """
    from scipy.linalg import norm as sp_norm

    # Compute golden baseline (average of training weight matrices)
    golden_Ws = [w["W"] for w in golden_weights]
    W_baseline = np.mean(golden_Ws, axis=0)
    baseline_norm = sp_norm(W_baseline, "fro") + 1e-10

    # Compute baseline score distribution for thresholding
    baseline_scores = []
    for w in golden_weights:
        s = sp_norm(w["W"] - W_baseline, "fro") / baseline_norm
        baseline_scores.append(s)
    baseline_scores = np.array(baseline_scores)

    # Compute test scores
    n_test = len(test_weights)
    s_abs = np.zeros(n_test)
    s_change = np.zeros(n_test)
    s_trend = np.zeros(n_test)

    prev_W = W_baseline
    lookback = 5

    for i, tw in enumerate(test_weights):
        W_test = tw["W"]
        s_abs[i] = sp_norm(W_test - W_baseline, "fro") / baseline_norm
        prev_norm = sp_norm(prev_W, "fro") + 1e-10
        s_change[i] = sp_norm(W_test - prev_W, "fro") / prev_norm
        if i >= lookback:
            s_trend[i] = s_abs[i] - s_abs[i - lookback]
        prev_W = W_test

    # Thresholding
    if threshold_mode == "fixed":
        # Static: mean + 3*std from baseline distribution
        mu = np.mean(baseline_scores)
        sigma = np.std(baseline_scores) + 1e-10
        threshold_abs = mu + 3.0 * sigma
    else:
        # Adaptive: rolling median + 3*MAD, initialized from baseline
        # Threshold adapts as test progresses using recent score history
        mu = np.median(baseline_scores)
        mad = np.median(np.abs(baseline_scores - mu)) + 1e-10
        threshold_abs = mu + 3.0 * 1.4826 * mad  # 1.4826 scales MAD to std

    # Generate predictions
    if detection_mode == "single":
        # Only s_abs (no multi-metric confirmation)
        scores = s_abs
        predictions = (s_abs >= threshold_abs).astype(int)
    else:
        # Multi-metric: composite score combining all three signals
        # s_abs: deviation from baseline (primary)
        # s_change: rate of structural change (secondary)
        # s_trend: direction of change (tertiary, only positive = worsening)
        scores = s_abs + 0.3 * s_change + 0.2 * np.clip(s_trend, 0, None)
        predictions = np.zeros(n_test, dtype=int)

        # Baseline statistics for s_change
        baseline_change_scores = []
        for i in range(1, len(golden_Ws)):
            bc = sp_norm(golden_Ws[i] - golden_Ws[i - 1], "fro") / baseline_norm
            baseline_change_scores.append(bc)
        change_mu = np.mean(baseline_change_scores) if baseline_change_scores else 0.0
        change_sigma = np.std(baseline_change_scores) + 1e-10 if baseline_change_scores else 1e-10
        change_threshold = change_mu + 2.0 * change_sigma

        for i in range(n_test):
            if s_abs[i] < threshold_abs:
                continue  # Normal window
            # Candidate anomaly -- use s_change and s_trend to confirm
            if s_change[i] > change_threshold and s_trend[i] > 0:
                predictions[i] = 1  # New anomaly onset (high change + worsening)
            elif s_change[i] > change_threshold and s_trend[i] <= 0:
                pass  # Recovery fluctuation (high change but improving) -- NOT anomaly
            elif s_abs[i] >= threshold_abs * 1.5:
                predictions[i] = 1  # Persistent/cascade: very high deviation
            # else: borderline, leave as 0

        # Adaptive threshold update: adjust threshold using recent test scores
        if threshold_mode != "fixed":
            adapt_window = 50
            for i in range(n_test):
                if i >= adapt_window:
                    recent = s_abs[max(0, i - adapt_window):i]
                    rolling_med = np.median(recent)
                    rolling_mad = np.median(np.abs(recent - rolling_med)) + 1e-10
                    adapted_th = rolling_med + 3.0 * 1.4826 * rolling_mad
                    # Use the stricter of baseline and adaptive threshold
                    effective_th = max(threshold_abs, adapted_th)
                    if s_abs[i] < effective_th:
                        predictions[i] = 0  # Override: below adaptive threshold

    return {
        "scores": scores,
        "s_abs": s_abs,
        "s_change": s_change,
        "s_trend": s_trend,
        "predictions": predictions,
        "threshold": threshold_abs,
    }


def window_labels_from_point_labels(
    point_labels: np.ndarray,
    n_windows: int,
    window_size: int = 100,
    stride: int = 10,
    p: int = 5,
) -> np.ndarray:
    """
    Convert point-level ground truth labels to window-level labels.

    A window is labeled anomalous if ANY point in it is anomalous.
    """
    window_labels = np.zeros(n_windows, dtype=int)
    for w in range(n_windows):
        start = w * stride + p
        end = start + window_size
        if end > len(point_labels):
            break
        if np.any(point_labels[start:end] == 1):
            window_labels[w] = 1
    return window_labels
