#!/usr/bin/env python3
"""
Dense CAM Model -- No Tucker Decomposition.

Stores the full W[d,d,K] and A[d,d,p,K] coefficient tensors directly.
This is the "No Tucker" ablation baseline. Only feasible for small d (<100)
due to O(d^2 * p * K) parameter count.

Purpose: prove that Tucker preserves accuracy (F1 within tolerance)
         while reducing memory by orders of magnitude.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline


class DenseCAMModel(nn.Module):
    """
    Full-rank P-spline CAM model without any tensor decomposition.
    Directly parameterizes W[d,d,K] and A[d,d,p,K].
    """

    def __init__(self, d: int, p: int, n_knots: int = 5, lambda_smooth: float = 0.01, device="cpu"):
        super().__init__()
        self.d = d
        self.p = p
        self.n_knots = n_knots
        self.lambda_smooth = lambda_smooth
        self.degree = 3
        self.K = n_knots + self.degree
        self.device = device

        # Full coefficient tensors (no decomposition)
        self.W_coefs = nn.Parameter(
            torch.randn(d, d, self.K, device=device)
            * (1.0 / np.sqrt(d * self.K))
        )
        self.A_coefs = nn.Parameter(
            torch.randn(d, d, p, self.K, device=device)
            * (1.0 / np.sqrt(d * p * self.K))
        )

        self.register_buffer("B_w", torch.ones(1, self.K, device=device))
        self.register_buffer("B_a", torch.ones(1, self.K, device=device))
        self.register_buffer("W_mask", torch.ones(d, d, device=device))
        self.register_buffer("A_mask", torch.ones(d, d, p, device=device))

        total_params = d * d * self.K + d * d * p * self.K
        mem_gb = total_params * 4 / (1024 ** 3)
        print(f"[Dense CAM] d={d}, p={p}, K={self.K}")
        print(f"[Dense CAM] Parameters: {total_params:,} ({mem_gb:.2f} GB)")

    def set_basis_matrices(self, B_w, B_a):
        self.B_w = B_w.to(self.device)
        self.B_a = B_a.to(self.device)

    def _compute_basis_matrix(self, X, degree=3):
        n = X.shape[0]
        knots = np.linspace(0, 1, self.n_knots)
        t = np.concatenate(
            [np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)]
        )
        basis_list = []
        x_vals = np.linspace(0, 1, n)
        for i in range(self.K):
            coef = np.zeros(self.K)
            coef[i] = 1.0
            bspl = BSpline(t, coef, degree)
            basis_list.append(bspl(x_vals))
        B = np.column_stack(basis_list)
        return torch.tensor(B, dtype=torch.float32, device=self.device)

    def get_W_coefs(self):
        return self.W_coefs

    def get_A_coefs(self):
        return self.A_coefs

    def forward(self, X, Xlags):
        n, d = X.shape

        # Contemporaneous: X @ W_mat where W_mat[i,j] = sum_k W[i,j,k]*B[n,k]
        # Compute effective weight matrix per sample via basis functions
        # B_w: (n, K), W_coefs: (d, d, K)
        # W_eff[n,i,j] = sum_k W_coefs[i,j,k] * B_w[n,k]
        # Then pred_w[n,j] = sum_i X[n,i] * W_eff[n,i,j]
        W_eff = torch.einsum("ijk,nk->nij", self.W_coefs, self.B_w[:n])
        contrib_w = torch.einsum("ni,nij->nj", X, W_eff)

        # Lagged: Xlags reshaped to (n, d, p)
        Xlags_3d = Xlags.view(n, self.p, d).permute(0, 2, 1)  # (n, d, p)
        # A_coefs: (d, d, p, K), B_a: (n, K)
        # A_eff[n,i,j,l] = sum_k A_coefs[i,j,l,k] * B_a[n,k]
        A_eff = torch.einsum("ijlk,nk->nijl", self.A_coefs, self.B_a[:n])
        # pred_a[n,j] = sum_{i,l} Xlags_3d[n,i,l] * A_eff[n,i,j,l]
        contrib_a = torch.einsum("nil,nijl->nj", Xlags_3d, A_eff)

        return contrib_w + contrib_a

    def get_weight_matrix(self):
        W = self.W_coefs.mean(dim=2) * self.W_mask
        return W

    def get_all_weight_matrices(self):
        W = self.get_weight_matrix()
        A_lags = []
        for lag in range(self.p):
            A_lag = self.A_coefs[:, :, lag, :].mean(dim=2) * self.A_mask[:, :, lag]
            A_lags.append(A_lag)
        return W, A_lags

    def compute_smoothness_penalty(self):
        W_diff = self.W_coefs[:, :, 1:] - self.W_coefs[:, :, :-1]
        A_diff = self.A_coefs[:, :, :, 1:] - self.A_coefs[:, :, :, :-1]
        return self.lambda_smooth * (torch.sum(W_diff ** 2) + torch.sum(A_diff ** 2))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
