#!/usr/bin/env python3
"""
CP-Decomposed CAM Model.

Uses CANDECOMP/PARAFAC (CP) decomposition instead of Tucker.
CP factorizes tensors as sum of rank-1 components (no core tensor):

  W[i,j,k] = sum_r lambda_r * u1[i,r] * u2[j,r] * u3[k,r]

Simpler than Tucker (fewer hyperparameters, no core), but potentially
less expressive since CP cannot capture mode interactions independently.

Purpose: answer "Why Tucker, not CP?" -- a likely reviewer question.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline


class CPCAMModel(nn.Module):
    """
    CAM model with CP-decomposed coefficient tensors.

    W[d,d,K] = sum_r lambda_w[r] * u1[i,r] * u2[j,r] * u3[k,r]
    A[d,d,p,K] = sum_r lambda_a[r] * v1[i,r] * v2[j,r] * v3[l,r] * v4[k,r]
    """

    def __init__(
        self,
        d: int,
        p: int,
        n_knots: int = 5,
        rank_w: int = 20,
        rank_a: int = 10,
        lambda_smooth: float = 0.01,
        device="cpu",
    ):
        super().__init__()
        self.d = d
        self.p = p
        self.n_knots = n_knots
        self.rank_w = rank_w
        self.rank_a = rank_a
        self.lambda_smooth = lambda_smooth
        self.device = device
        self.degree = 3
        self.K = n_knots + self.degree

        # CP factors for W: d x d x K
        scale_w = 1.0 / np.sqrt(rank_w * d)
        self.W_lambda = nn.Parameter(torch.ones(rank_w, device=device))
        self.W_U1 = nn.Parameter(torch.randn(d, rank_w, device=device) * scale_w)
        self.W_U2 = nn.Parameter(torch.randn(d, rank_w, device=device) * scale_w)
        self.W_U3 = nn.Parameter(torch.randn(self.K, rank_w, device=device) * scale_w)

        # CP factors for A: d x d x p x K
        scale_a = 1.0 / np.sqrt(rank_a * d * p)
        self.A_lambda = nn.Parameter(torch.ones(rank_a, device=device))
        self.A_U1 = nn.Parameter(torch.randn(d, rank_a, device=device) * scale_a)
        self.A_U2 = nn.Parameter(torch.randn(d, rank_a, device=device) * scale_a)
        self.A_U3 = nn.Parameter(torch.randn(p, rank_a, device=device) * scale_a)
        self.A_U4 = nn.Parameter(torch.randn(self.K, rank_a, device=device) * scale_a)

        self.register_buffer("B_w", torch.ones(1, self.K, device=device))
        self.register_buffer("B_a", torch.ones(1, self.K, device=device))
        self.register_buffer("W_mask", torch.ones(d, d, device=device))
        self.register_buffer("A_mask", torch.ones(d, d, p, device=device))

        cp_params = (
            rank_w + d * rank_w * 2 + self.K * rank_w
            + rank_a + d * rank_a * 2 + p * rank_a + self.K * rank_a
        )
        dense_params = d * d * self.K + d * d * p * self.K
        print(f"[CP CAM] d={d}, p={p}, K={self.K}, R_w={rank_w}, R_a={rank_a}")
        print(f"[CP CAM] CP params: {cp_params:,}, Dense: {dense_params:,}, Reduction: {dense_params / cp_params:.1f}x")

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
        """Reconstruct W[d,d,K] from CP factors."""
        # W[i,j,k] = sum_r lambda[r] * U1[i,r] * U2[j,r] * U3[k,r]
        return torch.einsum(
            "r,ir,jr,kr->ijk", self.W_lambda, self.W_U1, self.W_U2, self.W_U3
        )

    def get_A_coefs_chunked(self, chunk_size=None):
        """Reconstruct A in chunks (same interface as Tucker model)."""
        if chunk_size is None:
            target_bytes = 100 * 1024 * 1024
            chunk_size = max(
                10,
                int(target_bytes / (self.d * self.p * self.K * 4)),
            )
            chunk_size = min(chunk_size, self.d)

        for i in range(0, self.d, chunk_size):
            i_end = min(i + chunk_size, self.d)
            A_chunk = torch.einsum(
                "r,ir,jr,lr,kr->ijlk",
                self.A_lambda,
                self.A_U1[i:i_end],
                self.A_U2,
                self.A_U3,
                self.A_U4,
            )
            yield i, i_end, A_chunk

    def forward(self, X, Xlags):
        n = X.shape[0]

        # Contemporaneous via CP contraction (analogous to Tucker but no core)
        # path: basis_proj[n,r] = B_w[n,k] * U3[k,r] (element-wise sum over k)
        # input_proj[n,r] = X[n,i] * U1[i,r]
        # combined[n,r] = lambda[r] * input_proj[n,r] * basis_proj[n,r]
        # output[n,j] = combined[n,r] * U2[j,r]
        basis_proj_w = torch.matmul(self.B_w[:n], self.W_U3)  # (n, R_w)
        input_proj_w = torch.matmul(X, self.W_U1)  # (n, R_w)
        combined_w = self.W_lambda * input_proj_w * basis_proj_w  # (n, R_w)
        contrib_w = torch.matmul(combined_w, self.W_U2.T)  # (n, d)

        # Lagged via CP contraction
        Xlags_3d = Xlags.view(n, self.p, self.d).permute(0, 2, 1)  # (n, d, p)
        basis_proj_a = torch.matmul(self.B_a[:n], self.A_U4)  # (n, R_a)
        # Contract over d and p: x_proj[n,r] = sum_{i,l} X[n,i,l] * U1[i,r] * U3[l,r]
        x_lag_proj = torch.einsum("nil,lr->nir", Xlags_3d, self.A_U3)  # (n, d, R_a)
        x_full_proj = torch.einsum("nir,ir->nr", x_lag_proj, self.A_U1)  # (n, R_a)
        combined_a = self.A_lambda * x_full_proj * basis_proj_a  # (n, R_a)
        contrib_a = torch.matmul(combined_a, self.A_U2.T)  # (n, d)

        return contrib_w + contrib_a

    def get_weight_matrix(self):
        W_coefs = self.get_W_coefs()
        W = W_coefs.mean(dim=2) * self.W_mask
        return W

    def get_all_weight_matrices(self):
        W = self.get_weight_matrix()
        A_lags = []
        for lag in range(self.p):
            A_lag = torch.zeros(self.d, self.d, device=self.device)
            for i_start, i_end, A_chunk in self.get_A_coefs_chunked():
                A_lag[i_start:i_end] = (
                    A_chunk[:, :, lag, :].mean(dim=2)
                    * self.A_mask[i_start:i_end, :, lag]
                )
            A_lags.append(A_lag)
        return W, A_lags

    def compute_smoothness_penalty(self):
        W_coefs = self.get_W_coefs()
        W_diff = W_coefs[:, :, 1:] - W_coefs[:, :, :-1]
        penalty = torch.sum(W_diff ** 2)
        for _, _, A_chunk in self.get_A_coefs_chunked():
            A_diff = A_chunk[:, :, :, 1:] - A_chunk[:, :, :, :-1]
            penalty = penalty + torch.sum(A_diff ** 2)
        return self.lambda_smooth * penalty

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
