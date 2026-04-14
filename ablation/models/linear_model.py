#!/usr/bin/env python3
"""
Linear VAR Model -- No P-Splines.

Standard linear structural vector autoregression:
  x_i(t) = sum_j W[i,j] * x_j(t) + sum_j sum_l A[i,j,l] * x_j(t-l) + eps

No basis functions, no nonlinearity. This is the "Linear (No P-splines)"
ablation that tests whether nonlinearity actually helps on these datasets.

Equivalent to what DYNOTEARS would learn (modulo optimization differences).
"""

import torch
import torch.nn as nn
import numpy as np


class LinearSVARModel(nn.Module):
    """
    Linear Structural VAR model.
    Directly stores W[d,d] and A[d,d,p] weight matrices.
    """

    def __init__(self, d: int, p: int, device="cpu"):
        super().__init__()
        self.d = d
        self.p = p
        self.K = 1  # Compatibility: linear has no basis expansion
        self.device = device

        scale = 1.0 / np.sqrt(d)
        self.W = nn.Parameter(torch.randn(d, d, device=device) * scale)
        self.A = nn.Parameter(torch.randn(d, d, p, device=device) * scale)

        self.register_buffer("W_mask", torch.ones(d, d, device=device))
        self.register_buffer("A_mask", torch.ones(d, d, p, device=device))

        total_params = d * d + d * d * p
        print(f"[Linear SVAR] d={d}, p={p}, Parameters: {total_params:,}")

    def set_basis_matrices(self, B_w, B_a):
        pass  # No basis matrices needed

    def _compute_basis_matrix(self, X, degree=3):
        return None  # Not used

    def get_W_coefs(self):
        # Return W as (d, d, 1) for interface compatibility
        return self.W.unsqueeze(2)

    def forward(self, X, Xlags):
        n, d = X.shape
        # Contemporaneous: X @ W^T
        contrib_w = torch.matmul(X, self.W.T)

        # Lagged: sum over lags
        Xlags_3d = Xlags.view(n, self.p, d).permute(0, 2, 1)  # (n, d, p)
        # A[i,j,l]: effect of x_j(t-l) on x_i(t)
        # contrib_a[n,i] = sum_{j,l} Xlags_3d[n,j,l] * A[i,j,l]
        contrib_a = torch.einsum("njl,ijl->ni", Xlags_3d, self.A)

        return contrib_w + contrib_a

    def get_weight_matrix(self):
        return self.W * self.W_mask

    def get_all_weight_matrices(self):
        W = self.get_weight_matrix()
        A_lags = []
        for lag in range(self.p):
            A_lags.append(self.A[:, :, lag] * self.A_mask[:, :, lag])
        return W, A_lags

    def compute_smoothness_penalty(self):
        return torch.tensor(0.0, device=self.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
