"""
Utility functions for sparse matrix operations in PyTorch.

Provides efficient conversion between scipy sparse matrices and PyTorch sparse
tensors, and batched sparse-dense matrix multiplication used by ChebConv layers.
"""

import torch
import scipy.sparse as sp
import numpy as np


def scipy_sparse_to_torch(A: sp.spmatrix, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert a scipy sparse matrix to a PyTorch sparse COO tensor.

    Args:
        A: scipy sparse matrix of shape [M, N]
        dtype: desired torch dtype for the values

    Returns:
        torch sparse COO tensor of shape [M, N]
    """
    # Ensure CSR/COO format is available
    A = A.tocoo().astype(np.float32)
    row = torch.from_numpy(A.row.astype(np.int64))
    col = torch.from_numpy(A.col.astype(np.int64))
    values = torch.from_numpy(A.data).to(dtype)
    indices = torch.stack([row, col], dim=0)
    shape = torch.Size(A.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def apply_sparse_to_batch(S: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Efficient batched sparse-dense matrix multiplication.

    Computes  out[b] = S @ x[b]  for each batch element b.

    The trick is to reshape x from [B, N, F] into [N, B*F], apply the sparse
    matrix S (shape [M, N]) to get [M, B*F], then reshape back to [B, M, F].

    Args:
        S: sparse tensor of shape [M, N]
        x: dense tensor of shape [B, N, F]

    Returns:
        dense tensor of shape [B, M, F]
    """
    B, N, F = x.shape
    M = S.shape[0]

    # Reshape x from [B, N, F] to [N, B*F] for a single batched matmul
    x_reshaped = x.permute(1, 0, 2).reshape(N, B * F)  # [N, B*F]

    # Sparse matmul: [M, N] @ [N, B*F] → [M, B*F]
    out = torch.sparse.mm(S, x_reshaped)  # [M, B*F]

    # Reshape back to [B, M, F]
    out = out.reshape(M, B, F).permute(1, 0, 2)  # [B, M, F]
    return out
