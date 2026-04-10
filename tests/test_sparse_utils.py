"""Tests for src/utils/sparse_utils.py"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src.utils.sparse_utils import apply_sparse_to_batch, scipy_sparse_to_torch


class TestScipySparseToTorch:
    def test_shape(self):
        A = sp.eye(5, format="csr")
        T = scipy_sparse_to_torch(A)
        assert T.shape == (5, 5)

    def test_is_sparse(self):
        A = sp.eye(5, format="csr")
        T = scipy_sparse_to_torch(A)
        assert T.is_sparse

    def test_identity_values(self):
        A = sp.eye(4, format="csr")
        T = scipy_sparse_to_torch(A).to_dense()
        expected = torch.eye(4)
        assert torch.allclose(T, expected)

    def test_non_square(self):
        A = sp.random(3, 7, density=0.5, format="csr")
        T = scipy_sparse_to_torch(A)
        assert T.shape == (3, 7)

    def test_dtype_float32(self):
        A = sp.eye(3, format="csr", dtype=np.float64)
        T = scipy_sparse_to_torch(A, dtype=torch.float32)
        assert T.dtype == torch.float32


class TestApplySparseToBlatch:
    def _identity_sparse(self, n: int) -> torch.Tensor:
        return scipy_sparse_to_torch(sp.eye(n, format="csr"))

    def test_identity_passthrough(self):
        """S = I should leave x unchanged."""
        N = 6
        S = self._identity_sparse(N)
        x = torch.randn(3, N, 5)
        out = apply_sparse_to_batch(S, x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_output_shape(self):
        M, N, B, F = 4, 6, 2, 3
        S = scipy_sparse_to_torch(sp.random(M, N, density=0.5, format="csr").astype(np.float32))
        x = torch.randn(B, N, F)
        out = apply_sparse_to_batch(S, x)
        assert out.shape == (B, M, F)

    def test_batched_consistency(self):
        """Each batch element should equal the single-item result."""
        N, F = 5, 3
        A = sp.random(N, N, density=0.5, format="csr").astype(np.float32)
        S = scipy_sparse_to_torch(A)
        x = torch.randn(4, N, F)

        out_batch = apply_sparse_to_batch(S, x)
        A_dense = torch.from_numpy(A.toarray())
        for b in range(4):
            expected = (A_dense @ x[b])   # [N, F]
            assert torch.allclose(out_batch[b], expected, atol=1e-5)
