"""Tests for src/layers/graph_conv.py"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.layers.graph_conv import ChebConv, GCNConv, _compute_scaled_laplacian, _compute_normalised_adjacency


class TestComputeScaledLaplacian:
    def test_shape(self, adjacency):
        L = _compute_scaled_laplacian(adjacency)
        N = adjacency.shape[0]
        assert L.shape == (N, N)

    def test_symmetric(self, adjacency):
        L = _compute_scaled_laplacian(adjacency)
        diff = L - L.T
        assert np.allclose(diff.data, 0, atol=1e-6)

    def test_eigenvalues_in_range(self, adjacency):
        """Eigenvalues of L̃ should lie in [-1, 1]."""
        L = _compute_scaled_laplacian(adjacency)
        eigs = np.linalg.eigvalsh(L.toarray())
        assert eigs.min() >= -1.0 - 1e-5
        assert eigs.max() <= 1.0 + 1e-5

    def test_no_self_loops_in_input_preserved(self, adjacency):
        """Self-loops in the adjacency are removed before computing L."""
        A_with_loops = adjacency + sp.eye(adjacency.shape[0], format="csr")
        L_clean = _compute_scaled_laplacian(adjacency)
        L_loops = _compute_scaled_laplacian(A_with_loops)
        assert np.allclose(L_clean.toarray(), L_loops.toarray(), atol=1e-6)


class TestComputeNormalisedAdjacency:
    def test_shape(self, adjacency):
        A_hat = _compute_normalised_adjacency(adjacency)
        assert A_hat.shape == adjacency.shape

    def test_symmetric(self, adjacency):
        A_hat = _compute_normalised_adjacency(adjacency)
        diff = A_hat - A_hat.T
        assert np.allclose(diff.data, 0, atol=1e-6)

    def test_has_self_loops(self, adjacency):
        """Â = D̃^{-1/2}(A+I)D̃^{-1/2} must have non-zero diagonal."""
        A_hat = _compute_normalised_adjacency(adjacency)
        assert (A_hat.diagonal() > 0).all()


class TestChebConv:
    def _make_layer(self, adjacency, in_f=3, out_f=6, order=2, activation=nn.ELU()):
        return ChebConv(
            in_features=in_f,
            out_features=out_f,
            order=order,
            adjacency=adjacency,
            activation=activation,
        )

    def test_output_shape(self, adjacency):
        B, N, F_in, F_out = 2, adjacency.shape[0], 3, 6
        layer = self._make_layer(adjacency, in_f=F_in, out_f=F_out)
        x = torch.randn(B, N, F_in)
        out = layer(x)
        assert out.shape == (B, N, F_out)

    def test_order_1_output_shape(self, adjacency):
        """Order=1 means only T_0 (identity), should still work."""
        B, N = 2, adjacency.shape[0]
        layer = self._make_layer(adjacency, order=1)
        x = torch.randn(B, N, 3)
        out = layer(x)
        assert out.shape == (B, N, 6)

    def test_linear_activation_none(self, adjacency):
        """Passing activation=None should not raise and output is unbounded."""
        layer = self._make_layer(adjacency, activation=None)
        x = torch.randn(2, adjacency.shape[0], 3)
        out = layer(x)
        assert out.shape[2] == 6

    def test_l_tilde_registered_as_buffer(self, adjacency):
        layer = self._make_layer(adjacency)
        assert "L_tilde" in dict(layer.named_buffers())

    def test_weight_shape(self, adjacency):
        in_f, out_f, order = 3, 8, 3
        layer = self._make_layer(adjacency, in_f=in_f, out_f=out_f, order=order)
        assert layer.weight.shape == (in_f * order, out_f)

    def test_gradients_flow(self, adjacency):
        layer = self._make_layer(adjacency)
        x = torch.randn(2, adjacency.shape[0], 3)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

    def test_device_move(self, adjacency):
        """Buffers and parameters should move together with .to()."""
        layer = self._make_layer(adjacency)
        # Just check CPU→CPU doesn't break anything
        layer = layer.to(torch.device("cpu"))
        x = torch.randn(1, adjacency.shape[0], 3)
        out = layer(x)
        assert out.shape[0] == 1


class TestGCNConv:
    def _make_layer(self, adjacency, in_f=3, out_f=6):
        return GCNConv(in_features=in_f, out_features=out_f, adjacency=adjacency)

    def test_output_shape(self, adjacency):
        B, N = 2, adjacency.shape[0]
        layer = self._make_layer(adjacency)
        x = torch.randn(B, N, 3)
        out = layer(x)
        assert out.shape == (B, N, 6)

    def test_a_hat_registered_as_buffer(self, adjacency):
        layer = self._make_layer(adjacency)
        assert "A_hat" in dict(layer.named_buffers())

    def test_gradients_flow(self, adjacency):
        layer = self._make_layer(adjacency)
        x = torch.randn(2, adjacency.shape[0], 3)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad is not None
