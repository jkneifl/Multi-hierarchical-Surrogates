"""Tests for src/layers/graph_conv.py"""

import pytest
import torch
import torch.nn as nn

from src.layers.graph_conv import ChebConv, adjacency_to_edge_index


class TestAdjacencyToEdgeIndex:
    def test_shape(self, adjacency):
        ei = adjacency_to_edge_index(adjacency)
        assert ei.shape[0] == 2
        assert ei.shape[1] == adjacency.nnz

    def test_dtype(self, adjacency):
        ei = adjacency_to_edge_index(adjacency)
        assert ei.dtype == torch.int64

    def test_values_in_range(self, adjacency):
        ei = adjacency_to_edge_index(adjacency)
        N = adjacency.shape[0]
        assert ei.min().item() >= 0
        assert ei.max().item() < N


class TestChebConv:
    def _make(self, adjacency, in_f=3, out_f=6, order=2, activation=nn.ELU()):
        return ChebConv(in_f, out_f, order=order, adjacency=adjacency, activation=activation)

    def test_output_shape(self, adjacency):
        B, N = 2, adjacency.shape[0]
        layer = self._make(adjacency)
        out = layer(torch.randn(B, N, 3))
        assert out.shape == (B, N, 6)

    def test_order_1(self, adjacency):
        B, N = 2, adjacency.shape[0]
        layer = self._make(adjacency, order=1)
        out = layer(torch.randn(B, N, 3))
        assert out.shape == (B, N, 6)

    def test_activation_none(self, adjacency):
        B, N = 2, adjacency.shape[0]
        layer = self._make(adjacency, activation=None)
        out = layer(torch.randn(B, N, 3))
        assert out.shape == (B, N, 6)

    def test_edge_index_registered_as_buffer(self, adjacency):
        layer = self._make(adjacency)
        assert "edge_index" in dict(layer.named_buffers())

    def test_gradients_flow(self, adjacency):
        layer = self._make(adjacency)
        out = layer(torch.randn(2, adjacency.shape[0], 3))
        out.sum().backward()
        assert all(p.grad is not None for p in layer.parameters())

    def test_device_move(self, adjacency):
        layer = self._make(adjacency).to(torch.device("cpu"))
        out = layer(torch.randn(1, adjacency.shape[0], 3))
        assert out.shape[0] == 1

    def test_batch_independence(self, adjacency):
        """Outputs for different batch elements must be independent."""
        N = adjacency.shape[0]
        layer = self._make(adjacency)
        layer.eval()
        x = torch.randn(2, N, 3)
        out1 = layer(x)
        x2 = x.clone()
        x2[0] = torch.zeros_like(x2[0])
        out2 = layer(x2)
        assert not torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])
