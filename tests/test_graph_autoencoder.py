"""Tests for src/models/graph_autoencoder.py"""

import pytest
import torch

from src.models.graph_autoencoder import GraphDecoder, GraphEncoder


class TestGraphEncoder:
    def _make(self, adjacency, model_cfg):
        N = adjacency.shape[0]
        return GraphEncoder(
            n_nodes=N,
            in_features=model_cfg["n_features"],
            latent_dim=model_cfg["latent_dim"],
            filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"],
            adjacency=adjacency,
        )

    def test_output_shape(self, adjacency, model_cfg):
        B, N = 3, adjacency.shape[0]
        enc = self._make(adjacency, model_cfg)
        x = torch.randn(B, N, model_cfg["n_features"])
        z = enc(x)
        assert z.shape == (B, model_cfg["latent_dim"])

    def test_gradients_flow(self, adjacency, model_cfg):
        B, N = 2, adjacency.shape[0]
        enc = self._make(adjacency, model_cfg)
        x = torch.randn(B, N, model_cfg["n_features"])
        z = enc(x)
        z.sum().backward()
        for p in enc.parameters():
            assert p.grad is not None

    def test_batch_independence(self, adjacency, model_cfg):
        """Outputs for different batch items should be independent."""
        N = adjacency.shape[0]
        enc = self._make(adjacency, model_cfg)
        enc.eval()
        x = torch.randn(2, N, model_cfg["n_features"])
        z = enc(x)
        # Changing one input should not affect the other's output
        x2 = x.clone()
        x2[0] = torch.zeros_like(x2[0])
        z2 = enc(x2)
        assert not torch.allclose(z[0], z2[0])
        assert torch.allclose(z[1], z2[1])


class TestGraphDecoder:
    def _make(self, adjacency, model_cfg):
        N = adjacency.shape[0]
        return GraphDecoder(
            n_nodes=N,
            out_features=model_cfg["n_features"],
            latent_dim=model_cfg["latent_dim"],
            filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"],
            adjacency=adjacency,
        )

    def test_output_shape(self, adjacency, model_cfg):
        B, N = 3, adjacency.shape[0]
        dec = self._make(adjacency, model_cfg)
        z = torch.randn(B, model_cfg["latent_dim"])
        y = dec(z)
        assert y.shape == (B, N, model_cfg["n_features"])

    def test_gradients_flow(self, adjacency, model_cfg):
        B = 2
        dec = self._make(adjacency, model_cfg)
        z = torch.randn(B, model_cfg["latent_dim"])
        y = dec(z)
        y.sum().backward()
        for p in dec.parameters():
            assert p.grad is not None


class TestEncoderDecoderRoundtrip:
    def test_roundtrip_shape(self, adjacency, model_cfg):
        N = adjacency.shape[0]
        enc = GraphEncoder(
            n_nodes=N,
            in_features=model_cfg["n_features"],
            latent_dim=model_cfg["latent_dim"],
            filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"],
            adjacency=adjacency,
        )
        dec = GraphDecoder(
            n_nodes=N,
            out_features=model_cfg["n_features"],
            latent_dim=model_cfg["latent_dim"],
            filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"],
            adjacency=adjacency,
        )
        B = 4
        x = torch.randn(B, N, model_cfg["n_features"])
        y = dec(enc(x))
        assert y.shape == x.shape
