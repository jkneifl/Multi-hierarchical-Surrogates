"""Tests for src/models/surrogate.py (MLP, MLPAutoencoder, UpsamplingConvolution)."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.models.surrogate import MLP, MLPAutoencoder, UpsamplingConvolution
from src.utils.mesh_sampling import build_sampling_matrices


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape(self, model_cfg):
        mlp = MLP(param_dim=model_cfg["param_dim"], latent_dim=model_cfg["latent_dim"],
                  hidden_sizes=model_cfg["mlp_hidden"])
        x = torch.randn(5, model_cfg["param_dim"])
        z = mlp(x)
        assert z.shape == (5, model_cfg["latent_dim"])

    def test_gradients(self, model_cfg):
        mlp = MLP(param_dim=model_cfg["param_dim"], latent_dim=model_cfg["latent_dim"])
        x = torch.randn(3, model_cfg["param_dim"])
        mlp(x).sum().backward()
        for p in mlp.parameters():
            assert p.grad is not None

    def test_default_hidden_sizes(self, model_cfg):
        mlp = MLP(param_dim=model_cfg["param_dim"], latent_dim=model_cfg["latent_dim"])
        # Default hidden = [64, 64, 64] → 4 Linear layers
        linears = [m for m in mlp.net if isinstance(m, nn.Linear)]
        assert len(linears) == 4  # 3 hidden + 1 output


# ---------------------------------------------------------------------------
# MLPAutoencoder
# ---------------------------------------------------------------------------

class TestMLPAutoencoder:
    def _make(self, adjacency, model_cfg, coarse_model=None):
        N = adjacency.shape[0]
        return MLPAutoencoder(
            n_nodes=N,
            param_dim=model_cfg["param_dim"],
            latent_dim=model_cfg["latent_dim"],
            filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"],
            adjacency=adjacency,
            mlp_hidden=model_cfg["mlp_hidden"],
            n_features=model_cfg["n_features"],
            coarse_model=coarse_model,
        )

    def test_encode_shape(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 3
        model = self._make(adjacency, model_cfg)
        y = torch.randn(B, N, model_cfg["n_features"])
        z = model.encode(y)
        assert z.shape == (B, model_cfg["latent_dim"])

    def test_decode_shape(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 3
        model = self._make(adjacency, model_cfg)
        z = torch.randn(B, model_cfg["latent_dim"])
        y = model.decode(z)
        assert y.shape == (B, N, model_cfg["n_features"])

    def test_forward_shape(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 3
        model = self._make(adjacency, model_cfg)
        x = torch.randn(B, model_cfg["param_dim"])
        y = model(x)
        assert y.shape == (B, N, model_cfg["n_features"])

    def test_reconstruct_shape(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 2
        model = self._make(adjacency, model_cfg)
        y = torch.randn(B, N, model_cfg["n_features"])
        y_rec = model.reconstruct(y)
        assert y_rec.shape == y.shape

    def test_compute_loss_returns_four_scalars(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 4
        model = self._make(adjacency, model_cfg)
        x = torch.randn(B, model_cfg["param_dim"])
        y = torch.randn(B, N, model_cfg["n_features"])
        total, L_rec, L_x, L_z = model.compute_loss(x, y)
        for t in [total, L_rec, L_x, L_z]:
            assert t.shape == ()   # scalar

    def test_loss_non_negative(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 4
        model = self._make(adjacency, model_cfg)
        x = torch.randn(B, model_cfg["param_dim"])
        y = torch.randn(B, N, model_cfg["n_features"])
        total, L_rec, L_x, L_z = model.compute_loss(x, y)
        assert total.item() >= 0

    def test_loss_backward(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 2
        model = self._make(adjacency, model_cfg)
        x = torch.randn(B, model_cfg["param_dim"])
        y = torch.randn(B, N, model_cfg["n_features"])
        total, _, _, _ = model.compute_loss(x, y)
        total.backward()
        # At least the encoder weights should have gradients
        for p in model.encoder.parameters():
            assert p.grad is not None

    def test_residual_encode_adds_coarse(self, grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
        """encode() downsamples y to coarse level before calling coarse_model.encode()."""
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        D, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)

        N_fine, N_coarse, B = len(fine_verts), len(coarse_verts), 2

        coarse_model = MLPAutoencoder(
            n_nodes=N_coarse, param_dim=model_cfg["param_dim"],
            latent_dim=model_cfg["latent_dim"], filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"], adjacency=coarse_adjacency,
            mlp_hidden=model_cfg["mlp_hidden"], n_features=model_cfg["n_features"],
        )
        fine_model = MLPAutoencoder(
            n_nodes=N_fine, param_dim=model_cfg["param_dim"],
            latent_dim=model_cfg["latent_dim"], filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"], adjacency=adjacency,
            mlp_hidden=model_cfg["mlp_hidden"], n_features=model_cfg["n_features"],
            coarse_model=coarse_model, downsampling_matrix=D, upsampling_matrix=U,
        )

        y_fine = torch.randn(B, N_fine, model_cfg["n_features"])
        with torch.no_grad():
            z_fine_only = fine_model.encoder(y_fine)   # without residual path
            z_residual  = fine_model.encode(y_fine)    # with coarse residual added

        # Residual path adds coarse contribution → z must differ
        assert not torch.allclose(z_residual, z_fine_only)

    def test_residual_decode_upsamples(self, grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
        """decode() upsamples the coarse decoded output before adding it."""
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        D, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)

        N_fine, N_coarse, B = len(fine_verts), len(coarse_verts), 2

        coarse_model = MLPAutoencoder(
            n_nodes=N_coarse, param_dim=model_cfg["param_dim"],
            latent_dim=model_cfg["latent_dim"], filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"], adjacency=coarse_adjacency,
            mlp_hidden=model_cfg["mlp_hidden"], n_features=model_cfg["n_features"],
        )
        fine_model = MLPAutoencoder(
            n_nodes=N_fine, param_dim=model_cfg["param_dim"],
            latent_dim=model_cfg["latent_dim"], filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"], adjacency=adjacency,
            mlp_hidden=model_cfg["mlp_hidden"], n_features=model_cfg["n_features"],
            coarse_model=coarse_model, downsampling_matrix=D, upsampling_matrix=U,
        )

        z = torch.randn(B, model_cfg["latent_dim"])
        with torch.no_grad():
            y_hat = fine_model.decode(z)

        assert y_hat.shape == (B, N_fine, model_cfg["n_features"])


# ---------------------------------------------------------------------------
# UpsamplingConvolution
# ---------------------------------------------------------------------------

class TestUpsamplingConvolution:
    def _make_upsampler(self, grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        N_fine = len(fine_verts)
        N_coarse = len(coarse_verts)
        _, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)

        coarse_model = MLPAutoencoder(
            n_nodes=N_coarse,
            param_dim=model_cfg["param_dim"],
            latent_dim=model_cfg["latent_dim"],
            filter_sizes=model_cfg["filter_sizes"],
            cheb_order=model_cfg["cheb_order"],
            adjacency=coarse_adjacency,
            mlp_hidden=model_cfg["mlp_hidden"],
            n_features=model_cfg["n_features"],
        )

        upsampler = UpsamplingConvolution(
            coarse_model=coarse_model,
            upsampling_matrix=U,
            n_nodes_fine=N_fine,
            n_nodes_coarse=N_coarse,
            n_features=model_cfg["n_features"],
        )
        return upsampler, N_fine, N_coarse

    def test_decode_shape(self, grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
        upsampler, N_fine, _ = self._make_upsampler(
            grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg
        )
        B = 3
        z = torch.randn(B, model_cfg["latent_dim"])
        y = upsampler.decode(z)
        assert y.shape == (B, N_fine, model_cfg["n_features"])

    def test_forward_shape(self, grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
        upsampler, N_fine, _ = self._make_upsampler(
            grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg
        )
        B = 3
        x = torch.randn(B, model_cfg["param_dim"])
        y = upsampler(x)
        assert y.shape == (B, N_fine, model_cfg["n_features"])

    def test_coarse_model_frozen(self, grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
        upsampler, _, _ = self._make_upsampler(
            grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg
        )
        for p in upsampler.coarse_model.parameters():
            assert not p.requires_grad

    def test_upsampler_weights_trainable(self, grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
        upsampler, _, _ = self._make_upsampler(
            grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg
        )
        for p in upsampler.upsampler.parameters():
            assert p.requires_grad

    def test_upsampler_init_equals_interpolated_coarse(self, grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
        """At init (upsampler weights=0) decode returns U @ coarse_decoded (no residual)."""
        from src.utils.sparse_utils import apply_sparse_to_batch
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        _, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        upsampler, N_fine, _ = self._make_upsampler(
            grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg
        )
        upsampler.eval()
        with torch.no_grad():
            z = torch.randn(2, model_cfg["latent_dim"])
            y_hat = upsampler.decode(z)
            # Expected: U @ coarse_decoder(z), residual term is zero at init
            y_coarse = upsampler.coarse_model.decoder(z)
            U_torch = upsampler.U
            y_expected = apply_sparse_to_batch(U_torch, y_coarse)
        assert torch.allclose(y_hat, y_expected, atol=1e-6)
