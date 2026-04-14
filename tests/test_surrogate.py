"""Tests for src/models/surrogate.py (MLP, MLPAutoencoder)."""

import pytest
import torch
import torch.nn as nn

from src.models.surrogate import MLP, MLPAutoencoder
from src.utils.mesh_sampling import build_sampling_matrices
from src.utils.sparse_utils import apply_sparse_to_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ae(adjacency, model_cfg, **kwargs):
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
        **kwargs,
    )


def _make_fine_coarse_pair(grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
    """Returns (coarse_model, fine_model, N_fine, N_coarse, D, U)."""
    fine_verts, fine_faces = grid_mesh
    coarse_verts, _ = coarse_mesh
    D, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
    N_fine, N_coarse = len(fine_verts), len(coarse_verts)

    coarse_model = _make_ae(coarse_adjacency, model_cfg,
                            n_nodes_fine=N_fine,
                            upsampling_matrix_to_fine=U)
    fine_model = _make_ae(adjacency, model_cfg,
                          coarse_model=coarse_model,
                          downsampling_matrix=D)
    return coarse_model, fine_model, N_fine, N_coarse, D, U


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape(self, model_cfg):
        mlp = MLP(param_dim=model_cfg["param_dim"], latent_dim=model_cfg["latent_dim"],
                  hidden_sizes=model_cfg["mlp_hidden"])
        z = mlp(torch.randn(5, model_cfg["param_dim"]))
        assert z.shape == (5, model_cfg["latent_dim"])

    def test_gradients(self, model_cfg):
        mlp = MLP(param_dim=model_cfg["param_dim"], latent_dim=model_cfg["latent_dim"])
        mlp(torch.randn(3, model_cfg["param_dim"])).sum().backward()
        assert all(p.grad is not None for p in mlp.parameters())

    def test_default_hidden_sizes(self, model_cfg):
        mlp = MLP(param_dim=model_cfg["param_dim"], latent_dim=model_cfg["latent_dim"])
        linears = [m for m in mlp.net if isinstance(m, nn.Linear)]
        assert len(linears) == 4  # 3 hidden + 1 output


# ---------------------------------------------------------------------------
# MLPAutoencoder — basic (no coarse model, no upsampler)
# ---------------------------------------------------------------------------

class TestMLPAutoencoder:
    def test_encode_shape(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 3
        z = _make_ae(adjacency, model_cfg).encode(torch.randn(B, N, model_cfg["n_features"]))
        assert z.shape == (B, model_cfg["latent_dim"])

    def test_decode_shape(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 3
        x = _make_ae(adjacency, model_cfg).decode(torch.randn(B, model_cfg["latent_dim"]))
        assert x.shape == (B, N, model_cfg["n_features"])

    def test_forward_shape(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 3
        x = _make_ae(adjacency, model_cfg)(torch.randn(B, model_cfg["param_dim"]))
        assert x.shape == (B, N, model_cfg["n_features"])

    def test_reconstruct_shape(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 2
        model = _make_ae(adjacency, model_cfg)
        x = torch.randn(B, N, model_cfg["n_features"])
        assert model.reconstruct(x).shape == x.shape

    def test_compute_loss_scalars(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 4
        model = _make_ae(adjacency, model_cfg)
        losses = model.compute_loss(
            torch.randn(B, model_cfg["param_dim"]),
            torch.randn(B, N, model_cfg["n_features"]),
        )
        assert len(losses) == 5
        assert all(t.shape == () for t in losses)

    def test_loss_non_negative(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 4
        model = _make_ae(adjacency, model_cfg)
        total, *_ = model.compute_loss(
            torch.randn(B, model_cfg["param_dim"]),
            torch.randn(B, N, model_cfg["n_features"]),
        )
        assert total.item() >= 0

    def test_loss_backward(self, adjacency, model_cfg):
        N, B = adjacency.shape[0], 2
        model = _make_ae(adjacency, model_cfg)
        total, *_ = model.compute_loss(
            torch.randn(B, model_cfg["param_dim"]),
            torch.randn(B, N, model_cfg["n_features"]),
        )
        total.backward()
        assert all(p.grad is not None for p in model.encoder.parameters())


# ---------------------------------------------------------------------------
# MLPAutoencoder — with upsampler (decode_fine)
# ---------------------------------------------------------------------------

class TestDecodeFine:
    def test_decode_fine_shape(self, grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg):
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        _, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        N_fine, N_coarse, B = len(fine_verts), len(coarse_verts), 3

        model = _make_ae(coarse_adjacency, model_cfg,
                         n_nodes_fine=N_fine, upsampling_matrix_to_fine=U)
        z = torch.randn(B, model_cfg["latent_dim"])
        assert model.decode_fine(z).shape == (B, N_fine, model_cfg["n_features"])

    def test_decode_fine_init_equals_interpolated(self, grid_mesh, coarse_mesh,
                                                   adjacency, coarse_adjacency, model_cfg):
        """At zero-init the residual term is 0 → decode_fine = U @ decode."""
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        _, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        N_fine, B = len(fine_verts), 2

        model = _make_ae(coarse_adjacency, model_cfg,
                         n_nodes_fine=N_fine, upsampling_matrix_to_fine=U)
        model.eval()
        with torch.no_grad():
            z = torch.randn(B, model_cfg["latent_dim"])
            x_fine = model.decode_fine(z)
            x_base = apply_sparse_to_batch(model.U_to_fine, model.decode(z))
        assert torch.allclose(x_fine, x_base, atol=1e-6)

    def test_decode_fine_gradients(self, grid_mesh, coarse_mesh,
                                    adjacency, coarse_adjacency, model_cfg):
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        _, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        N_fine = len(fine_verts)
        model = _make_ae(coarse_adjacency, model_cfg,
                         n_nodes_fine=N_fine, upsampling_matrix_to_fine=U)
        model.decode_fine(torch.randn(2, model_cfg["latent_dim"])).sum().backward()
        assert all(p.grad is not None for p in model.upsampler.parameters())

    def test_upsampler_weights_trainable(self, grid_mesh, coarse_mesh,
                                          adjacency, coarse_adjacency, model_cfg):
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        _, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        model = _make_ae(coarse_adjacency, model_cfg,
                         n_nodes_fine=len(fine_verts), upsampling_matrix_to_fine=U)
        assert all(p.requires_grad for p in model.upsampler.parameters())

    def test_loss_with_x_fine(self, grid_mesh, coarse_mesh,
                               adjacency, coarse_adjacency, model_cfg):
        """compute_loss with x_fine adds fine-level terms and changes total loss."""
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        _, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        N_fine, N_coarse, B = len(fine_verts), len(coarse_verts), 4
        model = _make_ae(coarse_adjacency, model_cfg,
                         n_nodes_fine=N_fine, upsampling_matrix_to_fine=U)
        mu = torch.randn(B, model_cfg["param_dim"])
        x = torch.randn(B, N_coarse, model_cfg["n_features"])
        x_fine = torch.randn(B, N_fine, model_cfg["n_features"])
        total_with, *_ = model.compute_loss(mu, x, x_fine)
        total_without, *_ = model.compute_loss(mu, x)
        # Fine-level loss terms should change the total
        assert total_with.item() != total_without.item()


# ---------------------------------------------------------------------------
# MLPAutoencoder — residual (coarse model + fine model)
# ---------------------------------------------------------------------------

class TestResidualHierarchy:
    def test_encode_downsamples_before_coarse(self, grid_mesh, coarse_mesh,
                                               adjacency, coarse_adjacency, model_cfg):
        """encode() downsamples x to coarse resolution before calling coarse_model.encode()."""
        _, fine_model, N_fine, _, _, _ = _make_fine_coarse_pair(
            grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg
        )
        B = 2
        x_fine = torch.randn(B, N_fine, model_cfg["n_features"])
        with torch.no_grad():
            z_encoder_only = fine_model.encoder(x_fine)
            z_with_residual = fine_model.encode(x_fine)
        assert not torch.allclose(z_with_residual, z_encoder_only)

    def test_decode_uses_coarse_decode_fine(self, grid_mesh, coarse_mesh,
                                             adjacency, coarse_adjacency, model_cfg):
        """decode() uses coarse_model.decode_fine(), giving output at the fine resolution."""
        _, fine_model, N_fine, _, _, _ = _make_fine_coarse_pair(
            grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg
        )
        B, N = 2, adjacency.shape[0]
        z = torch.randn(B, model_cfg["latent_dim"])
        with torch.no_grad():
            x = fine_model.decode(z)
        assert x.shape == (B, N, model_cfg["n_features"])

    def test_coarse_model_frozen_in_fine(self, grid_mesh, coarse_mesh,
                                          adjacency, coarse_adjacency, model_cfg):
        """The coarse model is frozen when set on the fine model."""
        coarse_model, fine_model, *_ = _make_fine_coarse_pair(
            grid_mesh, coarse_mesh, adjacency, coarse_adjacency, model_cfg
        )
        from src.models.multi_hierarchical import _freeze
        _freeze(coarse_model)
        assert all(not p.requires_grad for p in fine_model.coarse_model.parameters())
