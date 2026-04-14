"""
MLPAutoencoder — the building block of the multi-hierarchical surrogate.

Combines a graph autoencoder (GraphEncoder + GraphDecoder) with a small MLP
that maps simulation parameters μ to the latent space z.  Optionally wraps a
coarser model to form a residual hierarchy, and optionally contains a linear
upsampler that maps this level's decoded output to the next finer level.

                         ┌─────────────────────────────────┐
   parameters μ ──► MLP ►│ z  ──► decoder ──► x_this       │
                         │                                  │
                         │       decode_fine (if present):  │
                         │  x_this ──► U·x + upsampler(x)  │
                         └─────────────────────────────────┘

Training loss:
    total = λ_rec · L_rec  +  λ_x · L_x  +  λ_z · L_z  +  λ_up · L_up
    L_up  = MSE(x_fine, decode_fine(encode(x)))
          + MSE(x_fine, decode_fine(mlp(mu)))          (zero when x_fine absent)
"""

from __future__ import annotations

import scipy.sparse as sp
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from src.models.graph_autoencoder import GraphEncoder, GraphDecoder
from src.utils.sparse_utils import scipy_sparse_to_torch, apply_sparse_to_batch


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Small fully-connected network mapping parameters μ → latent vector z."""

    def __init__(
        self,
        param_dim: int,
        latent_dim: int,
        hidden_sizes: List[int] = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        layers: List[nn.Module] = []
        in_dim = param_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.net(mu)


# ---------------------------------------------------------------------------
# MLPAutoencoder
# ---------------------------------------------------------------------------

class MLPAutoencoder(nn.Module):
    """
    Graph autoencoder + parameter MLP with optional coarse residual and
    optional upsampler to the next finer level.

    Parameters
    ----------
    n_nodes : int
        Number of nodes at this level.
    param_dim : int
        Dimension of the parameter+time input μ.
    latent_dim : int
        Latent bottleneck size.
    filter_sizes : list of int
        ChebConv channel widths.
    cheb_order : int
        Chebyshev polynomial order.
    adjacency : scipy sparse [N, N]
    mlp_hidden : list of int
    n_features : int
        Node feature dimension (3 for xyz displacements).
    lambda_rec, lambda_x, lambda_z : float
        Loss weights.
    coarse_model : MLPAutoencoder, optional
        Frozen coarser model.  When set, downsampling_matrix must be provided.
    downsampling_matrix : scipy sparse [N_coarser, N_this], optional
        Used in encode() to map x to the coarser level before calling
        coarse_model.encode().
    n_nodes_fine : int, optional
        Number of nodes at the next *finer* level.  Required together with
        upsampling_matrix_to_fine to enable decode_fine().
    upsampling_matrix_to_fine : scipy sparse [N_fine, N_this], optional
        Fixed nearest-neighbour interpolation from this level to the finer
        level.  The trainable upsampler learns a correction on top.
    lambda_up : float
        Weight for the upsampling loss terms L_rec_fine and L_x_fine
        (default 1.0).
    """

    def __init__(
        self,
        n_nodes: int,
        param_dim: int,
        latent_dim: int,
        filter_sizes: List[int],
        cheb_order: int,
        adjacency: sp.spmatrix,
        mlp_hidden: List[int] = None,
        n_features: int = 3,
        lambda_rec: float = 1.0,
        lambda_x: float = 1.0,
        lambda_z: float = 0.0,
        lambda_up: float = 1.0,
        coarse_model: Optional[MLPAutoencoder] = None,
        downsampling_matrix=None,
        n_nodes_fine: int = None,
        upsampling_matrix_to_fine=None,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [64, 64, 64]

        self.n_nodes     = n_nodes
        self.n_features  = n_features
        self.latent_dim  = latent_dim
        self.lambda_rec  = lambda_rec
        self.lambda_x    = lambda_x
        self.lambda_z    = lambda_z
        self.lambda_up   = lambda_up
        self.coarse_model = coarse_model
        self.n_nodes_fine = n_nodes_fine

        self.encoder = GraphEncoder(
            n_nodes=n_nodes, in_features=n_features, latent_dim=latent_dim,
            filter_sizes=filter_sizes, cheb_order=cheb_order, adjacency=adjacency,
        )
        self.decoder = GraphDecoder(
            n_nodes=n_nodes, out_features=n_features, latent_dim=latent_dim,
            filter_sizes=filter_sizes, cheb_order=cheb_order, adjacency=adjacency,
        )
        self.mlp = MLP(param_dim=param_dim, latent_dim=latent_dim, hidden_sizes=mlp_hidden)

        # D: downsample this level → coarser (for encode residual)
        D = scipy_sparse_to_torch(downsampling_matrix.astype("float32")) if downsampling_matrix is not None else None
        self.register_buffer("D", D)

        # Upsampler to next finer level (optional)
        if upsampling_matrix_to_fine is not None:
            U = scipy_sparse_to_torch(upsampling_matrix_to_fine.astype("float32"))
            self.register_buffer("U_to_fine", U)
            self.upsampler = nn.Linear(n_nodes * n_features, n_nodes_fine * n_features)
            nn.init.zeros_(self.upsampler.weight)
            nn.init.zeros_(self.upsampler.bias)
        else:
            self.register_buffer("U_to_fine", None)
            self.upsampler = None

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, F] → [B, latent_dim]"""
        z = self.encoder(x)
        if self.coarse_model is not None:
            x_coarse = apply_sparse_to_batch(self.D, x)
            z = z + self.coarse_model.encode(x_coarse)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim] → [B, N, F]"""
        x = self.decoder(z)
        if self.coarse_model is not None:
            x = x + self.coarse_model.decode_fine(z)
        return x

    def decode_fine(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode z to the next finer level using fixed interpolation + learned
        residual correction.  [B, latent_dim] → [B, N_fine, F]
        """
        B = z.shape[0]
        x = self.decode(z)                                               # [B, N, F]
        x_base     = apply_sparse_to_batch(self.U_to_fine, x)           # [B, N_fine, F]
        x_residual = self.upsampler(x.reshape(B, -1))
        x_residual = x_residual.reshape(B, self.n_nodes_fine, self.n_features)
        return x_base + x_residual

    def predict_latent(self, mu: torch.Tensor) -> torch.Tensor:
        """[B, param_dim] → [B, latent_dim]"""
        return self.mlp(mu)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """[B, param_dim] → [B, N, F]"""
        return self.decode(self.predict_latent(mu))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, F] → [B, N, F]"""
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        mu: torch.Tensor,
        x: torch.Tensor,
        x_fine: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined training loss.

        L_rec = MSE(x,  decode(encode(x)))
        L_x   = MSE(x,  decode(mlp(mu)))
        L_z   = MSE(encode(x).detach(), mlp(mu))
        L_up  = MSE(x_fine, decode_fine(encode(x)))
              + MSE(x_fine, decode_fine(mlp(mu)))   (zero when x_fine is None)

        total = λ_rec·L_rec + λ_x·L_x + λ_z·L_z + λ_up·L_up

        Returns (total, L_rec, L_x, L_z, L_up).
        """
        mse = nn.functional.mse_loss

        z_enc  = self.encode(x)
        x_rec  = self.decode(z_enc)
        L_rec  = mse(x_rec, x)

        z_mlp  = self.predict_latent(mu)
        x_pred = self.decode(z_mlp)
        L_x    = mse(x_pred, x)

        L_z    = mse(z_mlp, z_enc.detach())

        total = self.lambda_rec * L_rec + self.lambda_x * L_x + self.lambda_z * L_z

        L_up = torch.zeros((), device=x.device)
        if x_fine is not None and self.upsampler is not None:
            L_up = (mse(self.decode_fine(z_enc), x_fine)
                  + mse(self.decode_fine(z_mlp), x_fine))
            total = total + self.lambda_up * L_up

        return total, L_rec, L_x, L_z, L_up
