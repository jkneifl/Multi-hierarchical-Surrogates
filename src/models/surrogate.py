"""
MLPAutoencoder and UpsamplingConvolution — the two building blocks of the
multi-hierarchical surrogate.

MLPAutoencoder
--------------
Combines a graph autoencoder (GraphEncoder + GraphDecoder) with a small MLP
that maps simulation parameters to the latent space.  Optionally wraps a
coarser model to form a residual (correction) hierarchy.

Training loss:
    total = λ_rec * L_rec + λ_x * L_x + λ_z * L_z

where
    L_rec = MSE(y,  decoder(encoder(y)))          — reconstruction
    L_x   = MSE(y,  decoder(mlp(x)))              — prediction quality
    L_z   = MSE(encoder(y).detach(), mlp(x))      — latent alignment

UpsamplingConvolution
---------------------
Wraps a frozen coarser MLPAutoencoder together with a single trainable linear
"upsampler" that corrects the coarse prediction onto the finer mesh.
"""

from __future__ import annotations

import scipy.sparse as sp
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from src.layers.graph_conv import ChebConv
from src.models.graph_autoencoder import GraphEncoder, GraphDecoder
from src.utils.sparse_utils import scipy_sparse_to_torch, apply_sparse_to_batch


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Small fully-connected network that maps (params, time) → latent vector.

    Architecture:
        Linear(param_dim) → ELU → Linear(64) → ELU → ... → Linear(latent_dim)

    Parameters
    ----------
    param_dim : int
        Dimension of the input (number of parameters + 1 for time).
    latent_dim : int
        Output dimension (size of the latent bottleneck).
    hidden_sizes : list of int
        Widths of the hidden layers (default [64, 64, 64]).
    """

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
        # Final linear layer — no activation
        layers.append(nn.Linear(in_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, param_dim]

        Returns:
            z: [B, latent_dim]
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# MLPAutoencoder
# ---------------------------------------------------------------------------

class MLPAutoencoder(nn.Module):
    """
    Graph autoencoder + parameter MLP with optional coarse residual model.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the mesh at this level.
    param_dim : int
        Dimensionality of the parameter+time input to the MLP.
    latent_dim : int
        Size of the latent bottleneck vector.
    filter_sizes : list of int
        Channel widths for the ChebConv layers, e.g. [6, 12, 24].
    cheb_order : int
        Chebyshev polynomial order.
    adjacency : scipy sparse matrix [N, N]
        Adjacency of the mesh at this level.
    mlp_hidden : list of int
        Hidden layer widths for the MLP.
    n_features : int
        Number of node features (3 for x/y/z displacements).
    lambda_rec : float
        Weight for reconstruction loss.
    lambda_x : float
        Weight for prediction (decoder(mlp(x))) loss.
    lambda_z : float
        Weight for latent alignment loss.
    coarse_model : MLPAutoencoder or UpsamplingConvolution, optional
        Frozen coarser model used as a residual base.
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
        coarse_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [64, 64, 64]

        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.lambda_rec = lambda_rec
        self.lambda_x = lambda_x
        self.lambda_z = lambda_z
        self.coarse_model = coarse_model  # may be None

        self.encoder = GraphEncoder(
            n_nodes=n_nodes,
            in_features=n_features,
            latent_dim=latent_dim,
            filter_sizes=filter_sizes,
            cheb_order=cheb_order,
            adjacency=adjacency,
        )
        self.decoder = GraphDecoder(
            n_nodes=n_nodes,
            out_features=n_features,
            latent_dim=latent_dim,
            filter_sizes=filter_sizes,
            cheb_order=cheb_order,
            adjacency=adjacency,
        )
        self.mlp = MLP(
            param_dim=param_dim,
            latent_dim=latent_dim,
            hidden_sizes=mlp_hidden,
        )

    # ------------------------------------------------------------------
    # Encode / decode with optional coarse residual
    # ------------------------------------------------------------------

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """
        Encode a displacement field to a latent vector.

        When a coarse model is present the latent vectors are *added* (residual
        hierarchy): z = encoder(y) + coarse_model.encode(y).

        Args:
            y: [B, N, 3] displacement field at this level

        Returns:
            z: [B, latent_dim]
        """
        z = self.encoder(y)
        if self.coarse_model is not None:
            z = z + self.coarse_model.encode(y)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent vector to a displacement field.

        When a coarse model is present, the decoded fields are *added*
        (residual hierarchy): y_hat = decoder(z) + coarse_model.decode(z).

        Args:
            z: [B, latent_dim]

        Returns:
            y_hat: [B, N, 3]
        """
        y_hat = self.decoder(z)
        if self.coarse_model is not None:
            y_hat = y_hat + self.coarse_model.decode(z)
        return y_hat

    def predict_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map simulation parameters to a latent vector via the MLP.

        Args:
            x: [B, param_dim]

        Returns:
            z: [B, latent_dim]
        """
        return self.mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full prediction pipeline: parameters → latent → displacement field.

        Args:
            x: [B, param_dim]

        Returns:
            y_hat: [B, N, 3]
        """
        z = self.predict_latent(x)
        return self.decode(z)

    def reconstruct(self, y: torch.Tensor) -> torch.Tensor:
        """
        Autoencoder reconstruction (encode then decode).

        Args:
            y: [B, N, 3]

        Returns:
            y_rec: [B, N, 3]
        """
        z = self.encode(y)
        return self.decode(z)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined training loss.

        L_rec = MSE(y, decoder(encoder(y)))
        L_x   = MSE(y, decoder(mlp(x)))
        L_z   = MSE(encoder(y).detach(), mlp(x))

        Args:
            x: [B, param_dim] parameter inputs
            y: [B, N, 3]      ground-truth displacement fields

        Returns:
            (total, L_rec, L_x, L_z) — all scalar tensors
        """
        mse = nn.functional.mse_loss

        # Reconstruction branch
        z_enc = self.encode(y)            # [B, latent_dim]
        y_rec = self.decode(z_enc)        # [B, N, 3]
        L_rec = mse(y_rec, y)

        # Prediction branch
        z_mlp = self.predict_latent(x)   # [B, latent_dim]
        y_pred = self.decode(z_mlp)      # [B, N, 3]
        L_x = mse(y_pred, y)

        # Latent alignment (stop-gradient on encoder side)
        L_z = mse(z_mlp, z_enc.detach())

        total = self.lambda_rec * L_rec + self.lambda_x * L_x + self.lambda_z * L_z
        return total, L_rec, L_x, L_z


# ---------------------------------------------------------------------------
# UpsamplingConvolution
# ---------------------------------------------------------------------------

class UpsamplingConvolution(nn.Module):
    """
    One-level upsampling module that refines a coarser prediction to a finer
    mesh via a trainable linear correction.

    The coarser model is frozen; only the linear upsampler is trained.

    Parameters
    ----------
    coarse_model : MLPAutoencoder
        The frozen model at the next coarser level.
    downsampling_matrix : scipy sparse [N_coarse, N_fine]
        D matrix that maps fine-mesh nodes to coarse-mesh nodes.
        Used internally to obtain coarse-level inputs from fine-level data.
    n_nodes_fine : int
        Number of nodes on the fine mesh.
    n_nodes_coarse : int
        Number of nodes on the coarse mesh (= downsampling_matrix.shape[0]).
    n_features : int
        Number of node features (default 3).
    """

    def __init__(
        self,
        coarse_model: MLPAutoencoder,
        downsampling_matrix: sp.spmatrix,
        n_nodes_fine: int,
        n_nodes_coarse: int,
        n_features: int = 3,
    ):
        super().__init__()
        self.n_nodes_fine = n_nodes_fine
        self.n_nodes_coarse = n_nodes_coarse
        self.n_features = n_features

        # Freeze the coarse model
        self.coarse_model = coarse_model
        for p in self.coarse_model.parameters():
            p.requires_grad_(False)

        # Store D as a non-trainable sparse buffer
        D_torch = scipy_sparse_to_torch(downsampling_matrix.astype("float32"))
        self.register_buffer("D", D_torch)

        # Trainable upsampler: coarse flat → fine flat
        # Initialise weights to zero so it starts as a pure coarse prediction
        self.upsampler = nn.Linear(
            n_nodes_coarse * n_features,
            n_nodes_fine * n_features,
            bias=True,
        )
        nn.init.zeros_(self.upsampler.weight)
        nn.init.zeros_(self.upsampler.bias)

    # ------------------------------------------------------------------

    def encode(self, y_fine: torch.Tensor) -> torch.Tensor:
        """
        Downsample y_fine to the coarse level and encode.

        Calls ``coarse_model.encoder`` directly (NOT ``coarse_model.encode``)
        to avoid double-adding residuals from deeper levels.

        Args:
            y_fine: [B, N_fine, F]

        Returns:
            z: [B, latent_dim]
        """
        y_coarse = apply_sparse_to_batch(self.D, y_fine)   # [B, N_coarse, F]
        return self.coarse_model.encoder(y_coarse)          # encoder only, no residual

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent to fine mesh via coarse decoder + linear upsampler.

        Steps:
          1. coarse_model.decoder(z) → y_coarse_hat [B, N_coarse, F]
          2. flatten → [B, N_coarse*F]
          3. upsampler → [B, N_fine*F]
          4. reshape → [B, N_fine, F]

        Args:
            z: [B, latent_dim]

        Returns:
            y_fine_hat: [B, N_fine, F]
        """
        B = z.shape[0]
        y_coarse_hat = self.coarse_model.decoder(z)                    # [B, N_coarse, F]
        flat_coarse = y_coarse_hat.reshape(B, self.n_nodes_coarse * self.n_features)
        flat_fine = self.upsampler(flat_coarse)                        # [B, N_fine*F]
        return flat_fine.reshape(B, self.n_nodes_fine, self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict fine-mesh displacements from parameters.

        Calls the frozen coarse MLP then the upsampling decoder.

        Args:
            x: [B, param_dim]

        Returns:
            y_fine_hat: [B, N_fine, F]
        """
        z = self.coarse_model.mlp(x)   # latent from the coarse MLP
        return self.decode(z)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        Y_coarse: torch.Tensor,
        Y_fine: torch.Tensor,
        epochs: int = 500,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> None:
        """
        Train the upsampler weights while keeping the coarse model frozen.

        Loss: MSE(Y_fine, decode(encoder(Y_coarse)))
        where encoder = frozen coarse_model.encoder.

        Args:
            Y_coarse : [N_samples, N_coarse, F] coarse displacement fields
            Y_fine   : [N_samples, N_fine,   F] corresponding fine fields
            epochs   : number of training epochs
            batch_size: mini-batch size
            lr       : Adam learning rate
            device   : torch device (defaults to the device of upsampler weights)
            verbose  : print epoch losses every 100 epochs
        """
        if device is None:
            device = next(self.upsampler.parameters()).device

        self.to(device)
        Y_coarse = Y_coarse.to(device)
        Y_fine = Y_fine.to(device)

        optimizer = torch.optim.Adam(
            self.upsampler.parameters(), lr=lr
        )

        N = Y_coarse.shape[0]
        mse = nn.functional.mse_loss

        for epoch in range(1, epochs + 1):
            self.train()
            perm = torch.randperm(N, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start: start + batch_size]
                yc = Y_coarse[idx]
                yf = Y_fine[idx]

                # Encode coarse (frozen encoder) → latent
                with torch.no_grad():
                    z = self.coarse_model.encoder(yc)

                # Decode to fine (trainable upsampler)
                yf_pred = self.decode(z)

                loss = mse(yf_pred, yf)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if verbose and epoch % 100 == 0:
                print(f"  UpsamplingConvolution epoch {epoch}/{epochs}  "
                      f"loss={epoch_loss / n_batches:.6f}")
