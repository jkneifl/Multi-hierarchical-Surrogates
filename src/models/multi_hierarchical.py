"""
MultiHierarchicalSurrogate — orchestrates training across all hierarchy levels.

The surrogate is trained coarsest-first:
  1. At the coarsest level (l = coarsening_level): train a plain MLPAutoencoder
     that also contains an upsampler to level l-1.
  2. At each finer level (l = coarsening_level-1 … 1): train a new
     MLPAutoencoder on the data downsampled to level l, using the already-
     trained coarser model's decode_fine() as a frozen residual base, and
     adding its own upsampler to level l-1.

Each MLPAutoencoder therefore simultaneously learns:
  - a coarse-level autoencoder (encode/decode at level l)
  - an upsampler that maps its own decoded output to level l-1

Indexing convention
-------------------
adjacency_list[0]     — adjacency of the *finest* mesh (full resolution)
adjacency_list[l]     — adjacency of the mesh l levels coarser
downsampling_list[i]  — D_i: [N_{i+1}, N_i] maps level i → level i+1 (coarser)
upsampling_list[i]    — U_i: [N_i, N_{i+1}] maps level i+1 → level i (finer)

So adjacency_list[-1] is the coarsest level used in training.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.surrogate import MLPAutoencoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_downsample(x: np.ndarray, D: sp.spmatrix) -> np.ndarray:
    """
    Downsample state x from a fine mesh to a coarser mesh using scipy sparse D.

    Args:
        x : [B, N_fine, F]
        D : [N_coarse, N_fine] scipy sparse

    Returns:
        [B, N_coarse, F] numpy array
    """
    B, N, F = x.shape
    x_t = x.transpose(1, 0, 2).reshape(N, B * F)   # [N_fine, B*F]
    out = D @ x_t                                    # [N_coarse, B*F]
    M = D.shape[0]
    return out.reshape(M, B, F).transpose(1, 0, 2)  # [B, N_coarse, F]


def _freeze(module: nn.Module) -> None:
    """Freeze all parameters of a module (in-place)."""
    for p in module.parameters():
        p.requires_grad_(False)


def _train_autoencoder(
    model: MLPAutoencoder,
    mu_train: torch.Tensor,
    x_train: torch.Tensor,
    mu_test: torch.Tensor,
    x_test: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_path: str,
    device: torch.device,
    x_train_fine: Optional[torch.Tensor] = None,
    x_test_fine: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> MLPAutoencoder:
    """
    Train one MLPAutoencoder, saving the best checkpoint by validation loss.

    x_train_fine / x_test_fine are the ground-truth state fields one level
    finer than x_train / x_test.  When provided, the model's upsampler is
    trained jointly via additional fine-level reconstruction losses.
    """
    model = model.to(device)
    mu_train = mu_train.to(device)
    x_train  = x_train.to(device)
    mu_test  = mu_test.to(device)
    x_test   = x_test.to(device)

    if x_train_fine is not None:
        x_train_fine = x_train_fine.to(device)
        x_test_fine  = x_test_fine.to(device)
        dataset = TensorDataset(mu_train, x_train, x_train_fine)
    else:
        dataset = TensorDataset(mu_train, x_train)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.5
    )

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in loader:
            mu_b, x_b = batch[0], batch[1]
            x_fine_b  = batch[2] if len(batch) > 2 else None
            optimizer.zero_grad()
            total, _, _, _ = model.compute_loss(mu_b, x_b, x_fine_b)
            total.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_total, val_rec, val_x, val_z = model.compute_loss(
                mu_test, x_test, x_test_fine
            )

        scheduler.step(val_total)

        if val_total.item() < best_val_loss:
            best_val_loss = val_total.item()
            torch.save(model.state_dict(), checkpoint_path)

        if verbose and epoch % 100 == 0:
            print(
                f"  Epoch {epoch:4d}/{n_epochs}  "
                f"val_total={val_total.item():.6f}  "
                f"val_rec={val_rec.item():.6f}  "
                f"val_x={val_x.item():.6f}  "
                f"val_z={val_z.item():.6f}  "
                f"best={best_val_loss:.6f}"
            )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


# ---------------------------------------------------------------------------
# MultiHierarchicalSurrogate
# ---------------------------------------------------------------------------

class MultiHierarchicalSurrogate:
    """
    Multi-hierarchical graph surrogate model.

    Parameters
    ----------
    adjacency_list : list of scipy sparse matrices
        [A_0, A_1, ..., A_L]  where A_0 is the finest mesh and A_L is the
        coarsest.
    downsampling_list : list of scipy sparse matrices
        [D_0, D_1, ..., D_{L-1}]  D_i : [N_{i+1}, N_i] (fine→coarse).
    upsampling_list : list of scipy sparse matrices
        [U_0, U_1, ..., U_{L-1}]  U_i : [N_i, N_{i+1}] (coarse→fine).
    reduced_order : int
        Latent dimension (default 4).
    n_features : int
        Node feature size, 3 for xyz displacements (default 3).
    filter_sizes : list of int
        ChebConv channel widths (default [6, 12, 24]).
    cheb_order : int
        Chebyshev polynomial order (default 3).
    mlp_hidden : list of int
        Hidden layer widths for the MLP (default [64, 64, 64]).
    lambda_rec, lambda_x, lambda_z : float
        Loss weights (defaults 1, 1, 0).
    param_dim : int
        Dimensionality of the parameter vector μ including time (default 4).
    device : torch.device or str, optional
        Compute device (default: 'cuda' if available else 'cpu').
    """

    def __init__(
        self,
        adjacency_list: List[sp.spmatrix],
        downsampling_list: List[sp.spmatrix],
        upsampling_list: List[sp.spmatrix],
        reduced_order: int = 4,
        n_features: int = 3,
        filter_sizes: List[int] = None,
        cheb_order: int = 3,
        mlp_hidden: List[int] = None,
        lambda_rec: float = 1.0,
        lambda_x: float = 1.0,
        lambda_z: float = 0.0,
        param_dim: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.adjacency_list = adjacency_list
        self.downsampling_list = downsampling_list
        self.upsampling_list = upsampling_list
        self.reduced_order = reduced_order
        self.n_features = n_features
        self.filter_sizes = filter_sizes if filter_sizes is not None else [6, 12, 24]
        self.cheb_order = cheb_order
        self.mlp_hidden = mlp_hidden if mlp_hidden is not None else [64, 64, 64]
        self.lambda_rec = lambda_rec
        self.lambda_x = lambda_x
        self.lambda_z = lambda_z
        self.param_dim = param_dim

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Will be populated during fit().
        # autoencoders[l] is the MLPAutoencoder for level l.
        # Each model also carries an upsampler to level l-1 inside it.
        self.autoencoders: List[Optional[MLPAutoencoder]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _downsample_to_level(self, x: np.ndarray, level: int) -> np.ndarray:
        """
        Downsample full-resolution state x to the requested coarsening level by
        composing the downsampling matrices D_0, D_1, ..., D_{level-1}.

        level=0 → no downsampling (return x as-is)
        level=1 → apply D_0
        level=2 → apply D_1 ∘ D_0
        ...

        Args:
            x     : [B, N_fine, F] numpy array
            level : int ≥ 0

        Returns:
            [B, N_level, F] numpy array
        """
        out = x
        for i in range(level):
            out = _np_downsample(out, self.downsampling_list[i])
        return out

    def _n_nodes_at_level(self, level: int, x_fine: np.ndarray) -> int:
        """Number of nodes at coarsening level ``level``."""
        if level == 0:
            return x_fine.shape[1]
        return int(self.downsampling_list[level - 1].shape[0])

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        mu_train: np.ndarray,
        x_train: np.ndarray,
        mu_test: np.ndarray,
        x_test: np.ndarray,
        coarsening_level: int = 3,
        n_epochs: int = 1500,
        batch_size: int = 32,
        lr: float = 1e-3,
        save_dir: str = "checkpoints",
        verbose: bool = True,
    ) -> None:
        """
        Train the full hierarchy from coarsest to finest.

        Parameters
        ----------
        mu_train : [N_train, param_dim]  parameter inputs μ (numpy)
        x_train  : [N_train, N_nodes_full, 3]  full-resolution state fields
        mu_test  : [N_test,  param_dim]
        x_test   : [N_test,  N_nodes_full, 3]
        coarsening_level : int
            Total number of coarsening levels to use (≥ 1).
        n_epochs : int
            Training epochs per level autoencoder.
        batch_size : int
        lr : float
        save_dir : str
            Directory where model checkpoints are written.
        verbose : bool
        """
        os.makedirs(save_dir, exist_ok=True)

        mu_train_t = torch.from_numpy(mu_train.astype(np.float32))
        mu_test_t  = torch.from_numpy(mu_test.astype(np.float32))

        self.autoencoders = [None] * (coarsening_level + 1)

        coarse_model: Optional[MLPAutoencoder] = None

        # Train coarsest-first so each model can reference a frozen coarser neighbour
        for level in range(coarsening_level, 0, -1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Training MLPAutoencoder at coarsening level {level}")
                print(f"{'='*60}")

            x_train_level = self._downsample_to_level(x_train, level)
            x_test_level  = self._downsample_to_level(x_test,  level)
            # Fine data for the upsampler (one level finer = level-1)
            x_train_fine  = self._downsample_to_level(x_train, level - 1)
            x_test_fine   = self._downsample_to_level(x_test,  level - 1)

            x_train_t      = torch.from_numpy(x_train_level.astype(np.float32))
            x_test_t       = torch.from_numpy(x_test_level.astype(np.float32))
            x_train_fine_t = torch.from_numpy(x_train_fine.astype(np.float32))
            x_test_fine_t  = torch.from_numpy(x_test_fine.astype(np.float32))

            n_nodes      = x_train_t.shape[1]
            n_nodes_fine = x_train_fine_t.shape[1]
            adjacency    = self.adjacency_list[level]

            if verbose:
                print(f"  Nodes at level {level}: {n_nodes}  →  fine: {n_nodes_fine}")

            # D: this level → coarser (for encode residual path)
            D = self.downsampling_list[level] if coarse_model is not None else None
            # U_to_fine: this level → finer (for decode_fine)
            U_to_fine = self.upsampling_list[level - 1]

            model = MLPAutoencoder(
                n_nodes=n_nodes,
                param_dim=self.param_dim,
                latent_dim=self.reduced_order,
                filter_sizes=self.filter_sizes,
                cheb_order=self.cheb_order,
                adjacency=adjacency,
                mlp_hidden=self.mlp_hidden,
                n_features=self.n_features,
                lambda_rec=self.lambda_rec,
                lambda_x=self.lambda_x,
                lambda_z=self.lambda_z,
                coarse_model=coarse_model,
                downsampling_matrix=D,
                n_nodes_fine=n_nodes_fine,
                upsampling_matrix_to_fine=U_to_fine,
            )

            checkpoint_path = os.path.join(save_dir, f"autoencoder_level_{level}.pt")

            model = _train_autoencoder(
                model=model,
                mu_train=mu_train_t,
                x_train=x_train_t,
                mu_test=mu_test_t,
                x_test=x_test_t,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                checkpoint_path=checkpoint_path,
                device=self.device,
                x_train_fine=x_train_fine_t,
                x_test_fine=x_test_fine_t,
                verbose=verbose,
            )

            _freeze(model)
            self.autoencoders[level] = model
            coarse_model = model

        if verbose:
            print("\nTraining complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, mu: np.ndarray, level: int = 0) -> np.ndarray:
        """
        Predict state fields at a given coarsening level.

        Parameters
        ----------
        mu    : [N, param_dim] parameter inputs μ (numpy)
        level : int
            0 = finest mesh (-1 also maps to finest).  Positive values give
            coarser predictions directly from the MLPAutoencoder.

        Returns
        -------
        x_pred : [N, N_nodes_level, 3] numpy array
        """
        if level == -1:
            level = 0

        mu_t = torch.from_numpy(mu.astype(np.float32)).to(self.device)

        # level 0 = full resolution: use the finest trained autoencoder's
        # decode_fine(), which maps its coarse prediction to level 0.
        # level > 0: use that autoencoder's decode() directly.
        if level == 0:
            ae = self.autoencoders[1]
            if ae is None:
                raise ValueError("No model at level 1. Make sure fit() has been called.")
            ae.eval()
            with torch.no_grad():
                x_pred = ae.decode_fine(ae.predict_latent(mu_t))
        else:
            ae = self.autoencoders[level]
            if ae is None:
                raise ValueError(
                    f"No model available for level={level}. "
                    "Make sure fit() has been called."
                )
            ae.eval()
            with torch.no_grad():
                x_pred = ae(mu_t)

        return x_pred.cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save all model state dicts and configuration to a single file.

        Args:
            path: file path (e.g. 'model.pt')
        """
        state = {
            "config": {
                "reduced_order": self.reduced_order,
                "n_features": self.n_features,
                "filter_sizes": self.filter_sizes,
                "cheb_order": self.cheb_order,
                "mlp_hidden": self.mlp_hidden,
                "lambda_rec": self.lambda_rec,
                "lambda_x": self.lambda_x,
                "lambda_z": self.lambda_z,
                "param_dim": self.param_dim,
            },
            "autoencoders": {
                l: m.state_dict()
                for l, m in enumerate(self.autoencoders)
                if m is not None
            },
        }
        torch.save(state, path)
        print(f"Saved model to {path}")

    def load(self, path: str) -> None:
        """
        Load model state dicts from a file written by ``save``.

        The adjacency/downsampling/upsampling lists must already have been
        provided to the constructor so that the module architectures can be
        re-instantiated.

        Args:
            path: file path written by ``save``
        """
        checkpoint = torch.load(path, map_location=self.device)
        cfg = checkpoint["config"]
        ae_states = checkpoint["autoencoders"]

        self.autoencoders = [None] * (max(ae_states.keys()) + 1)

        # Rebuild coarsest-first so each model can reference its frozen coarser neighbour
        coarse_model = None
        for level in sorted(ae_states.keys(), reverse=True):
            n_nodes     = self.downsampling_list[level - 1].shape[0]
            n_nodes_fine = (self.downsampling_list[level - 2].shape[0]
                            if level >= 2 else self.downsampling_list[level - 1].shape[1])
            adjacency   = self.adjacency_list[level]
            D           = self.downsampling_list[level] if coarse_model is not None else None
            U_to_fine   = self.upsampling_list[level - 1]

            model = MLPAutoencoder(
                n_nodes=n_nodes,
                param_dim=cfg["param_dim"],
                latent_dim=cfg["reduced_order"],
                filter_sizes=cfg["filter_sizes"],
                cheb_order=cfg["cheb_order"],
                adjacency=adjacency,
                mlp_hidden=cfg["mlp_hidden"],
                n_features=cfg["n_features"],
                lambda_rec=cfg["lambda_rec"],
                lambda_x=cfg["lambda_x"],
                lambda_z=cfg["lambda_z"],
                coarse_model=coarse_model,
                downsampling_matrix=D,
                n_nodes_fine=n_nodes_fine,
                upsampling_matrix_to_fine=U_to_fine,
            ).to(self.device)

            model.load_state_dict(ae_states[level])
            _freeze(model)
            self.autoencoders[level] = model
            coarse_model = model

        print(f"Loaded model from {path}")
