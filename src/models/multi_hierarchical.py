"""
MultiHierarchicalSurrogate — orchestrates training across all hierarchy levels.

The surrogate is trained coarsest-first:
  1. At the coarsest level (l = coarsening_level): train a plain MLPAutoencoder.
  2. At each finer level (l = coarsening_level-1 … 1): train a new
     MLPAutoencoder on the data downsampled to level l, using the already-
     trained coarser model as a frozen residual base.
  3. After each MLPAutoencoder is trained, if there is a still finer level,
     fit an UpsamplingConvolution to bridge one level.

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

from src.models.graph_autoencoder import GraphEncoder, GraphDecoder
from src.models.surrogate import MLPAutoencoder, UpsamplingConvolution
from src.utils.sparse_utils import scipy_sparse_to_torch, apply_sparse_to_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_downsample(y: np.ndarray, D: sp.spmatrix) -> np.ndarray:
    """
    Downsample y from a fine mesh to a coarser mesh using a scipy sparse D.

    Args:
        y : [B, N_fine, F]
        D : [N_coarse, N_fine] scipy sparse

    Returns:
        [B, N_coarse, F] numpy array
    """
    B, N, F = y.shape
    y_t = y.transpose(1, 0, 2).reshape(N, B * F)   # [N_fine, B*F]
    out = D @ y_t                                    # [N_coarse, B*F]
    M = D.shape[0]
    return out.reshape(M, B, F).transpose(1, 0, 2)  # [B, N_coarse, F]


def _freeze(module: nn.Module) -> None:
    """Freeze all parameters of a module (in-place)."""
    for p in module.parameters():
        p.requires_grad_(False)


def _train_autoencoder(
    model: MLPAutoencoder,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_path: str,
    device: torch.device,
    verbose: bool = True,
) -> MLPAutoencoder:
    """
    Train one MLPAutoencoder, saving the best checkpoint by validation loss.

    Returns the model loaded with the best weights.
    """
    model = model.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.5
    )

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        # ----- train -----
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            total, _, _, _ = model.compute_loss(x_batch, y_batch)
            total.backward()
            optimizer.step()

        # ----- validate -----
        model.eval()
        with torch.no_grad():
            val_total, val_rec, val_x, val_z = model.compute_loss(X_test, Y_test)

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

    # Reload best weights
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
        Dimensionality of the parameter vector including time (default 4).
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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Will be populated during fit()
        # autoencoders[l] is the MLPAutoencoder for level l
        # upsamplers[l] is the UpsamplingConvolution from level l+1 to l
        self.autoencoders: List[Optional[MLPAutoencoder]] = []
        self.upsamplers: List[Optional[UpsamplingConvolution]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _downsample_to_level(self, Y: np.ndarray, level: int) -> np.ndarray:
        """
        Downsample full-resolution Y to the requested coarsening level by
        composing the downsampling matrices D_0, D_1, ..., D_{level-1}.

        level=0 → no downsampling (return Y as-is)
        level=1 → apply D_0
        level=2 → apply D_1 ∘ D_0
        ...

        Args:
            Y     : [B, N_fine, F] numpy array
            level : int ≥ 0

        Returns:
            [B, N_level, F] numpy array
        """
        y = Y
        for i in range(level):
            y = _np_downsample(y, self.downsampling_list[i])
        return y

    def _n_nodes_at_level(self, level: int, Y_fine: np.ndarray) -> int:
        """Number of nodes at coarsening level ``level``."""
        if level == 0:
            return Y_fine.shape[1]
        return int(self.downsampling_list[level - 1].shape[0])

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
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
        X_train : [N_train, param_dim]  parameter inputs (numpy)
        Y_train : [N_train, N_nodes_full, 3]  full-resolution displacements
        X_test  : [N_test,  param_dim]
        Y_test  : [N_test,  N_nodes_full, 3]
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

        X_train_t = torch.from_numpy(X_train.astype(np.float32))
        X_test_t = torch.from_numpy(X_test.astype(np.float32))

        # Reset stored models
        self.autoencoders = [None] * (coarsening_level + 1)
        self.upsamplers = [None] * coarsening_level

        coarse_model: Optional[MLPAutoencoder] = None

        # Train from coarsest (level=coarsening_level) down to finest (level=1)
        for level in range(coarsening_level, 0, -1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Training MLPAutoencoder at coarsening level {level}")
                print(f"{'='*60}")

            # Downsample data to this level
            Y_train_level = self._downsample_to_level(Y_train, level)  # numpy
            Y_test_level = self._downsample_to_level(Y_test, level)

            Y_train_t = torch.from_numpy(Y_train_level.astype(np.float32))
            Y_test_t = torch.from_numpy(Y_test_level.astype(np.float32))

            n_nodes = Y_train_t.shape[1]
            adjacency = self.adjacency_list[level]

            if verbose:
                print(f"  Nodes at level {level}: {n_nodes}")

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
            )

            checkpoint_path = os.path.join(save_dir, f"autoencoder_level_{level}.pt")

            model = _train_autoencoder(
                model=model,
                X_train=X_train_t,
                Y_train=Y_train_t,
                X_test=X_test_t,
                Y_test=Y_test_t,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                checkpoint_path=checkpoint_path,
                device=self.device,
                verbose=verbose,
            )

            # Freeze and store
            _freeze(model)
            self.autoencoders[level] = model

            # The trained model becomes the residual base for the next finer level
            coarse_model = model

        # ------------------------------------------------------------------
        # Now fit UpsamplingConvolution layers (coarse→fine)
        # ------------------------------------------------------------------
        # upsamplers[l] bridges from level l+1 (coarser) to level l (finer).
        # We train from coarsest gap down.
        for level in range(coarsening_level - 1, -1, -1):
            finer_level = level          # the level we want to reach
            coarser_level = level + 1   # the already-trained coarser level

            if verbose:
                print(f"\n{'='*60}")
                print(f"Training UpsamplingConvolution: level {coarser_level} → {finer_level}")
                print(f"{'='*60}")

            # Data at both levels (numpy)
            Y_train_fine = self._downsample_to_level(Y_train, finer_level)
            Y_train_coarse = self._downsample_to_level(Y_train, coarser_level)
            Y_test_fine = self._downsample_to_level(Y_test, finer_level)
            Y_test_coarse = self._downsample_to_level(Y_test, coarser_level)

            n_nodes_fine = Y_train_fine.shape[1]
            n_nodes_coarse = Y_train_coarse.shape[1]

            Y_train_fine_t = torch.from_numpy(Y_train_fine.astype(np.float32))
            Y_train_coarse_t = torch.from_numpy(Y_train_coarse.astype(np.float32))

            # D maps fine→coarse, so downsampling_list[level]: [N_coarse, N_fine]
            D = self.downsampling_list[finer_level]

            coarser_ae = self.autoencoders[coarser_level]

            upsampler = UpsamplingConvolution(
                coarse_model=coarser_ae,
                downsampling_matrix=D,
                n_nodes_fine=n_nodes_fine,
                n_nodes_coarse=n_nodes_coarse,
                n_features=self.n_features,
            )

            upsampler.fit(
                Y_coarse=Y_train_coarse_t,
                Y_fine=Y_train_fine_t,
                epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                device=self.device,
                verbose=verbose,
            )

            # Freeze and store
            _freeze(upsampler)
            self.upsamplers[finer_level] = upsampler

            # Save upsampler state
            up_path = os.path.join(
                save_dir, f"upsampler_level_{coarser_level}_to_{finer_level}.pt"
            )
            torch.save(upsampler.state_dict(), up_path)

        if verbose:
            print("\nTraining complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, level: int = 0) -> np.ndarray:
        """
        Predict displacement fields at a given coarsening level.

        Parameters
        ----------
        X     : [N, param_dim] parameter inputs (numpy)
        level : int
            0 = finest mesh (-1 also maps to finest).  Positive values give
            coarser predictions directly from the MLPAutoencoder.

        Returns
        -------
        Y_pred : [N, N_nodes_level, 3] numpy array
        """
        if level == -1:
            level = 0

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)

        if level == 0 and self.upsamplers[0] is not None:
            # Use the UpsamplingConvolution for finest level
            model = self.upsamplers[0]
            model.eval()
            with torch.no_grad():
                Y_pred = model(X_t)
        elif level > 0 and self.autoencoders[level] is not None:
            model = self.autoencoders[level]
            model.eval()
            with torch.no_grad():
                Y_pred = model(X_t)
        else:
            raise ValueError(
                f"No model available for level={level}. "
                "Make sure fit() has been called."
            )

        return Y_pred.cpu().numpy()

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
            "upsamplers": {
                l: u.state_dict()
                for l, u in enumerate(self.upsamplers)
                if u is not None
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
        up_states = checkpoint["upsamplers"]

        n_ae = max(ae_states.keys()) + 1 if ae_states else 0
        n_up = max(up_states.keys()) + 1 if up_states else 0
        total_levels = max(n_ae, n_up)

        self.autoencoders = [None] * total_levels
        self.upsamplers = [None] * total_levels

        coarse_model = None
        # Rebuild autoencoders from coarsest (highest index) to finest
        for level in sorted(ae_states.keys()):
            n_nodes = self.downsampling_list[level - 1].shape[0] if level > 0 else None
            if n_nodes is None:
                # level 0 not stored as autoencoder normally; skip
                continue
            adjacency = self.adjacency_list[level]

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
            ).to(self.device)

            model.load_state_dict(ae_states[level])
            _freeze(model)
            self.autoencoders[level] = model
            coarse_model = model

        # Rebuild upsamplers
        for finer_level, state_dict in up_states.items():
            coarser_level = finer_level + 1
            n_nodes_coarse = self.downsampling_list[finer_level].shape[0]
            n_nodes_fine = (
                self.downsampling_list[finer_level - 1].shape[0]
                if finer_level > 0
                else None
            )
            if n_nodes_fine is None:
                # Determine from downsampling shape
                n_nodes_fine = self.downsampling_list[finer_level].shape[1]

            D = self.downsampling_list[finer_level]
            coarser_ae = self.autoencoders[coarser_level]

            upsampler = UpsamplingConvolution(
                coarse_model=coarser_ae,
                downsampling_matrix=D,
                n_nodes_fine=n_nodes_fine,
                n_nodes_coarse=n_nodes_coarse,
                n_features=cfg["n_features"],
            ).to(self.device)

            upsampler.load_state_dict(state_dict)
            _freeze(upsampler)
            self.upsamplers[finer_level] = upsampler

        print(f"Loaded model from {path}")
