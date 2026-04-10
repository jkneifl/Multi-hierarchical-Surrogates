"""
CrashSimDataset — HDF5-backed PyTorch Dataset for crash simulation data.

Expected HDF5 layout
--------------------
    kart/reference_configuration : [N_sim, N_nodes, 3]
    kart/displacements            : [N_sim, N_timesteps, N_nodes, 3]
    parameter                     : [N_sim, 3]   (velocity, angle, yield_stress)
    time                          : [N_timesteps]

Each sample returned by __getitem__ is a (x, y) pair:
    x : [4] float32 tensor  — (time, param_0, param_1, param_2), normalised to [-1, 1]
    y : [N_nodes, 3] float32 tensor  — displacement field at that timestep / simulation
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


class CrashSimDataset(Dataset):
    """
    PyTorch Dataset wrapping an HDF5 crash-simulation data file.

    The flat index ``idx`` maps to simulation ``sim_idx`` and timestep
    ``t_idx`` via:
        sim_idx = idx // N_timesteps
        t_idx   = idx  % N_timesteps

    Parameters
    ----------
    data_path : str
        Path to the HDF5 file.
    split : {'train', 'test'}
        Which split to materialise.
    test_fraction : float
        Fraction of simulations reserved for testing (default 0.25).
    normalize_params : bool
        If True, normalise parameters and time to [-1, 1] using the training
        set statistics only (avoids data leakage).
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        test_fraction: float = 0.25,
        normalize_params: bool = True,
    ):
        super().__init__()
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.data_path = data_path
        self.split = split
        self.test_fraction = test_fraction
        self.normalize_params = normalize_params

        self._load(data_path, split, test_fraction, normalize_params)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load(
        self,
        data_path: str,
        split: str,
        test_fraction: float,
        normalize_params: bool,
    ) -> None:
        import h5py

        with h5py.File(data_path, "r") as f:
            # Displacements: [N_sim, N_timesteps, N_nodes, 3]
            displacements = f["kart/displacements"][:]  # load fully into RAM
            # Parameters: [N_sim, 3]
            params = f["parameter"][:]
            # Time: [N_timesteps]
            time = f["time"][:]

        N_sim, N_timesteps, N_nodes, n_feat = displacements.shape

        # --- train / test split on simulation axis ---
        n_test = max(1, int(np.round(N_sim * test_fraction)))
        n_train = N_sim - n_test
        # Deterministic split: last n_test simulations go to test set
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, N_sim)

        sel = train_idx if split == "train" else test_idx

        displacements = displacements[sel]  # [N_sel, T, N_nodes, 3]
        params = params[sel]                # [N_sel, 3]

        # --- normalise to [-1, 1] using training statistics ---
        if normalize_params:
            train_params = params if split == "train" else None
            # We always need training stats; re-read if we're in test split
            if split == "test":
                with h5py.File(data_path, "r") as f:
                    train_params_all = f["parameter"][:n_train]
                p_min = train_params_all.min(axis=0)
                p_max = train_params_all.max(axis=0)

                time_all = f["time"][:] if False else time  # already loaded
                t_min = time.min()
                t_max = time.max()
            else:
                p_min = params.min(axis=0)
                p_max = params.max(axis=0)
                t_min = time.min()
                t_max = time.max()

            # Normalise params: [N_sel, 3]
            denom_p = p_max - p_min
            denom_p[denom_p == 0.0] = 1.0
            params_norm = 2.0 * (params - p_min) / denom_p - 1.0

            # Normalise time: [T]
            denom_t = t_max - t_min if (t_max - t_min) != 0.0 else 1.0
            time_norm = 2.0 * (time - t_min) / denom_t - 1.0
        else:
            params_norm = params
            time_norm = time

        # --- flatten to (N_sel * T) samples ---
        N_sel = len(sel)
        # displacements: [N_sel, T, N_nodes, 3] → [N_sel*T, N_nodes, 3]
        Y = displacements.reshape(N_sel * N_timesteps, N_nodes, n_feat).astype(np.float32)

        # Build X: [N_sel * T, 4] = (time_t, p0, p1, p2)
        # time_norm: [T], params_norm: [N_sel, 3]
        # Broadcast: for each sim repeat time, for each timestep repeat params
        time_rep = np.tile(time_norm, N_sel)                   # [N_sel*T]
        params_rep = np.repeat(params_norm, N_timesteps, axis=0)  # [N_sel*T, 3]
        X = np.concatenate(
            [time_rep[:, None], params_rep], axis=1
        ).astype(np.float32)  # [N_sel*T, 4]

        self.X = torch.from_numpy(X)  # [N_sel*T, 4]
        self.Y = torch.from_numpy(Y)  # [N_sel*T, N_nodes, 3]
        self.N_nodes = N_nodes
        self.N_timesteps = N_timesteps
        self.n_samples = N_sel * N_timesteps

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x : [4] float32  — (time, param_0, param_1, param_2)
        y : [N_nodes, 3] float32 — displacements
        """
        return self.X[idx], self.Y[idx]
