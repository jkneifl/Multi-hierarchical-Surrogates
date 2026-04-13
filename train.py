"""
Training script for the Multi-Hierarchical GCN Surrogate.

Usage
-----
    python train.py \\
        --data_path data/crash_sim.h5 \\
        --save_dir  checkpoints/run_01 \\
        --coarsening_level 3 \\
        --n_epochs 1500

The script will:
  1. Load (or generate) the mesh hierarchy (adjacency, downsampling, upsampling).
  2. Load the HDF5 dataset and split into train/test.
  3. Train MultiHierarchicalSurrogate.
  4. Save the final model.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the multi-hierarchical GCN surrogate model."
    )

    # Data / paths
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the HDF5 data file.",
    )
    parser.add_argument(
        "--madu_path",
        type=str,
        default=None,
        help=(
            "Path to a pre-computed mesh hierarchy pickle file. "
            "If the file does not exist it will be generated and saved here."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory where model checkpoints are saved.",
    )

    # Mesh hierarchy
    parser.add_argument(
        "--coarsening_level",
        type=int,
        default=3,
        help="Number of coarsening levels to use (default: 3).",
    )
    parser.add_argument(
        "--decimation_factors",
        type=float,
        nargs="+",
        default=[8, 4, 4],
        help=(
            "Triangle-count divisors per level (finest→coarsest). "
            "factor=8 means target = len(faces)/8 triangles. "
            "Default: [8, 4, 4]."
        ),
    )

    # Model hyper-parameters
    parser.add_argument(
        "--reduced_order",
        type=int,
        default=4,
        help="Latent dimension (default: 4).",
    )
    parser.add_argument(
        "--cheb_order",
        type=int,
        default=3,
        help="Chebyshev polynomial order (default: 3).",
    )
    parser.add_argument(
        "--filter_sizes",
        type=int,
        nargs="+",
        default=[6, 12, 24],
        help="ChebConv channel widths (default: 6 12 24).",
    )
    parser.add_argument(
        "--mlp_hidden",
        type=int,
        nargs="+",
        default=[64, 64, 64],
        help="MLP hidden layer sizes (default: 64 64 64).",
    )

    # Loss weights
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_x",   type=float, default=1.0)
    parser.add_argument("--lambda_z",   type=float, default=0.0)

    # Training
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1500,
        help="Training epochs per hierarchy level (default: 1500).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.25,
        help="Fraction of simulations used for testing (default: 0.25).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mesh hierarchy helpers
# ---------------------------------------------------------------------------

def _load_or_build_hierarchy(
    args: argparse.Namespace,
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
) -> dict:
    """
    Load the mesh hierarchy from a pickle file, or generate and save it.

    Returns a dict with keys:
        meshes_list, adjacency_list, downsampling_list, upsampling_list
    """
    from src.utils.mesh_sampling import generate_transform_matrices

    if args.madu_path is not None and os.path.exists(args.madu_path):
        print(f"Loading pre-computed mesh hierarchy from {args.madu_path}")
        with open(args.madu_path, "rb") as f:
            hierarchy = pickle.load(f)
        return hierarchy

    print("Generating mesh hierarchy …")
    # decimation_factors are given as divisors; convert to fractions for
    # generate_transform_matrices which expects fractions in (0, 1).
    factors_as_fractions = [1.0 / d for d in args.decimation_factors]
    # Limit to coarsening_level steps
    factors_as_fractions = factors_as_fractions[: args.coarsening_level]

    meshes_list, adjacency_list, downsampling_list, upsampling_list = (
        generate_transform_matrices(
            vertices=reference_vertices,
            faces=reference_faces,
            decimation_factors=factors_as_fractions,
        )
    )

    hierarchy = {
        "meshes_list": meshes_list,
        "adjacency_list": adjacency_list,
        "downsampling_list": downsampling_list,
        "upsampling_list": upsampling_list,
    }

    if args.madu_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.madu_path)), exist_ok=True)
        with open(args.madu_path, "wb") as f:
            pickle.dump(hierarchy, f)
        print(f"Saved mesh hierarchy to {args.madu_path}")

    return hierarchy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading dataset from {args.data_path} …")

    import h5py

    with h5py.File(args.data_path, "r") as f:
        # Reference configuration — use the first simulation's reference mesh
        ref_config = f["kart/reference_configuration"][:]   # [N_sim, N_nodes, 3]
        displacements = f["kart/displacements"][:]          # [N_sim, T, N_nodes, 3]
        params_raw = f["parameter"][:]                      # [N_sim, 3]
        time_raw = f["time"][:]                             # [T]

    N_sim, N_timesteps, N_nodes, _ = displacements.shape
    reference_vertices = ref_config[0]  # [N_nodes, 3] — first simulation's reference

    print(f"  Simulations:  {N_sim}")
    print(f"  Timesteps:    {N_timesteps}")
    print(f"  Nodes (full): {N_nodes}")

    # ------------------------------------------------------------------
    # 2. Split into train / test
    # ------------------------------------------------------------------
    n_test = max(1, int(np.round(N_sim * args.test_fraction)))
    n_train = N_sim - n_test
    print(f"  Train sims: {n_train}, Test sims: {n_test}")

    # Normalise parameters
    params_train = params_raw[:n_train]
    params_test = params_raw[n_train:]

    p_min = params_train.min(axis=0)
    p_max = params_train.max(axis=0)
    denom_p = np.where(p_max - p_min > 0, p_max - p_min, 1.0)
    params_train_norm = 2.0 * (params_train - p_min) / denom_p - 1.0
    params_test_norm = 2.0 * (params_test - p_min) / denom_p - 1.0

    t_min, t_max = time_raw.min(), time_raw.max()
    denom_t = (t_max - t_min) if (t_max - t_min) != 0.0 else 1.0
    time_norm = 2.0 * (time_raw - t_min) / denom_t - 1.0

    def _build_mu(params_norm: np.ndarray, n_sims: int) -> np.ndarray:
        """Build [n_sims*T, 4] parameter array from normalised params and time."""
        time_rep = np.tile(time_norm, n_sims)                          # [n_sims*T]
        params_rep = np.repeat(params_norm, N_timesteps, axis=0)       # [n_sims*T, 3]
        return np.concatenate([time_rep[:, None], params_rep], axis=1).astype(np.float32)

    mu_train = _build_mu(params_train_norm, n_train)   # [n_train*T, 4]
    mu_val  = _build_mu(params_test_norm, n_test)     # [n_test*T, 4]

    x_train = displacements[:n_train].reshape(n_train * N_timesteps, N_nodes, 3).astype(np.float32)
    x_val  = displacements[n_train:].reshape(n_test * N_timesteps, N_nodes, 3).astype(np.float32)

    print(f"  mu_train: {mu_train.shape}, x_train: {x_train.shape}")
    print(f"  mu_val:  {mu_val.shape},  x_val:  {x_val.shape}")

    # ------------------------------------------------------------------
    # 3. Build / load mesh hierarchy
    # ------------------------------------------------------------------
    # We need faces for mesh simplification; try to load from HDF5 or fall
    # back to a Delaunay triangulation of the reference vertices.
    try:
        with h5py.File(args.data_path, "r") as f:
            reference_faces = f["kart/reference_faces"][:]  # [F, 3]
        print(f"  Reference faces loaded from HDF5: {reference_faces.shape}")
    except KeyError:
        print(
            "  WARNING: 'kart/reference_faces' not found in HDF5. "
            "Falling back to scipy Delaunay triangulation of reference vertices."
        )
        from scipy.spatial import Delaunay
        tri = Delaunay(reference_vertices[:, :2])   # 2-D projection
        reference_faces = tri.simplices.astype(np.int32)
        print(f"  Delaunay triangulation produced {len(reference_faces)} faces.")

    hierarchy = _load_or_build_hierarchy(args, reference_vertices, reference_faces)
    adjacency_list = hierarchy["adjacency_list"]
    downsampling_list = hierarchy["downsampling_list"]
    upsampling_list = hierarchy["upsampling_list"]

    # Print node counts per level
    for i, (v, _) in enumerate(hierarchy["meshes_list"]):
        print(f"  Level {i}: {len(v)} nodes")

    # ------------------------------------------------------------------
    # 4. Build and train the surrogate
    # ------------------------------------------------------------------
    from src.models.multi_hierarchical import MultiHierarchicalSurrogate

    surrogate = MultiHierarchicalSurrogate(
        adjacency_list=adjacency_list,
        downsampling_list=downsampling_list,
        upsampling_list=upsampling_list,
        reduced_order=args.reduced_order,
        n_features=3,
        filter_sizes=args.filter_sizes,
        cheb_order=args.cheb_order,
        mlp_hidden=args.mlp_hidden,
        lambda_rec=args.lambda_rec,
        lambda_x=args.lambda_x,
        lambda_z=args.lambda_z,
        param_dim=mu_train.shape[1],   # 4 = time + 3 params
    )

    surrogate.fit(
        mu_train=mu_train,
        x_train=x_train,
        mu_val=mu_val,
        x_val=x_val,
        coarsening_level=args.coarsening_level,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 5. Save final model
    # ------------------------------------------------------------------
    final_path = os.path.join(args.save_dir, "surrogate_final.pt")
    surrogate.save(final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == "__main__":
    main()
