"""
Debug / PyCharm run script for the Multi-Hierarchical GCN Surrogate.

Edit the CONFIG block below and press Run — no shell arguments needed.
"""

import os
import pickle

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG  ←  edit these before running
# ---------------------------------------------------------------------------

DATA_PATH       = "data/crash_sim.h5"       # path to the HDF5 data file
HIERARCHY_PATH  = "data/mesh_hierarchy.pkl" # cached mesh hierarchy (created if missing)
SAVE_DIR        = "checkpoints/debug"       # where to write model checkpoints

# Mesh hierarchy
COARSENING_LEVEL    = 3           # number of coarsening levels
DECIMATION_FACTORS  = [8, 4, 4]  # triangle-count divisors per level (fine→coarse)

# Model
REDUCED_ORDER  = 4            # latent dimension
CHEB_ORDER     = 3            # Chebyshev polynomial order
FILTER_SIZES   = [6, 12, 24]  # ChebConv channel widths
MLP_HIDDEN     = [64, 64, 64] # MLP hidden layer widths

# Loss weights
LAMBDA_REC = 1.0
LAMBDA_X   = 1.0
LAMBDA_Z   = 0.0

# Training
N_EPOCHS      = 1500
BATCH_SIZE    = 32
LR            = 1e-3
TEST_FRACTION = 0.25

# ---------------------------------------------------------------------------


def _load_or_build_hierarchy(reference_vertices, reference_faces):
    from src.utils.mesh_sampling import generate_transform_matrices

    if os.path.exists(HIERARCHY_PATH):
        print(f"Loading mesh hierarchy from {HIERARCHY_PATH}")
        with open(HIERARCHY_PATH, "rb") as f:
            return pickle.load(f)

    print("Building mesh hierarchy …")
    factors = [1.0 / d for d in DECIMATION_FACTORS][:COARSENING_LEVEL]
    meshes, adjacency_list, downsampling_list, upsampling_list = generate_transform_matrices(
        vertices=reference_vertices,
        faces=reference_faces,
        decimation_factors=factors,
    )
    hierarchy = dict(
        meshes_list=meshes,
        adjacency_list=adjacency_list,
        downsampling_list=downsampling_list,
        upsampling_list=upsampling_list,
    )
    os.makedirs(os.path.dirname(os.path.abspath(HIERARCHY_PATH)), exist_ok=True)
    with open(HIERARCHY_PATH, "wb") as f:
        pickle.dump(hierarchy, f)
    print(f"Saved mesh hierarchy to {HIERARCHY_PATH}")
    return hierarchy


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading dataset from {DATA_PATH} …")
    with h5py.File(DATA_PATH, "r") as f:
        ref_config    = f["kart/reference_configuration"][:]  # [N_sim, N_nodes, 3]
        displacements = f["kart/displacements"][:]            # [N_sim, T, N_nodes, 3]
        params_raw    = f["parameter"][:]                     # [N_sim, 3]
        time_raw      = f["time"][:]                          # [T]

    N_sim, N_timesteps, N_nodes, _ = displacements.shape
    reference_vertices = ref_config[0]   # first simulation's reference mesh

    print(f"  Simulations : {N_sim}")
    print(f"  Timesteps   : {N_timesteps}")
    print(f"  Nodes (full): {N_nodes}")

    # ------------------------------------------------------------------
    # 2. Train / test split and normalisation
    # ------------------------------------------------------------------
    n_test  = max(1, int(np.round(N_sim * TEST_FRACTION)))
    n_train = N_sim - n_test
    print(f"  Train sims: {n_train}, Test sims: {n_test}")

    params_train = params_raw[:n_train]
    params_test  = params_raw[n_train:]

    p_min   = params_train.min(axis=0)
    p_max   = params_train.max(axis=0)
    denom_p = np.where(p_max - p_min > 0, p_max - p_min, 1.0)
    params_train_norm = 2.0 * (params_train - p_min) / denom_p - 1.0
    params_test_norm  = 2.0 * (params_test  - p_min) / denom_p - 1.0

    t_min, t_max = time_raw.min(), time_raw.max()
    denom_t  = (t_max - t_min) if (t_max - t_min) != 0.0 else 1.0
    time_norm = 2.0 * (time_raw - t_min) / denom_t - 1.0

    def _build_X(params_norm, n_sims):
        time_rep   = np.tile(time_norm, n_sims)
        params_rep = np.repeat(params_norm, N_timesteps, axis=0)
        return np.concatenate([time_rep[:, None], params_rep], axis=1).astype(np.float32)

    X_train = _build_X(params_train_norm, n_train)
    X_test  = _build_X(params_test_norm,  n_test)
    Y_train = displacements[:n_train].reshape(n_train * N_timesteps, N_nodes, 3).astype(np.float32)
    Y_test  = displacements[n_train:].reshape(n_test  * N_timesteps, N_nodes, 3).astype(np.float32)

    print(f"  X_train: {X_train.shape},  Y_train: {Y_train.shape}")
    print(f"  X_test : {X_test.shape},   Y_test : {Y_test.shape}")

    # ------------------------------------------------------------------
    # 3. Mesh hierarchy
    # ------------------------------------------------------------------
    try:
        with h5py.File(DATA_PATH, "r") as f:
            reference_faces = f["kart/reference_faces"][:]
        print(f"  Reference faces loaded: {reference_faces.shape}")
    except KeyError:
        print("  WARNING: 'kart/reference_faces' not found — using Delaunay fallback.")
        from scipy.spatial import Delaunay
        tri = Delaunay(reference_vertices[:, :2])
        reference_faces = tri.simplices.astype(np.int32)
        print(f"  Delaunay produced {len(reference_faces)} faces.")

    hierarchy = _load_or_build_hierarchy(reference_vertices, reference_faces)
    adjacency_list    = hierarchy["adjacency_list"]
    downsampling_list = hierarchy["downsampling_list"]
    upsampling_list   = hierarchy["upsampling_list"]

    for i, (v, _) in enumerate(hierarchy["meshes_list"]):
        print(f"  Level {i}: {len(v)} nodes")

    # ------------------------------------------------------------------
    # 4. Build and train
    # ------------------------------------------------------------------
    from src.models.multi_hierarchical import MultiHierarchicalSurrogate

    surrogate = MultiHierarchicalSurrogate(
        adjacency_list=adjacency_list,
        downsampling_list=downsampling_list,
        upsampling_list=upsampling_list,
        reduced_order=REDUCED_ORDER,
        n_features=3,
        filter_sizes=FILTER_SIZES,
        cheb_order=CHEB_ORDER,
        mlp_hidden=MLP_HIDDEN,
        lambda_rec=LAMBDA_REC,
        lambda_x=LAMBDA_X,
        lambda_z=LAMBDA_Z,
        param_dim=X_train.shape[1],
    )

    surrogate.fit(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        coarsening_level=COARSENING_LEVEL,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        save_dir=SAVE_DIR,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 5. Save and quick sanity check
    # ------------------------------------------------------------------
    final_path = os.path.join(SAVE_DIR, "surrogate_final.pt")
    surrogate.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    # Quick prediction check
    Y_pred = surrogate.predict(X_test[:5], level=0)
    print(f"Prediction shape (fine, 5 samples): {Y_pred.shape}")
    print(f"Prediction range: [{Y_pred.min():.4f}, {Y_pred.max():.4f}]")

    mse = np.mean((Y_pred - Y_test[:5]) ** 2)
    print(f"MSE on first 5 test samples: {mse:.6f}")


if __name__ == "__main__":
    main()
