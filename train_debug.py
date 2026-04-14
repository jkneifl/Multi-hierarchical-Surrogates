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

DATA_PATH       = "/Users/jonaskneifl/Develop/00_MOR_ML_Models/01_models/01_Kart/05_Clean/06_data/kart_simulations_dissertation.pkl"       # path to the HDF5 data file
HIERARCHY_PATH  = "meshing/madu.pkl" # cached mesh hierarchy (created if missing)
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
TRAIN = True
N_EPOCHS      = 100
BATCH_SIZE    = 32
LR            = 1e-3
TEST_FRACTION = 0.25
VAL_FRACTION = 0.25

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
    with open(DATA_PATH, "rb") as sim_data:
        data = pickle.load(sim_data)
        # Loading resources
        displacements = data["kart"]["displacements"]
        params_raw = data["params"]
        time_raw = data["time"][0]
        ref_config = data["kart"]["ref_coords"]
        reference_faces = data["kart"]["faces"]

    N_sim, N_timesteps, N_nodes, _ = displacements.shape
    reference_vertices = ref_config[0]   # first simulation's reference mesh

    print(f"  Simulations : {N_sim}")
    print(f"  Timesteps   : {N_timesteps}")
    print(f"  Nodes (full): {N_nodes}")

    # ------------------------------------------------------------------
    # 2. Train / test split and normalisation
    # ------------------------------------------------------------------
    n_test  = max(1, int(np.round(N_sim * TEST_FRACTION)))
    n_val  = max(1, int(np.round(N_sim * VAL_FRACTION)))
    n_train = N_sim - n_test - n_val
    print(f"  Train sims: {n_train}, Test sims: {n_test}")

    params_train = params_raw[:n_train]
    params_val = params_raw[n_train:n_train+n_val]
    params_test  = params_raw[n_train+n_val:]

    p_min   = params_train.min(axis=0)
    p_max   = params_train.max(axis=0)
    denom_p = np.where(p_max - p_min > 0, p_max - p_min, 1.0)
    params_train_norm = 2.0 * (params_train - p_min) / denom_p - 1.0
    params_val_norm  = 2.0 * (params_val    - p_min) / denom_p - 1.0
    params_test_norm  = 2.0 * (params_test  - p_min) / denom_p - 1.0

    t_min, t_max = time_raw.min(), time_raw.max()
    denom_t  = (t_max - t_min) if (t_max - t_min) != 0.0 else 1.0
    time_norm = 2.0 * (time_raw - t_min) / denom_t - 1.0

    def _build_mu(params_norm, n_sims):
        time_rep   = np.tile(time_norm, n_sims)
        params_rep = np.repeat(params_norm, N_timesteps, axis=0)
        return np.concatenate([time_rep[:, None], params_rep], axis=1).astype(np.float32)

    mu_train = _build_mu(params_train_norm, n_train)
    mu_val  = _build_mu(params_val_norm,  n_val)
    mu_test  = _build_mu(params_test_norm,  n_test)
    x_train  = displacements[:n_train].reshape(n_train * N_timesteps, N_nodes, 3).astype(np.float32)
    x_val   = displacements[n_train:n_train+n_val].reshape(n_val  * N_timesteps, N_nodes, 3).astype(np.float32)
    x_test   = displacements[n_train+n_val:].reshape(n_test  * N_timesteps, N_nodes, 3).astype(np.float32)

    print(f"  mu_train: {mu_train.shape},  x_train: {x_train.shape}")
    print(f"  mu_val : {mu_val.shape},   x_val : {x_val.shape}")
    print(f"  mu_test : {mu_test.shape},   x_test : {x_test.shape}")

    # ------------------------------------------------------------------
    # 3. Mesh hierarchy
    # ------------------------------------------------------------------
    hierarchy = _load_or_build_hierarchy(reference_vertices, reference_faces)
    adjacency_list    = hierarchy["adjacency_list"]
    downsampling_list = hierarchy["downsampling_list"]
    upsampling_list   = hierarchy["upsampling_list"]

    for i, (v, _) in enumerate(hierarchy["meshes_list"]):
        print(f"  Level {i}: {len(v)} nodes")

    # # visualize coarsened meshes
    # from visualizer import Visualizer
    # vis = Visualizer(background_color='white')
    # for i, (v, f) in enumerate(hierarchy["meshes_list"]):
    #     vis.animate(
    #         v[np.newaxis],
    #         faces=f,
    #         color=[0.5, 0.5, 0.5],
    #         shift=False,
    #         point_size=4,
    #     )

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
        param_dim=mu_train.shape[1],
    )

    if TRAIN:
        surrogate.fit(
            mu_train=mu_train,
            x_train=x_train,
            mu_val=mu_val,
            x_val=x_val,
            coarsening_level=COARSENING_LEVEL,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            save_dir=SAVE_DIR,
            verbose=True,
            cache_weights=True,
        )
        final_path = os.path.join(SAVE_DIR, "surrogate_final.pt")
        surrogate.save(final_path)
        print(f"\nFinal model saved to {final_path}")
    else:
        print("\nTRAINING DISABLED — skipping to final save and sanity check.")
        surrogate.load(os.path.join(SAVE_DIR, "surrogate_final.pt"))

    # ------------------------------------------------------------------
    # 5. Quick sanity check
    # ------------------------------------------------------------------

    # Quick prediction check
    x_pred = surrogate.predict(mu_test, level=0)
    print(f"Prediction shape (fine, 5 samples): {x_pred.shape}")
    print(f"Prediction range: [{x_pred.min():.4f}, {x_pred.max():.4f}]")

    mse = np.mean((x_pred - x_test) ** 2)
    print(f"MSE on first 5 test samples: {mse:.6f}")

    from visualizer import Visualizer
    vis = Visualizer(background_color='white')
    vis.animate(
        [x_pred+reference_vertices, x_test+reference_vertices],
        faces=[reference_faces, reference_faces],
        color=["blue", "red"],
        shift=False,
        point_size=4,
    )



if __name__ == "__main__":
    main()
