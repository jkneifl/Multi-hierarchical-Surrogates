"""
Synthetic plate-bending example for the Multi-Hierarchical GCN Surrogate.

No external data required.  A parametric sinusoidal bending field is computed
analytically on a flat square plate mesh, giving a clean surrogate-learning
problem where the ground truth is exactly known.

Mesh:        25 × 25 regular grid  →  625 nodes, ~1 152 triangles
Parameters:  amplitude A ∈ [0.5, 2.0]  and  wavenumber k ∈ [1, 3]
             + normalised time  t ∈ [0, 1]  →  param_dim = 3
Hierarchy:   2 coarsening levels (1/4 triangles each)
Displacement:
    u_x =  0.05 · A · t · cos(k·π·x) · sin(π·y)
    u_y =  0.05 · A · t · sin(k·π·x) · cos(π·y)
    u_z =        A · t · sin(k·π·x)  · sin(π·y)

Run:
    python examples/plate_bending.py
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
# matplotlib.use("Agg")          # no display needed; swap to "TkAgg" / "Qt5Agg" to show
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# Make sure the project root is on the path when running from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.mesh_sampling import generate_transform_matrices
from src.models.multi_hierarchical import MultiHierarchicalSurrogate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NX, NY           = 25, 25          # grid resolution → 625 nodes
N_SIMS           = 200             # number of (A, k) parameter combinations
N_TIMESTEPS      = 10              # time snapshots per simulation
COARSENING_LEVEL = 2               # levels of mesh coarsening
DECIMATION_FACTORS = [0.25, 0.25]  # triangle fraction kept at each level

REDUCED_ORDER = 8
CHEB_ORDER    = 3
FILTER_SIZES  = [8, 16, 32]
MLP_HIDDEN    = [64, 64, 64]
N_EPOCHS      = 500
BATCH_SIZE    = 64
LR            = 1e-3

TRAIN_FRACTION = 0.6
VAL_FRACTION   = 0.2
# remaining 0.2 → test

TRAIN = False
SAVE_DIR       = "checkpoints/plate_bending"
HIERARCHY_PATH = "checkpoints/plate_bending_hierarchy.pkl"

# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def make_plate_mesh(nx: int, ny: int):
    """Regular nx × ny grid in [0,1]² triangulated by splitting each quad."""
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    verts = np.column_stack([xx.ravel(), yy.ravel(),
                              np.zeros(nx * ny, dtype=np.float32)])

    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + i + 1
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + i + 1
            faces.append([v11, v00, v10])
            faces.append([v01, v00, v11])
    return verts, np.array(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# Analytical displacement field
# ---------------------------------------------------------------------------

def plate_displacement(verts: np.ndarray, A: float, k: float, t: float) -> np.ndarray:
    """
    Sinusoidal bending displacement  [N, 3]  for a unit-square plate.

    Args:
        verts : [N, 3]  rest-configuration vertex positions
        A     : amplitude
        k     : wavenumber (integer multiples of π along x)
        t     : time ∈ [0, 1]
    """
    x, y = verts[:, 0], verts[:, 1]
    u_x = 0.05 * A * t * np.cos(k * np.pi * x) * np.sin(np.pi * y)
    u_y = 0.05 * A * t * np.sin(k * np.pi * x) * np.cos(np.pi * y)
    u_z =        A * t * np.sin(k * np.pi * x)  * np.sin(np.pi * y)
    return np.stack([u_x, u_y, u_z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(verts: np.ndarray):
    """
    Sample N_SIMS · N_TIMESTEPS snapshots.

    Returns
    -------
    mu   : [N_total, 3]   (t_norm, A_norm, k_norm)
    x    : [N_total, N, 3]
    """
    rng = np.random.default_rng(42)
    A_vals = rng.uniform(0.5, 2.0, N_SIMS).astype(np.float32)
    k_vals = rng.uniform(1.0, 3.0, N_SIMS).astype(np.float32)
    t_vals = np.linspace(0.0, 1.0, N_TIMESTEPS, dtype=np.float32)

    all_mu, all_x = [], []
    for A, k in zip(A_vals, k_vals):
        for t in t_vals:
            all_mu.append([t, A, k])
            all_x.append(plate_displacement(verts, A, k, t))

    mu = np.array(all_mu, dtype=np.float32)   # [N_total, 3]
    x  = np.array(all_x,  dtype=np.float32)   # [N_total, N, 3]
    return mu, x


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_level(
    level: int,
    verts: np.ndarray,
    faces: np.ndarray,
    x_pred: np.ndarray,
    x_gt: np.ndarray,
    sample_idx: int = 0,
    out_path: str | None = None,
):
    """
    Side-by-side trisurf of prediction vs ground truth for one sample.
    Colour encodes the z-displacement (u_z).
    """
    tri = Triangulation(verts[:, 0], verts[:, 1], faces)
    uz_pred = x_pred[sample_idx, :, 2]
    uz_gt   = x_gt[sample_idx, :, 2]
    vmin = min(uz_gt.min(), uz_pred.min())
    vmax = max(uz_gt.max(), uz_pred.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, uz, title in zip(
        axes,
        [uz_pred, uz_gt],
        [f"Prediction  (level {level})", f"Ground truth  (level {level})"],
    ):
        tcf = ax.tripcolor(tri, uz, shading="flat", cmap="RdBu_r",
                           vmin=vmin, vmax=vmax)
        ax.triplot(tri, color="k", linewidth=0.2, alpha=0.4)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x");  ax.set_ylabel("y")
        fig.colorbar(tcf, ax=ax, label="u_z")

    mse = np.mean((uz_pred - uz_gt) ** 2)
    fig.suptitle(f"Level {level} — {len(verts)} nodes   MSE(u_z) = {mse:.2e}")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved figure → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Mesh & hierarchy
    # ------------------------------------------------------------------
    print("Building plate mesh …")
    verts, faces = make_plate_mesh(NX, NY)
    print(f"  Fine mesh: {len(verts)} nodes, {len(faces)} triangles")

    if os.path.exists(HIERARCHY_PATH):
        print(f"Loading mesh hierarchy from {HIERARCHY_PATH}")
        with open(HIERARCHY_PATH, "rb") as f:
            hierarchy = pickle.load(f)
    else:
        print("Building mesh hierarchy …")
        meshes_list, adjacency_list, downsampling_list, upsampling_list = \
            generate_transform_matrices(verts, faces, DECIMATION_FACTORS)
        hierarchy = dict(
            meshes_list=meshes_list,
            adjacency_list=adjacency_list,
            downsampling_list=downsampling_list,
            upsampling_list=upsampling_list,
        )
        with open(HIERARCHY_PATH, "wb") as f:
            pickle.dump(hierarchy, f)
        print(f"Saved hierarchy to {HIERARCHY_PATH}")

    for i, (v, _) in enumerate(hierarchy["meshes_list"]):
        print(f"  Level {i}: {len(v)} nodes")

    # ------------------------------------------------------------------
    # 2. Dataset
    # ------------------------------------------------------------------
    print("\nGenerating synthetic dataset …")
    mu, x = generate_dataset(verts)
    N_total = len(mu)
    print(f"  Total snapshots: {N_total}  mu: {mu.shape}  x: {x.shape}")

    n_train = int(N_total * TRAIN_FRACTION)
    n_val   = int(N_total * VAL_FRACTION)

    mu_train, x_train = mu[:n_train],           x[:n_train]
    mu_val,   x_val   = mu[n_train:n_train+n_val], x[n_train:n_train+n_val]
    mu_test,  x_test  = mu[n_train+n_val:],     x[n_train+n_val:]
    print(f"  Train: {len(mu_train)}  Val: {len(mu_val)}  Test: {len(mu_test)}")

    # ------------------------------------------------------------------
    # 3. Normalise μ
    # ------------------------------------------------------------------
    mu_min  = mu_train.min(axis=0)
    mu_max  = mu_train.max(axis=0)
    denom   = np.where(mu_max - mu_min > 0, mu_max - mu_min, 1.0)
    norm    = lambda m: (2.0 * (m - mu_min) / denom - 1.0).astype(np.float32)

    mu_train = norm(mu_train)
    mu_val   = norm(mu_val)
    mu_test  = norm(mu_test)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    surrogate = MultiHierarchicalSurrogate(
        adjacency_list=hierarchy["adjacency_list"],
        downsampling_list=hierarchy["downsampling_list"],
        upsampling_list=hierarchy["upsampling_list"],
        reduced_order=REDUCED_ORDER,
        n_features=3,
        filter_sizes=FILTER_SIZES,
        cheb_order=CHEB_ORDER,
        mlp_hidden=MLP_HIDDEN,
        param_dim=mu_train.shape[1],
        lambda_rec=1.0,
        lambda_x=1.0,
        lambda_z=0.0,
        lambda_up=1.0,
    )

    if TRAIN:
        surrogate.fit(
            mu_train=mu_train, x_train=x_train,
            mu_val=mu_val,     x_val=x_val,
            coarsening_level=COARSENING_LEVEL,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            save_dir=SAVE_DIR,
            verbose=True,
            cache_weights=True,
        )

        surrogate.save(os.path.join(SAVE_DIR, "surrogate_final.pt"))
    else:
        surrogate.load(os.path.join(SAVE_DIR, "surrogate_final.pt"))
        print(f"Loaded surrogate from {os.path.join(SAVE_DIR, 'surrogate_final.pt')}")
    # ------------------------------------------------------------------
    # 5. Evaluate and visualise all levels
    # ------------------------------------------------------------------
    meshes_list = hierarchy["meshes_list"]
    print()

    # sample_idx points to the middle timestep of the first test simulation so
    # t > 0 and the displacement field is non-zero for both subplots.
    sample_idx = N_TIMESTEPS // 2

    for level in range(COARSENING_LEVEL + 1):
        if level == 0:
            lv, lf = verts, faces
            x_gt = x_test
        else:
            lv, lf = meshes_list[level]
            x_gt = surrogate._downsample_to_level(x_test, level)

        x_pred = surrogate.predict(mu_test, level=level)
        mse  = np.mean((x_pred - x_gt) ** 2)
        rmse = np.sqrt(mse)
        print(
            f"Level {level}  nodes={lv.shape[0]:>5d}  "
            f"MSE={mse:.4e}  RMSE={rmse:.4e}"
        )

        plot_level(
            level=level,
            verts=lv,
            faces=lf,
            x_pred=x_pred,
            x_gt=x_gt,
            sample_idx=sample_idx,
            out_path=os.path.join(SAVE_DIR, f"prediction_level_{level}.png"),
        )

    # Visualizer: one window per hierarchy level, animating the first test
    # simulation (N_TIMESTEPS frames, t=0 → t=1).
    from visualizer import Visualizer
    for i, (v, f) in enumerate(hierarchy["meshes_list"]):
        x_pred = surrogate.predict(mu_test, level=i)
        x_gt   = surrogate._downsample_to_level(x_test, level=i)
        vis = Visualizer(background_color='white', shader='normalColor')
        vis.animate(
            [v + x_pred, v + x_gt],
            faces=[f, f],
            color=["blue", "red"],
            shift=True,
            point_size=4,
        )


if __name__ == "__main__":
    main()
