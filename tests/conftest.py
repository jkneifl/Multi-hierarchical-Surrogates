"""
Shared fixtures for the test suite.

All fixtures use tiny synthetic meshes so no real data or Open3D is needed.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Tiny grid mesh (4x4 = 16 nodes, triangulated)
# ---------------------------------------------------------------------------

def _make_grid_mesh(nx: int = 4, ny: int = 4):
    """
    Create a flat (z=0) regular grid mesh with (nx*ny) nodes and
    2*(nx-1)*(ny-1) triangles.
    """
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)], axis=1).astype(np.float32)

    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v0 = j * nx + i
            v1 = v0 + 1
            v2 = v0 + nx
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    faces = np.array(faces, dtype=np.int32)
    return vertices, faces


@pytest.fixture(scope="session")
def grid_mesh():
    """A 4×4 grid mesh: 16 nodes, 18 triangles."""
    return _make_grid_mesh(4, 4)


@pytest.fixture(scope="session")
def adjacency(grid_mesh):
    """Binary adjacency matrix for the 4×4 grid mesh."""
    from src.utils.mesh_sampling import build_adjacency_matrix
    verts, faces = grid_mesh
    return build_adjacency_matrix(verts, faces)


@pytest.fixture(scope="session")
def coarse_mesh(grid_mesh):
    """A coarser 3×3 grid mesh: 9 nodes."""
    return _make_grid_mesh(3, 3)


@pytest.fixture(scope="session")
def coarse_adjacency(coarse_mesh):
    from src.utils.mesh_sampling import build_adjacency_matrix
    verts, faces = coarse_mesh
    return build_adjacency_matrix(verts, faces)


# ---------------------------------------------------------------------------
# Model hyper-parameters kept small for speed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_cfg():
    return dict(
        n_features=3,
        latent_dim=4,
        filter_sizes=[4, 8],
        cheb_order=2,
        mlp_hidden=[16, 16],
        param_dim=4,
    )
