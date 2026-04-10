"""
Integration tests for MultiHierarchicalSurrogate.

Uses a tiny two-level hierarchy (fine 4×4, coarse 3×3) and very few epochs
so the test suite runs quickly without a GPU.
"""

import numpy as np
import pytest
import torch

from src.models.multi_hierarchical import MultiHierarchicalSurrogate
from src.utils.mesh_sampling import build_adjacency_matrix, build_sampling_matrices


# ---------------------------------------------------------------------------
# Fixtures: two-level hierarchy built from grid meshes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_level_hierarchy(grid_mesh, coarse_mesh):
    fine_verts, fine_faces = grid_mesh
    coarse_verts, coarse_faces = coarse_mesh

    A_fine = build_adjacency_matrix(fine_verts, fine_faces)
    A_coarse = build_adjacency_matrix(coarse_verts, coarse_faces)
    D, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)

    return dict(
        adjacency_list=[A_fine, A_coarse],
        downsampling_list=[D],
        upsampling_list=[U],
        n_nodes_fine=len(fine_verts),
        n_nodes_coarse=len(coarse_verts),
    )


@pytest.fixture(scope="module")
def tiny_data(two_level_hierarchy):
    """Tiny synthetic train/test arrays (8 train sims, 2 test, 3 timesteps)."""
    rng = np.random.default_rng(42)
    N_fine = two_level_hierarchy["n_nodes_fine"]
    N_train, N_test, T = 8, 2, 3
    param_dim = 4

    X_train = rng.random((N_train * T, param_dim), dtype=np.float32) * 2 - 1
    Y_train = rng.random((N_train * T, N_fine, 3), dtype=np.float32)
    X_test = rng.random((N_test * T, param_dim), dtype=np.float32) * 2 - 1
    Y_test = rng.random((N_test * T, N_fine, 3), dtype=np.float32)
    return X_train, Y_train, X_test, Y_test


@pytest.fixture(scope="module")
def tiny_cfg():
    return dict(
        reduced_order=2,
        filter_sizes=[4, 8],
        cheb_order=2,
        mlp_hidden=[8, 8],
        param_dim=4,
    )


@pytest.fixture(scope="module")
def trained_surrogate(two_level_hierarchy, tiny_data, tiny_cfg):
    """A surrogate trained for just 2 epochs — enough to test the pipeline."""
    X_train, Y_train, X_test, Y_test = tiny_data
    hier = two_level_hierarchy

    surrogate = MultiHierarchicalSurrogate(
        adjacency_list=hier["adjacency_list"],
        downsampling_list=hier["downsampling_list"],
        upsampling_list=hier["upsampling_list"],
        reduced_order=tiny_cfg["reduced_order"],
        n_features=3,
        filter_sizes=tiny_cfg["filter_sizes"],
        cheb_order=tiny_cfg["cheb_order"],
        mlp_hidden=tiny_cfg["mlp_hidden"],
        param_dim=tiny_cfg["param_dim"],
        device="cpu",
    )

    surrogate.fit(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        coarsening_level=1,
        n_epochs=2,
        batch_size=4,
        save_dir="/tmp/mh_test_checkpoints",
        verbose=False,
    )
    return surrogate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiHierarchicalSurrogate:
    def test_fit_populates_autoencoders(self, trained_surrogate):
        assert trained_surrogate.autoencoders[1] is not None

    def test_fit_populates_upsamplers(self, trained_surrogate):
        assert trained_surrogate.upsamplers[0] is not None

    def test_predict_coarse_shape(self, trained_surrogate, tiny_data, two_level_hierarchy):
        X_test = tiny_data[2]
        Y_pred = trained_surrogate.predict(X_test, level=1)
        N_coarse = two_level_hierarchy["n_nodes_coarse"]
        assert Y_pred.shape == (len(X_test), N_coarse, 3)

    def test_predict_fine_shape(self, trained_surrogate, tiny_data, two_level_hierarchy):
        X_test = tiny_data[2]
        Y_pred = trained_surrogate.predict(X_test, level=0)
        N_fine = two_level_hierarchy["n_nodes_fine"]
        assert Y_pred.shape == (len(X_test), N_fine, 3)

    def test_predict_returns_numpy(self, trained_surrogate, tiny_data):
        X_test = tiny_data[2]
        Y_pred = trained_surrogate.predict(X_test, level=1)
        assert isinstance(Y_pred, np.ndarray)

    def test_save_and_load(self, trained_surrogate, two_level_hierarchy, tiny_cfg, tmp_path):
        save_path = str(tmp_path / "surrogate.pt")
        trained_surrogate.save(save_path)

        hier = two_level_hierarchy
        loaded = MultiHierarchicalSurrogate(
            adjacency_list=hier["adjacency_list"],
            downsampling_list=hier["downsampling_list"],
            upsampling_list=hier["upsampling_list"],
            reduced_order=tiny_cfg["reduced_order"],
            n_features=3,
            filter_sizes=tiny_cfg["filter_sizes"],
            cheb_order=tiny_cfg["cheb_order"],
            mlp_hidden=tiny_cfg["mlp_hidden"],
            param_dim=tiny_cfg["param_dim"],
            device="cpu",
        )
        loaded.load(save_path)

        # Predictions should be identical after reload
        X_test = np.random.rand(3, tiny_cfg["param_dim"]).astype(np.float32)
        Y_orig = trained_surrogate.predict(X_test, level=1)
        Y_loaded = loaded.predict(X_test, level=1)
        assert np.allclose(Y_orig, Y_loaded, atol=1e-5)

    def test_invalid_level_raises(self, trained_surrogate, tiny_data):
        X_test = tiny_data[2]
        with pytest.raises((ValueError, IndexError)):
            trained_surrogate.predict(X_test, level=99)
