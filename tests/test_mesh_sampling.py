"""Tests for src/utils/mesh_sampling.py"""

import numpy as np
import pytest
import scipy.sparse as sp

from src.utils.mesh_sampling import build_adjacency_matrix, build_sampling_matrices


class TestBuildAdjacencyMatrix:
    def test_shape(self, grid_mesh):
        verts, faces = grid_mesh
        A = build_adjacency_matrix(verts, faces)
        N = len(verts)
        assert A.shape == (N, N)

    def test_symmetric(self, grid_mesh):
        verts, faces = grid_mesh
        A = build_adjacency_matrix(verts, faces)
        diff = (A - A.T)
        assert diff.nnz == 0 or np.allclose(diff.data, 0)

    def test_no_self_loops(self, grid_mesh):
        verts, faces = grid_mesh
        A = build_adjacency_matrix(verts, faces)
        assert np.allclose(A.diagonal(), 0)

    def test_binary_values(self, grid_mesh):
        verts, faces = grid_mesh
        A = build_adjacency_matrix(verts, faces)
        assert set(A.data.tolist()).issubset({0.0, 1.0})

    def test_connectivity(self, grid_mesh):
        """Every node should have at least one neighbour."""
        verts, faces = grid_mesh
        A = build_adjacency_matrix(verts, faces)
        degrees = np.asarray(A.sum(axis=1)).ravel()
        assert (degrees > 0).all()


class TestBuildSamplingMatrices:
    def test_shapes(self, grid_mesh, coarse_mesh):
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        D, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        N_fine = len(fine_verts)
        N_coarse = len(coarse_verts)
        assert D.shape == (N_coarse, N_fine)
        assert U.shape == (N_fine, N_coarse)

    def test_U_is_D_transpose(self, grid_mesh, coarse_mesh):
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        D, U = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        diff = (U - D.T)
        assert np.allclose(diff.data, 0, atol=1e-6)

    def test_D_row_normalised(self, grid_mesh, coarse_mesh):
        """Each row of D should sum to 1 (weighted average)."""
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        D, _ = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        row_sums = np.asarray(D.sum(axis=1)).ravel()
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_downsampling_reduces_nodes(self, grid_mesh, coarse_mesh):
        fine_verts, fine_faces = grid_mesh
        coarse_verts, _ = coarse_mesh
        D, _ = build_sampling_matrices(fine_verts, coarse_verts, fine_faces)
        assert D.shape[0] < D.shape[1]
