"""
Mesh sampling utilities using Open3D.

Provides functions to generate a hierarchy of coarser meshes via mesh
simplification, and to build the corresponding adjacency, downsampling, and
upsampling matrices needed by the multi-hierarchical surrogate.

Replaces the psbody.mesh-based approach with Open3D mesh simplification.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not available. Mesh simplification will not work.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_transform_matrices(
    vertices: np.ndarray,
    faces: np.ndarray,
    decimation_factors: List[float],
) -> Tuple[list, List[sp.spmatrix], List[sp.spmatrix], List[sp.spmatrix]]:
    """
    Generate a hierarchy of coarser meshes via iterative mesh simplification
    and build the corresponding graph matrices.

    Starting from the *finest* mesh (vertices, faces), each entry in
    ``decimation_factors`` specifies the fraction of triangles to *keep* when
    going one level coarser.  For example, [0.25, 0.25] produces two levels
    each with 25% of the triangles of the previous level.

    Args:
        vertices: float array of shape [N_fine, 3] — fine mesh vertex positions
        faces:    int array of shape   [F_fine, 3] — triangle face indices
        decimation_factors: list of floats in (0, 1); one per coarsening step.
                            The list goes from fine→coarse so that
                            meshes[0] is the finest and meshes[-1] is the
                            coarsest.

    Returns:
        meshes_list:      list of (vertices, faces) tuples, finest first
        adjacency_list:   list of scipy sparse adjacency matrices
        downsampling_list: list of D matrices  D[i]: [N_{i+1} × N_i] (fine→coarse)
        upsampling_list:  list of U matrices  U[i]: [N_i × N_{i+1}] (coarse→fine)
    """
    if not HAS_OPEN3D:
        raise RuntimeError("open3d is required for mesh simplification.")

    meshes_list = [(vertices, faces)]
    adjacency_list = [build_adjacency_matrix(vertices, faces)]
    downsampling_list = []
    upsampling_list = []

    current_verts = vertices
    current_faces = faces

    for factor in decimation_factors:
        target_triangles = max(1, int(len(current_faces) * factor))

        # Decimate mesh using Open3D
        o3d_mesh = _to_open3d_mesh(current_verts, current_faces)
        coarse_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
        coarse_mesh = coarse_mesh.remove_duplicated_vertices()
        coarse_mesh = coarse_mesh.remove_degenerate_triangles()
        coarse_mesh = coarse_mesh.remove_duplicated_triangles()
        coarse_mesh.compute_vertex_normals()

        coarse_verts = np.asarray(coarse_mesh.vertices, dtype=np.float32)
        coarse_faces = np.asarray(coarse_mesh.triangles, dtype=np.int32)

        # Build D and U between current (fine) and coarse levels
        D, U = build_sampling_matrices(current_verts, coarse_verts, current_faces)

        downsampling_list.append(D)
        upsampling_list.append(U)

        A_coarse = build_adjacency_matrix(coarse_verts, coarse_faces)
        meshes_list.append((coarse_verts, coarse_faces))
        adjacency_list.append(A_coarse)

        current_verts = coarse_verts
        current_faces = coarse_faces

    return meshes_list, adjacency_list, downsampling_list, upsampling_list


def build_adjacency_matrix(vertices: np.ndarray, faces: np.ndarray) -> sp.csr_matrix:
    """
    Build a binary (0/1) symmetric sparse adjacency matrix from mesh topology.

    Edges are the unique pairs of vertices connected by a triangle edge.
    Self-loops are NOT included.

    Args:
        vertices: [N, 3] vertex positions (used only for shape)
        faces:    [F, 3] triangle face indices

    Returns:
        scipy CSR sparse matrix of shape [N, N] with values in {0, 1}
    """
    N = len(vertices)
    row_idx = []
    col_idx = []

    # Each triangle contributes three undirected edges: (0,1), (1,2), (0,2)
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        row_idx.append(faces[:, i])
        col_idx.append(faces[:, j])
        # Symmetric
        row_idx.append(faces[:, j])
        col_idx.append(faces[:, i])

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    data = np.ones(len(row_idx), dtype=np.float32)

    A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    # Remove self-loops introduced by degenerate triangles (safety)
    A = A - sp.diags(A.diagonal())
    # Clip to binary (some entries may have been counted multiple times)
    A.data = np.ones_like(A.data)
    return A


def build_sampling_matrices(
    fine_verts: np.ndarray,
    coarse_verts: np.ndarray,
    fine_faces: np.ndarray,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Build downsampling matrix D and upsampling matrix U via nearest-neighbour
    mapping between the fine and coarse vertex sets.

    D[coarse_i, fine_j] = weight of fine node j contributing to coarse node i.
    Each fine node is mapped to its single nearest coarse node, so D is a
    row-normalised indicator matrix of shape [N_coarse, N_fine].

    U = D^T, giving a [N_fine, N_coarse] matrix that spreads coarse values
    back to fine nodes (simple nearest-neighbour interpolation).

    Args:
        fine_verts:   [N_fine,   3] vertex positions of the fine mesh
        coarse_verts: [N_coarse, 3] vertex positions of the coarse mesh
        fine_faces:   [F_fine, 3]  face indices of the fine mesh (unused here,
                      kept for API compatibility with barycentric alternatives)

    Returns:
        D: scipy CSR matrix [N_coarse, N_fine]  (downsampling)
        U: scipy CSR matrix [N_fine,   N_coarse] (upsampling)
    """
    N_fine = len(fine_verts)
    N_coarse = len(coarse_verts)

    # For each fine node find the nearest coarse node
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coarse_verts)
    _, indices = nbrs.kneighbors(fine_verts)   # [N_fine, 1]
    coarse_indices = indices[:, 0]             # [N_fine]

    # Build D: [N_coarse × N_fine]
    # Each fine node j maps to coarse node coarse_indices[j]
    row = coarse_indices                        # coarse node index
    col = np.arange(N_fine)                     # fine node index
    data = np.ones(N_fine, dtype=np.float32)

    D_raw = sp.csr_matrix((data, (row, col)), shape=(N_coarse, N_fine))

    # Row-normalise so that D is a proper weighted average
    row_sums = np.asarray(D_raw.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0              # avoid divide-by-zero
    D_norm = sp.diags(1.0 / row_sums) @ D_raw

    D = D_norm.tocsr().astype(np.float32)
    U = D.T.tocsr().astype(np.float32)

    return D, U


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_open3d_mesh(vertices: np.ndarray, faces: np.ndarray) -> "o3d.geometry.TriangleMesh":
    """Convert numpy arrays to an Open3D TriangleMesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    return mesh
