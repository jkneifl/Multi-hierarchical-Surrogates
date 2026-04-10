"""
Graph convolution layers: Chebyshev (ChebConv) and standard GCN (GCNConv).

Both layers operate on batched node-feature tensors of shape [B, N, F] and
accept a pre-computed sparse graph matrix (the normalised Laplacian for
ChebConv, or the symmetrically-normalised adjacency for GCNConv).

References:
    Defferrard et al., "Convolutional Neural Networks on Graphs with Fast
    Localized Spectral Filtering", NeurIPS 2016.
    Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
    Networks", ICLR 2017.
"""

import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.utils.sparse_utils import scipy_sparse_to_torch, apply_sparse_to_batch


# ---------------------------------------------------------------------------
# Public helpers (re-exported for convenience)
# ---------------------------------------------------------------------------

# Re-export the canonical sparse utilities so callers can import from here.
scipy_sparse_to_torch = scipy_sparse_to_torch   # noqa: F811
_sparse_mm_batched    = apply_sparse_to_batch    # noqa: F811


# ---------------------------------------------------------------------------
# ChebConv
# ---------------------------------------------------------------------------

class ChebConv(nn.Module):
    """
    Chebyshev spectral graph convolution layer.

    Operates on batched node features x ∈ ℝ^{B × N × F_in} and returns
    features of shape [B, N, F_out].

    The normalised Laplacian used here is

        L̃ = −D^{-1/2} A D^{-1/2}

    (i.e., the symmetric normalised Laplacian *minus* the identity), so that
    its eigenvalues lie in [−1, 1].  The Chebyshev polynomials are then:

        T_0(L̃) x = x
        T_1(L̃) x = L̃ x
        T_k(L̃) x = 2 L̃ T_{k-1}(L̃) x − T_{k-2}(L̃) x

    All K terms are stacked along a new trailing dimension, reshaped to
    [B*N, F_in*K], and multiplied by a learned weight matrix W ∈ ℝ^{F_in*K × F_out}.

    Args:
        in_features:  number of input  node features F_in
        out_features: number of output node features F_out
        order:        Chebyshev polynomial order K  (K=1 → plain graph conv)
        adjacency:    scipy sparse adjacency matrix A of shape [N, N]
                      (binary, no self-loops).  The layer computes L̃ internally.
        bias:         if True, add a learnable bias of shape [F_out]
        activation:   optional activation function applied after the linear
                      transform.  Pass ``None`` for a linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        order: int,
        adjacency: sp.spmatrix,
        bias: bool = True,
        activation: nn.Module | None = nn.ELU(),
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.order        = order
        self.activation   = activation

        # Pre-compute L̃ = −D^{-1/2} A D^{-1/2} from the adjacency matrix
        L_tilde = _compute_scaled_laplacian(adjacency)
        # Register as a non-parameter buffer so it moves with .to(device)
        self.register_buffer("L_tilde", scipy_sparse_to_torch(L_tilde))

        # Learnable weight W ∈ ℝ^{F_in*K × F_out}
        self.weight = nn.Parameter(torch.empty(in_features * order, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    # ------------------------------------------------------------------

    def _reset_parameters(self):
        """Initialise weights with truncated normal std=0.1, bias=0.1."""
        nn.init.trunc_normal_(self.weight, std=0.1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F_in] node feature tensor

        Returns:
            [B, N, F_out] transformed node features
        """
        # ---------- Chebyshev recurrence ----------
        # T_0(L̃) x = x
        x0 = x                                          # [B, N, F_in]
        chebs = [x0]

        if self.order > 1:
            # T_1(L̃) x = L̃ x
            x1 = apply_sparse_to_batch(self.L_tilde, x0)  # [B, N, F_in]
            chebs.append(x1)

        for _ in range(2, self.order):
            # T_k = 2 L̃ T_{k-1} − T_{k-2}
            xk = 2.0 * apply_sparse_to_batch(self.L_tilde, chebs[-1]) - chebs[-2]
            chebs.append(xk)

        # ---------- Stack and apply linear transform ----------
        B, N, F = x.shape
        # Stack along new trailing dim: list of K tensors [B,N,F] → [B,N,F,K]
        stacked = torch.stack(chebs, dim=-1)            # [B, N, F_in, K]
        # Merge feature and polynomial dims, then merge batch and node dims
        stacked = stacked.reshape(B * N, F * self.order) # [B*N, F_in*K]

        # Linear: [B*N, F_in*K] @ [F_in*K, F_out] → [B*N, F_out]
        out = stacked @ self.weight
        if self.bias is not None:
            out = out + self.bias

        out = out.reshape(B, N, self.out_features)       # [B, N, F_out]

        if self.activation is not None:
            out = self.activation(out)

        return out

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"order={self.order}")


# ---------------------------------------------------------------------------
# GCNConv
# ---------------------------------------------------------------------------

class GCNConv(nn.Module):
    """
    Single-layer Graph Convolutional Network (Kipf & Welling, 2017).

    Uses the symmetrically-normalised adjacency with added self-loops:

        Â = D̃^{-1/2} (A + I) D̃^{-1/2}

    and computes  H' = Â H W + b.

    Args:
        in_features:  F_in
        out_features: F_out
        adjacency:    scipy sparse adjacency [N, N] (no self-loops expected)
        bias:         add learnable bias
        activation:   optional activation after linear transform
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        adjacency: sp.spmatrix,
        bias: bool = True,
        activation: nn.Module | None = nn.ELU(),
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.activation   = activation

        A_hat = _compute_normalised_adjacency(adjacency)
        self.register_buffer("A_hat", scipy_sparse_to_torch(A_hat))

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F_in]

        Returns:
            [B, N, F_out]
        """
        # Propagation: Â x → [B, N, F_in]
        agg = apply_sparse_to_batch(self.A_hat, x)

        B, N, _ = agg.shape
        # Linear: [B*N, F_in] @ [F_in, F_out] → [B*N, F_out]
        out = agg.reshape(B * N, self.in_features) @ self.weight
        if self.bias is not None:
            out = out + self.bias
        out = out.reshape(B, N, self.out_features)

        if self.activation is not None:
            out = self.activation(out)
        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ---------------------------------------------------------------------------
# Internal helpers for computing graph operators
# ---------------------------------------------------------------------------

def _compute_scaled_laplacian(A: sp.spmatrix) -> sp.csr_matrix:
    """
    Compute the scaled Laplacian L̃ = −D^{-1/2} A D^{-1/2}.

    This is the standard symmetric normalised Laplacian minus the identity:
        L = I − D^{-1/2} A D^{-1/2}  →  L̃ = L − I = −D^{-1/2} A D^{-1/2}

    The eigenvalues of L̃ lie in [−1, 1], which is the domain over which the
    Chebyshev polynomials are defined.

    Self-loops in A are removed before computing the Laplacian.
    """
    A = A.astype(np.float32)
    A = A - sp.diags(A.diagonal())  # remove self-loops

    N = A.shape[0]
    # Degree vector  d_i = Σ_j A_{ij}
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    # L̃ = −D^{-1/2} A D^{-1/2}
    L_tilde = -(D_inv_sqrt @ A @ D_inv_sqrt)
    return L_tilde.tocsr().astype(np.float32)


def _compute_normalised_adjacency(A: sp.spmatrix) -> sp.csr_matrix:
    """
    Compute Â = D̃^{-1/2} Ã D̃^{-1/2}  where  Ã = A + I.

    Used by the GCN layer.
    """
    A = A.astype(np.float32)
    N = A.shape[0]
    A_tilde = A + sp.eye(N, format='csr', dtype=np.float32)  # add self-loops

    d = np.asarray(A_tilde.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_hat.tocsr().astype(np.float32)
