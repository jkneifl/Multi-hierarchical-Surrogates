"""
Graph convolution layer wrapping torch_geometric.nn.ChebConv.

Provides a batched [B, N, F] interface on top of PyG's node-level ChebConv
by block-expanding the shared edge_index across all B graphs in the batch.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv as _PyGChebConv


def adjacency_to_edge_index(adjacency: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse adjacency matrix to a [2, E] edge_index tensor."""
    A = adjacency.tocoo()
    row = torch.from_numpy(A.row.astype(np.int64))
    col = torch.from_numpy(A.col.astype(np.int64))
    return torch.stack([row, col], dim=0)


class ChebConv(nn.Module):
    """
    Chebyshev graph convolution for batched [B, N, F] node features.

    Args:
        in_features:  number of input node features F_in
        out_features: number of output node features F_out
        order:        Chebyshev polynomial order K
        adjacency:    scipy sparse [N, N] adjacency (no self-loops)
        bias:         add a learnable bias
        activation:   optional activation applied after the convolution
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
        self.n_nodes = adjacency.shape[0]
        self.activation = activation

        self.register_buffer("edge_index", adjacency_to_edge_index(adjacency))
        self.conv = _PyGChebConv(in_features, out_features, K=order, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F_in]
        Returns:
            [B, N, F_out]
        """
        B, N, F = x.shape

        # Expand edge_index for all B graphs (block-diagonal)
        offsets = torch.arange(B, device=x.device) * N            # [B]
        batch_ei = (
            self.edge_index[None] + offsets.view(B, 1, 1)         # [B, 2, E]
        ).permute(1, 0, 2).reshape(2, -1)                         # [2, B*E]

        out = self.conv(x.reshape(B * N, F), batch_ei)            # [B*N, F_out]
        out = out.reshape(B, N, -1)

        if self.activation is not None:
            out = self.activation(out)
        return out
