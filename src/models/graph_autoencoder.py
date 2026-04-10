"""
Graph Autoencoder (GCA) — encoder and decoder used at each hierarchy level.

The encoder applies a stack of ChebConv layers followed by a flattening step
and a linear bottleneck to produce a fixed-size latent vector.

The decoder mirrors this structure: a linear projection from the latent space
back to the node feature space, followed by a stack of ChebConv layers in
reverse order.  The final ChebConv layer uses a linear activation so that the
output can represent arbitrary displacement values.

Architecture (as in the paper, using filter sizes [6, 12, 24]):

    Encoder:
        ChebConv  3 → 6   ELU
        ChebConv  6 → 12  ELU
        ChebConv 12 → 24  ELU
        Flatten  [B, N, 24] → [B, N*24]
        Linear   N*24 → latent_dim

    Decoder:
        Linear   latent_dim → N*24
        Reshape  [B, N*24]  → [B, N, 24]
        ChebConv 24 → 12  ELU
        ChebConv 12 → 6   ELU
        ChebConv  6 → 3   (linear, no activation)
"""

import scipy.sparse as sp
import torch
import torch.nn as nn
from typing import List

from src.layers.graph_conv import ChebConv


class GraphEncoder(nn.Module):
    """
    Graph encoder: ChebConv stack + linear bottleneck.

    Args:
        n_nodes:      number of nodes N in the (coarse) mesh at this level
        in_features:  node feature size of the input (typically 3 for xyz)
        latent_dim:   size of the bottleneck latent vector
        filter_sizes: list of output feature sizes for each ChebConv layer,
                      e.g. [6, 12, 24]
        cheb_order:   Chebyshev polynomial order K (shared across layers)
        adjacency:    scipy sparse adjacency matrix [N, N] (no self-loops)
    """

    def __init__(
        self,
        n_nodes: int,
        in_features: int,
        latent_dim: int,
        filter_sizes: List[int],
        cheb_order: int,
        adjacency: sp.spmatrix,
    ):
        super().__init__()
        self.n_nodes     = n_nodes
        self.latent_dim  = latent_dim
        self.final_feats = filter_sizes[-1]

        # Build ChebConv stack
        conv_layers = []
        prev_feats = in_features
        for out_feats in filter_sizes:
            conv_layers.append(
                ChebConv(
                    in_features=prev_feats,
                    out_features=out_feats,
                    order=cheb_order,
                    adjacency=adjacency,
                    activation=nn.ELU(),
                )
            )
            prev_feats = out_feats
        self.conv_layers = nn.ModuleList(conv_layers)

        # Linear bottleneck: N * final_feats → latent_dim
        self.bottleneck = nn.Linear(n_nodes * self.final_feats, latent_dim)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_features] displacement field on the coarse mesh

        Returns:
            z: [B, latent_dim] latent representation
        """
        h = x
        for conv in self.conv_layers:
            h = conv(h)                              # [B, N, feats]

        B = h.shape[0]
        h = h.reshape(B, self.n_nodes * self.final_feats)  # flatten
        z = self.bottleneck(h)                       # [B, latent_dim]
        return z


class GraphDecoder(nn.Module):
    """
    Graph decoder: linear projection from latent + reversed ChebConv stack.

    Args:
        n_nodes:      number of nodes N in the (coarse) mesh at this level
        out_features: number of output node features (typically 3 for xyz)
        latent_dim:   size of the latent vector
        filter_sizes: list of feature sizes matching those used in the encoder,
                      e.g. [6, 12, 24].  The decoder reverses this list.
        cheb_order:   Chebyshev polynomial order K (shared across layers)
        adjacency:    scipy sparse adjacency matrix [N, N] (no self-loops)
    """

    def __init__(
        self,
        n_nodes: int,
        out_features: int,
        latent_dim: int,
        filter_sizes: List[int],
        cheb_order: int,
        adjacency: sp.spmatrix,
    ):
        super().__init__()
        self.n_nodes    = n_nodes
        self.latent_dim = latent_dim
        self.first_feats = filter_sizes[-1]   # largest feature size (innermost)

        # Linear projection: latent_dim → N * first_feats
        self.projection = nn.Linear(latent_dim, n_nodes * self.first_feats)

        # Reversed ChebConv stack; the last layer uses linear (no) activation
        reversed_sizes = list(reversed(filter_sizes))   # e.g. [24, 12, 6]
        conv_layers = []
        prev_feats = reversed_sizes[0]
        # Hidden layers with ELU
        for out_feats in reversed_sizes[1:]:
            conv_layers.append(
                ChebConv(
                    in_features=prev_feats,
                    out_features=out_feats,
                    order=cheb_order,
                    adjacency=adjacency,
                    activation=nn.ELU(),
                )
            )
            prev_feats = out_feats
        # Final layer: map to output features, no activation (linear)
        conv_layers.append(
            ChebConv(
                in_features=prev_feats,
                out_features=out_features,
                order=cheb_order,
                adjacency=adjacency,
                activation=None,   # linear output
            )
        )
        self.conv_layers = nn.ModuleList(conv_layers)

    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim] latent vector

        Returns:
            y: [B, N, out_features] reconstructed node features
        """
        B = z.shape[0]
        h = self.projection(z)                              # [B, N*first_feats]
        h = h.reshape(B, self.n_nodes, self.first_feats)   # [B, N, first_feats]

        for conv in self.conv_layers:
            h = conv(h)

        return h   # [B, N, out_features]
