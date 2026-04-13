"""
Graph Autoencoder (GCA) — encoder and decoder used at each hierarchy level.

Architecture (e.g. filter_sizes=[6, 12, 24]):

    Encoder:
        ChebConv  3 → 6   ELU
        ChebConv  6 → 12  ELU
        ChebConv 12 → 24  ELU
        Flatten + Linear  N*24 → latent_dim

    Decoder:
        Linear   latent_dim → N*24
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
        self.n_nodes = n_nodes
        self.final_feats = filter_sizes[-1]

        sizes = [in_features] + list(filter_sizes)
        self.convs = nn.ModuleList([
            ChebConv(sizes[i], sizes[i + 1], cheb_order, adjacency)
            for i in range(len(filter_sizes))
        ])
        self.bottleneck = nn.Linear(n_nodes * self.final_feats, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, in_features] → [B, latent_dim]"""
        for conv in self.convs:
            x = conv(x)
        return self.bottleneck(x.reshape(x.shape[0], -1))


class GraphDecoder(nn.Module):
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
        self.n_nodes = n_nodes
        self.first_feats = filter_sizes[-1]

        self.projection = nn.Linear(latent_dim, n_nodes * self.first_feats)

        # Reversed sizes; last layer is linear (no activation)
        sizes = list(reversed(filter_sizes)) + [out_features]
        activations = [nn.ELU()] * (len(sizes) - 2) + [None]
        self.convs = nn.ModuleList([
            ChebConv(sizes[i], sizes[i + 1], cheb_order, adjacency, activation=activations[i])
            for i in range(len(sizes) - 1)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim] → [B, N, out_features]"""
        B = z.shape[0]
        x = self.projection(z).reshape(B, self.n_nodes, self.first_feats)
        for conv in self.convs:
            x = conv(x)
        return x
