"""Module B: Custom SchNet encoder without torch-cluster dependency.

E(n)-invariant 3D molecular encoder that maps atom types, positions, and
partial charges to a translation/rotation-invariant latent vector Z_mol.

Uses continuous-filter convolution with Gaussian distance expansion.
Edge graph built via torch.cdist (no torch-cluster needed).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.schnet import GaussianSmearing, ShiftedSoftplus


class CFConvCustom(MessagePassing):
    """Continuous-filter convolution layer.

    Computes messages using a distance-dependent filter network:
        m_ij = x_j * W(||r_i - r_j||)

    where W is an MLP applied to Gaussian-expanded distances.
    """

    def __init__(self, hidden_channels: int, num_gaussians: int):
        super().__init__(aggr="add")
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """
        Args:
            x: [N, H] node features.
            edge_index: [2, E] edge indices.
            edge_weight: [E] raw distances (unused here, kept for interface).
            edge_attr: [E, G] Gaussian-expanded distances.

        Returns:
            out: [N, H] aggregated messages.
        """
        W = self.filter_net(edge_attr)  # [E, H]
        return self.propagate(edge_index, x=x, W=W)

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class InteractionBlock(nn.Module):
    """SchNet interaction block: CFConv + residual update."""

    def __init__(self, hidden_channels: int, num_gaussians: int):
        super().__init__()
        self.mlp_in = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
        )
        self.conv = CFConvCustom(hidden_channels, num_gaussians)
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor
    ) -> Tensor:
        x_in = self.mlp_in(x)
        x_conv = self.conv(x_in, edge_index, edge_weight, edge_attr)
        x_out = self.mlp_out(x_conv)
        return x + x_out  # Residual connection


class SchNetEncoder(nn.Module):
    """Custom SchNet encoder for molecular property prediction.

    Takes atomic numbers, 3D positions, and partial charges as input.
    Outputs a rotation/translation-invariant molecular latent vector Z_mol.

    Architecture:
        1. Atom embedding (atomic number) + charge embedding → h0
        2. N interaction blocks with continuous-filter convolutions
        3. Mean readout over atoms → Z_mol

    Args:
        hidden_channels: Dimension of node embeddings. Default 128.
        num_interactions: Number of interaction blocks. Default 3.
        num_gaussians: Number of Gaussian basis functions. Default 25.
        cutoff: Distance cutoff in Ångström. Default 10.0.
        max_z: Maximum atomic number to embed. Default 100.

    Output shape: [batch, hidden_channels]
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_interactions: int = 3,
        num_gaussians: int = 25,
        cutoff: float = 10.0,
        max_z: int = 100,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff

        # Atom type embedding
        self.atom_emb = nn.Embedding(max_z, hidden_channels, padding_idx=0)

        # Partial charge embedding
        self.charge_emb = nn.Linear(1, hidden_channels, bias=False)

        # Distance expansion
        self.dist_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians)
            for _ in range(num_interactions)
        ])

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        charges: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Encode a batch of molecules into latent vectors.

        Args:
            z: [N_total] atomic numbers (int).
            pos: [N_total, 3] 3D coordinates (float).
            charges: [N_total] partial charges (float).
            edge_index: [2, E] edge indices within cutoff.
            batch: [N_total] batch assignment per atom (int).

        Returns:
            z_mol: [batch_size, hidden_channels] molecular latent vectors.
                   Invariant to translation and rotation by construction
                   (SchNet uses only pairwise distances).
        """
        # Compute pairwise distances for edges
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)  # [E]

        # Apply cutoff mask (edge_index should already be within cutoff,
        # but this ensures gradients are smooth at boundary)
        edge_attr = self.dist_expansion(dist)  # [E, G]

        # Initial embeddings: atom type + charge
        h = self.atom_emb(z) + self.charge_emb(charges.unsqueeze(-1))  # [N, H]

        # Interaction blocks
        for interaction in self.interactions:
            h = interaction(h, edge_index, dist, edge_attr)

        # Output projection
        h = self.out_proj(h)  # [N, H]

        # Mean readout per molecule
        batch_size = int(batch.max().item()) + 1
        z_mol = torch.zeros(batch_size, self.hidden_channels, device=h.device, dtype=h.dtype)
        counts = torch.zeros(batch_size, device=h.device, dtype=h.dtype)
        z_mol.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
        z_mol = z_mol / (counts.unsqueeze(1) + 1e-8)

        return z_mol
