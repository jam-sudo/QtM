"""Module C: Hierarchical parameter projection from Z_mol to PBPK parameters.

Maps the molecular latent vector Z_mol to 30 drug-dependent ODE parameters
with physiological clamping to enforce biological boundaries.

Architecture:
    Z_mol [B, 128]
      → shared_mlp [B, 64]
      → tissue_group_attention [B, 5, 64]  (5 super-groups)
      → per-group heads → raw parameters
      → physiological clamping → constrained parameters

Output: DrugParams with 30 parameters total:
    - 15 Kp (tissue partition coefficients, residual on baseline)
    - 5 enzyme affinities
    - 4 PS products
    - 6 scalar ADME (fup, rbp, peff, renal_cl, particle_radius, solubility)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.ode_system import DrugParams


# ── Tissue super-group definitions ───────────────────────────────
# Maps each of 15 Kp tissues to one of 5 super-groups.
# Order matches KP_TISSUE_NAMES (alphabetical).
TISSUE_TO_GROUP = {
    "adipose": 0,       # Adipose
    "bone": 2,          # Slowly Perfused
    "brain": 1,         # Highly Perfused
    "gut_wall": 1,      # Highly Perfused
    "heart": 1,         # Highly Perfused
    "kidney": 1,        # Highly Perfused
    "liver": 1,         # Highly Perfused
    "lung": 1,          # Highly Perfused
    "muscle": 2,        # Slowly Perfused
    "pancreas": 3,      # Other
    "reproductive": 3,  # Other
    "rest": 3,          # Other
    "skin": 2,          # Slowly Perfused
    "spleen": 1,        # Highly Perfused
    "thymus": 3,        # Other
}
# Group 4 (Elimination) is handled separately for enzyme/ADME outputs.

GROUP_NAMES = ["Adipose", "Highly Perfused", "Slowly Perfused", "Other", "Elimination"]
N_GROUPS = 5
N_KP_TISSUES = 15
N_ENZYMES = 5
N_PS = 4
N_SCALAR_ADME = 6  # fup, rbp, peff, renal_cl, particle_radius, solubility


class HierarchicalProjector(nn.Module):
    """Project Z_mol to 30 drug-dependent PBPK parameters.

    Uses tissue-grouping attention: the 34 compartments are clustered into
    5 super-groups, a shared representation is extracted per group, then
    deconvolved to individual compartment parameters.

    All outputs are physiologically clamped:
        - Kp: residual on Kp_baseline via exp(tanh(Δ)) — range [base/e, base×e]
        - Enzyme affinities: softplus (strictly positive)
        - PS products: softplus (strictly positive)
        - fup: sigmoid scaled to [0.01, 1.0]
        - rbp: sigmoid scaled to [0.5, 2.0]
        - peff, renal_cl, particle_radius, solubility: softplus

    Args:
        z_dim: Dimension of Z_mol input. Default 128.
        hidden_dim: Internal hidden dimension. Default 64.
    """

    def __init__(self, z_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()

        # Shared encoder with dropout for regularization
        self.shared_mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Group attention: learn to weight 5 super-groups
        self.group_queries = nn.Parameter(torch.randn(N_GROUPS, hidden_dim) * 0.02)
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)

        # Per-group output heads
        # Group 0 (Adipose): 1 Kp
        # Group 1 (Highly Perfused): 7 Kp (brain, gut_wall, heart, kidney, liver, lung, spleen)
        # Group 2 (Slowly Perfused): 3 Kp (bone, muscle, skin) + 4 PS (same tissues + adipose mapped here)
        # Group 3 (Other): 4 Kp (pancreas, reproductive, rest, thymus)
        # Group 4 (Elimination): 5 enzyme affinities + 6 scalar ADME

        self.kp_head = nn.Linear(hidden_dim, N_KP_TISSUES)    # 15 delta_Kp values
        self.enzyme_head = nn.Linear(hidden_dim, N_ENZYMES)    # 5 enzyme affinities
        self.ps_head = nn.Linear(hidden_dim, N_PS)             # 4 PS products
        self.adme_head = nn.Linear(hidden_dim, N_SCALAR_ADME)  # 6 scalar ADME

        # Build group assignment tensor for Kp tissues
        self._group_indices = torch.tensor(
            [TISSUE_TO_GROUP[t] for t in sorted(TISSUE_TO_GROUP.keys())],
            dtype=torch.long,
        )

    def forward(self, z_mol: Tensor, kp_baseline: Tensor) -> DrugParams:
        """Project molecular latent to constrained PBPK parameters.

        Args:
            z_mol: [batch, z_dim] molecular latent vector from encoder.
            kp_baseline: [batch, 15] Rodgers & Rowland Kp baselines.

        Returns:
            DrugParams with all 30 parameters, physiologically clamped.
        """
        batch = z_mol.shape[0]

        # Shared representation
        h = self.shared_mlp(z_mol)  # [batch, hidden]

        # Group attention: compute attention-weighted context per group
        # queries: [5, H], keys: [batch, H]
        attn_keys = self.attn_proj(h)  # [batch, H]
        # Score: [batch, 5] = queries @ keys^T
        scores = torch.matmul(self.group_queries, attn_keys.t()).t()  # [batch, 5]
        attn_weights = F.softmax(scores, dim=-1)  # [batch, 5]

        # Weighted context: [batch, 5, H]
        group_context = attn_weights.unsqueeze(-1) * h.unsqueeze(1)  # [batch, 5, H]

        # Aggregate group contexts for each head
        # Use mean of all group contexts as unified representation
        h_unified = group_context.mean(dim=1)  # [batch, H]

        # ── Kp (residual) ────────────────────────────────────────
        delta_kp_raw = self.kp_head(h_unified)          # [batch, 15]
        # Kp = Kp_baseline × exp(tanh(Δ)) — range [base/e, base×e]
        kp = kp_baseline * torch.exp(torch.tanh(delta_kp_raw))
        # Hard clamp: Kp_min=1.0 guarantees rk4 (dt=0.005h) stability.
        # Worst case: kidney λ = Q*rbp/(V*Kp) = 74.1*2/(0.31*1.0) = 478 h⁻¹
        # → λ*dt = 2.39 < 2.78 (rk4 stability limit).
        # Kp<1.0 causes kidney λ*dt > 2.78 → solver divergence + NaN gradients.
        kp = kp.clamp(min=1.0, max=50.0)

        # ── Enzyme affinities ────────────────────────────────────
        # Bounded sigmoid [0.001, 0.1] µL/min/pmol.
        # Max CLint_liver = 9.2M × 0.1 × 6e-5 = 55 L/h (physiological).
        # Unbounded softplus caused CLint=893 L/h at init → solver blowup.
        enz_raw = self.enzyme_head(h_unified)            # [batch, 5]
        enzyme_affinities = torch.sigmoid(enz_raw) * 0.099 + 0.001

        # ── PS products ──────────────────────────────────────────
        # Bounded sigmoid [1.0, 50.0] L/h.
        # Prevents extreme diffusion flux that increases stiffness.
        ps_raw = self.ps_head(h_unified)                 # [batch, 4]
        ps = torch.sigmoid(ps_raw) * 49.0 + 1.0

        # ── Scalar ADME ──────────────────────────────────────────
        adme_raw = self.adme_head(h_unified)             # [batch, 6]
        # adme_raw[:, 0] → fup: sigmoid to [0.01, 1.0]
        fup = torch.sigmoid(adme_raw[:, 0]) * 0.99 + 0.01
        # adme_raw[:, 1] → rbp: sigmoid to [0.5, 2.0]
        rbp = torch.sigmoid(adme_raw[:, 1]) * 1.5 + 0.5
        # adme_raw[:, 2] → peff: bounded sigmoid [0.01, 10.0] (×10⁻⁴ cm/s)
        # Prevents extreme absorption rate ka ∝ Peff / particle_radius.
        peff = torch.sigmoid(adme_raw[:, 2]) * 9.99 + 0.01
        # adme_raw[:, 3] → renal_cl: bounded sigmoid [0.01, 20.0] L/h
        # Max ≈ 2.7× GFR, allows tubular secretion but prevents instability.
        renal_cl = torch.sigmoid(adme_raw[:, 3]) * 19.99 + 0.01
        # adme_raw[:, 4] → particle_radius: sigmoid to [5, 50] µm
        particle_radius = torch.sigmoid(adme_raw[:, 4]) * 45.0 + 5.0
        # adme_raw[:, 5] → solubility: softplus (mg/mL)
        _solubility = F.softplus(adme_raw[:, 5])  # not used in ODE but useful for multi-task

        return DrugParams(
            kp=kp,
            enzyme_affinities=enzyme_affinities,
            ps=ps,
            fup=fup,
            rbp=rbp,
            peff=peff,
            renal_cl=renal_cl,
            particle_radius=particle_radius,
        )
