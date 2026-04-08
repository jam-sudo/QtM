"""Module D: Batched PyTorch ODE RHS for the 34-compartment PBPK model.

Faithful reimplementation of Sisyphus engine/flux.py in batched, differentiable
PyTorch. All flux computations are fully vectorized — no Python loops in the
forward pass.

The ODE state y has shape [batch, 34] representing drug amounts (mg) in each
compartment. Drug-dependent parameters (Kp, enzyme affinities, PS, fup, etc.)
are provided per-batch as a DrugParams dataclass.

Usage with torchdiffeq:
    func = PBPKFunc(topology)
    wrapper = ODEWrapper(func, drug_params)
    solution = odeint_adjoint(wrapper, y0, t_eval, method="dopri5")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.data.graph_topology import PBPKTopology


@dataclass
class DrugParams:
    """Batched drug-dependent parameters for the ODE system.

    All tensors have batch dimension first.
    """
    # Tissue partition coefficients
    kp: Tensor              # [batch, 15] for 15 Kp tissues (ordered as KP_TISSUE_NAMES)

    # Enzyme affinities (µL/min/pmol)
    enzyme_affinities: Tensor  # [batch, 5] for 5 enzyme tags (ordered as ENZYME_TAGS)

    # Permeability-surface area products (L/h) for perm-limited tissues
    ps: Tensor              # [batch, 4] for adipose, bone, muscle, skin

    # Scalar ADME parameters
    fup: Tensor             # [batch] fraction unbound in plasma
    rbp: Tensor             # [batch] blood:plasma ratio
    peff: Tensor            # [batch] effective permeability (×10⁻⁴ cm/s)
    renal_cl: Tensor        # [batch] renal clearance (L/h)
    particle_radius: Tensor # [batch] particle radius (µm)


class PBPKFunc(nn.Module):
    """Batched ODE right-hand side for the 34-compartment PBPK model.

    Fixed topology tensors are stored as buffers. Drug parameters are
    injected per-solve via set_drug_params().

    The RHS computes dy/dt = f(t, y) for all 34 compartments simultaneously
    across a batch of molecules. All computations are vectorized.
    """

    def __init__(self, topo: PBPKTopology):
        super().__init__()

        # Store fixed topology as non-trainable buffers
        self.register_buffer("volumes", topo.volumes)               # [34]
        self.register_buffer("is_blood_pool", topo.is_blood_pool)   # [34] bool
        self.register_buffer("flow_src", topo.flow_src)             # [n_flow]
        self.register_buffer("flow_tgt", topo.flow_tgt)             # [n_flow]
        self.register_buffer("flow_rates", topo.flow_rates)         # [n_flow]
        self.register_buffer("diff_vasc_idx", topo.diff_vasc_idx)   # [n_diff]
        self.register_buffer("diff_tissue_idx", topo.diff_tissue_idx)
        self.register_buffer("diff_ps_default", topo.diff_ps_default)
        self.register_buffer("transit_src", topo.transit_src)       # [n_transit]
        self.register_buffer("transit_tgt", topo.transit_tgt)
        self.register_buffer("transit_rates", topo.transit_rates)
        self.register_buffer("absorb_src", topo.absorb_src)         # [n_absorb]
        self.register_buffer("absorb_tgt", topo.absorb_tgt)
        self.register_buffer("absorb_ka_fractions", topo.absorb_ka_fractions)
        self.register_buffer("clear_src", topo.clear_src)           # [n_clear]
        self.register_buffer("clear_tgt", topo.clear_tgt)
        self.register_buffer("enzyme_abundances", topo.enzyme_abundances)  # [n_enz_nodes, 5]
        self.register_buffer("enzyme_node_indices", topo.enzyme_node_indices)
        self.register_buffer("ivive_scalings", topo.ivive_scalings)
        self.register_buffer("kp_tissue_node_indices", topo.kp_tissue_node_indices)  # [15]

        # Store metadata
        self.n_nodes = topo.n_nodes
        self.clear_models = topo.clear_models
        self.diff_lookup_names = topo.diff_lookup_names
        self.kp_tissue_names = topo.kp_tissue_names
        self.venous_idx = topo.venous_idx
        self.kidney_idx = topo.kidney_idx

        # Pre-compute full-node Kp mapping for flow flux
        # For blood pools and lumens/sinks, Kp=1 and rbp correction applies differently
        # Build a mask: which nodes are "tissue" (need Kp correction in flow)
        self.register_buffer(
            "tissue_mask",
            (~topo.is_blood_pool & ~topo.is_lumen & ~topo.is_sink),  # [34] bool
        )

        # Precompute total inflow to clearance source nodes (for well-stirred model)
        # Liver total inflow = hepatic artery + portal vein flows
        # Gut wall total inflow = arterial flow to gut_wall
        # Kidney total inflow = arterial flow to kidney
        self._clear_inflows: Optional[Tensor] = None
        self._precompute_clearance_inflows(topo)

        self._drug_params: Optional[DrugParams] = None

    def _precompute_clearance_inflows(self, topo: PBPKTopology) -> None:
        """Compute total perfusion inflow for each clearance source node."""
        inflows = []
        for src_idx_item in topo.clear_src.tolist():
            # Sum all flow rates targeting this node
            mask = topo.flow_tgt == src_idx_item
            total = topo.flow_rates[mask].sum()
            inflows.append(total.item())
        self.register_buffer("clear_inflows", torch.tensor(inflows, dtype=torch.float32))

    def set_drug_params(self, params: DrugParams) -> None:
        """Inject drug parameters for the current batch."""
        self._drug_params = params

    def _expand_kp_to_nodes(self, kp_15: Tensor) -> Tensor:
        """Map 15 Kp tissue values to full 34-node vector.

        Non-tissue nodes (blood pools, lumens, sinks) get Kp=1.

        Args:
            kp_15: [batch, 15] Kp values.

        Returns:
            kp_full: [batch, 34] Kp values for all nodes.
        """
        batch = kp_15.shape[0]
        kp_full = torch.ones(batch, self.n_nodes, device=kp_15.device, dtype=kp_15.dtype)
        kp_full[:, self.kp_tissue_node_indices] = kp_15
        return kp_full

    def _expand_ps_to_diff(self, ps_4: Tensor) -> Tensor:
        """Map 4 PS values to diffusion edges, matching diff_lookup_names order.

        Args:
            ps_4: [batch, 4] PS products for adipose, bone, muscle, skin.

        Returns:
            ps_diff: [batch, n_diff] PS for each diffusion edge.
        """
        # diff_lookup_names order from YAML: adipose, muscle, bone, skin
        # ps_4 order: adipose, bone, muscle, skin (alphabetical from projector)
        # Need to map correctly
        ps_map = {"adipose": 0, "bone": 1, "muscle": 2, "skin": 3}
        indices = [ps_map[name] for name in self.diff_lookup_names]
        return ps_4[:, indices]

    def forward(self, t: float, y: Tensor) -> Tensor:
        """Compute dy/dt for the PBPK system.

        Args:
            t: Current time (hours). Unused in autonomous ODE but kept for interface.
            y: [batch, 34] amounts in mg.

        Returns:
            dydt: [batch, 34] rate of change (mg/h).
        """
        dp = self._drug_params
        assert dp is not None, "Call set_drug_params() before forward()"

        batch = y.shape[0]
        # Clamp amounts to non-negative for numerical stability
        # (negative amounts are non-physical and cause feedback instability)
        y = y.clamp(min=0.0)
        dydt = torch.zeros_like(y)

        # Expand drug params to full node dimensions
        kp_full = self._expand_kp_to_nodes(dp.kp)          # [batch, 34]
        fup = dp.fup.unsqueeze(1)                           # [batch, 1]
        rbp = dp.rbp.unsqueeze(1)                           # [batch, 1]

        # ── 1. Flow flux (31 edges) ──────────────────────────────
        # C_out for tissue: A * rbp / (V * Kp)
        # C_out for blood pool: A / V
        # Unified: C_out = A * rbp_factor / (V * kp_factor)
        #   where rbp_factor=rbp for tissue, =1 for blood pool
        #   and kp_factor=Kp for tissue, =1 for blood pool
        #   BUT: for blood pools, rbp and Kp are both 1 already in kp_full
        #   Actually: blood pool nodes have kp_full=1 by construction.
        #   For blood pool nodes, we want C_out = A / V (no rbp correction).
        #   For tissue nodes, C_out = A * rbp / (V * Kp).

        amounts_src = y[:, self.flow_src]                   # [batch, n_flow]
        v_src = self.volumes[self.flow_src]                 # [n_flow]
        kp_src = kp_full[:, self.flow_src]                  # [batch, n_flow]

        # Build rbp factor: rbp for tissue sources, 1.0 for blood pool sources
        is_tissue_src = self.tissue_mask[self.flow_src]     # [n_flow] bool
        rbp_factor = torch.where(
            is_tissue_src.unsqueeze(0).expand(batch, -1),
            rbp.expand(batch, len(self.flow_src)),
            torch.ones(batch, len(self.flow_src), device=y.device),
        )

        c_out = amounts_src * rbp_factor / (v_src.unsqueeze(0) * kp_src + 1e-30)
        flux_flow = self.flow_rates.unsqueeze(0) * c_out    # [batch, n_flow]

        # Subtract from source, add to target using scatter_add
        dydt.scatter_add_(1, self.flow_src.unsqueeze(0).expand(batch, -1), -flux_flow)
        dydt.scatter_add_(1, self.flow_tgt.unsqueeze(0).expand(batch, -1), flux_flow)

        # ── 2. Clearance flux (3 edges) ──────────────────────────
        for i, model in enumerate(self.clear_models):
            src_i = self.clear_src[i]
            tgt_i = self.clear_tgt[i]

            if model == "well_stirred":
                # CLint_organ = sum(abundance * affinity * ivive_scaling)
                # Find which enzyme node this is
                enz_mask = (self.enzyme_node_indices == src_i)
                if not enz_mask.any():
                    continue
                enz_idx = enz_mask.nonzero(as_tuple=True)[0][0]

                abundances = self.enzyme_abundances[enz_idx]     # [5]
                ivive = self.ivive_scalings[enz_idx]
                affinities = dp.enzyme_affinities                # [batch, 5]

                clint_organ = (abundances.unsqueeze(0) * affinities * ivive).sum(dim=1)  # [batch]

                q = self.clear_inflows[i]                        # scalar
                # CL = (Q * fup * CLint) / (Q + fup * CLint)
                fup_1d = dp.fup                                  # [batch]
                denom = q + fup_1d * clint_organ + 1e-30
                clh = (q * fup_1d * clint_organ) / denom         # [batch]

                # C_out from organ
                v = self.volumes[src_i]
                kp_i = kp_full[:, src_i]                         # [batch]
                c_out_i = y[:, src_i] * dp.rbp / (v * kp_i + 1e-30)  # [batch]

                rate = clh * c_out_i                             # [batch]

            elif model == "gfr_filtration":
                # rate = renal_cl * C_plasma
                v = self.volumes[src_i]
                kp_i = kp_full[:, src_i]
                c_plasma = y[:, src_i] * dp.rbp / (v * kp_i + 1e-30)
                rate = dp.renal_cl * c_plasma                    # [batch]

            else:
                continue

            dydt[:, src_i] -= rate
            dydt[:, tgt_i] += rate

        # ── 3. Transit flux (8 edges) ────────────────────────────
        amounts_transit = y[:, self.transit_src]                  # [batch, n_transit]
        flux_transit = self.transit_rates.unsqueeze(0) * amounts_transit
        dydt.scatter_add_(1, self.transit_src.unsqueeze(0).expand(batch, -1), -flux_transit)
        dydt.scatter_add_(1, self.transit_tgt.unsqueeze(0).expand(batch, -1), flux_transit)

        # ── 4. Absorption flux (8 edges) ─────────────────────────
        # ka = 2.88 * Peff * ka_fraction / particle_radius
        ka = (
            2.88
            * dp.peff.unsqueeze(1)                               # [batch, 1]
            * self.absorb_ka_fractions.unsqueeze(0)              # [1, n_absorb]
            / (dp.particle_radius.unsqueeze(1) + 1e-30)         # [batch, 1]
        )  # [batch, n_absorb]

        amounts_absorb = y[:, self.absorb_src]                   # [batch, n_absorb]
        flux_absorb = ka * amounts_absorb
        dydt.scatter_add_(1, self.absorb_src.unsqueeze(0).expand(batch, -1), -flux_absorb)
        dydt.scatter_add_(1, self.absorb_tgt.unsqueeze(0).expand(batch, -1), flux_absorb)

        # ── 5. Diffusion flux (4 edges) ──────────────────────────
        ps_diff = self._expand_ps_to_diff(dp.ps)                 # [batch, n_diff]

        v_vasc = self.volumes[self.diff_vasc_idx]                # [n_diff]
        v_tissue = self.volumes[self.diff_tissue_idx]            # [n_diff]
        kp_tissue = kp_full[:, self.diff_tissue_idx]             # [batch, n_diff]

        c_vasc = y[:, self.diff_vasc_idx] / (v_vasc.unsqueeze(0) + 1e-30)
        c_tissue = y[:, self.diff_tissue_idx] / (v_tissue.unsqueeze(0) + 1e-30)

        cu_vasc = dp.fup.unsqueeze(1) * c_vasc / (dp.rbp.unsqueeze(1) + 1e-30)
        cu_tissue = dp.fup.unsqueeze(1) * c_tissue / (kp_tissue + 1e-30)

        flux_diff = ps_diff * (cu_vasc - cu_tissue)              # [batch, n_diff]

        dydt.scatter_add_(1, self.diff_vasc_idx.unsqueeze(0).expand(batch, -1), -flux_diff)
        dydt.scatter_add_(1, self.diff_tissue_idx.unsqueeze(0).expand(batch, -1), flux_diff)

        return dydt


class ODEWrapper(nn.Module):
    """Wraps PBPKFunc for torchdiffeq which expects func(t, y) signature."""

    def __init__(self, pbpk_func: PBPKFunc, drug_params: DrugParams):
        super().__init__()
        self.pbpk_func = pbpk_func
        self.pbpk_func.set_drug_params(drug_params)

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        return self.pbpk_func(t, y)
