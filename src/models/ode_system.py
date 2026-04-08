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


class PSSAWrapper(nn.Module):
    """Pseudo-Steady State Approximation wrapper for the PBPK ODE.

    Converts venous_blood, arterial_blood, and portal_vein from differential
    to algebraic equations, reducing state dimension from 34 to 31.

    These 3 blood pools have time constants < 35s (vs PK timescales of hours),
    so their amounts satisfy instantaneous flow balance:

        A_bp = V_bp × Σ(Q_in × C_out_src) / Q_out_total

    This eliminates eigenvalues up to λ=1482 h⁻¹ (portal vein τ=2.4s),
    reducing the stiffness ratio from ~1482:1 to ~118:1.

    The 34-node topology is fully preserved — blood pool amounts are computed
    algebraically at each ODE step, not removed from the system.
    """

    def __init__(
        self,
        pbpk_func: PBPKFunc,
        topo: "PBPKTopology",
        drug_params: DrugParams,
    ):
        super().__init__()
        self.pbpk_func = pbpk_func
        self.pbpk_func.set_drug_params(drug_params)
        self.drug_params = drug_params
        self.n_full = topo.n_nodes

        device = topo.volumes.device

        # PSSA node indices (sorted)
        pssa_idx_list = sorted([
            topo.venous_idx, topo.arterial_idx, topo.portal_vein_idx,
        ])
        ode_idx_list = [i for i in range(topo.n_nodes) if i not in pssa_idx_list]

        self.register_buffer("pssa_indices", torch.tensor(pssa_idx_list, dtype=torch.long, device=device))
        self.register_buffer("ode_indices", torch.tensor(ode_idx_list, dtype=torch.long, device=device))
        self.n_reduced = len(ode_idx_list)

        # Full-to-reduced index mapping (-1 for PSSA nodes)
        full_to_red = torch.full((topo.n_nodes,), -1, dtype=torch.long, device=device)
        for r_idx, f_idx in enumerate(ode_idx_list):
            full_to_red[f_idx] = r_idx
        self.register_buffer("full_to_reduced", full_to_red)

        # Useful mapped indices
        self.stomach_idx_reduced = full_to_red[topo.stomach_idx].item()
        self.venous_pssa_pos = pssa_idx_list.index(topo.venous_idx)
        self.venous_idx_full = topo.venous_idx

        # Topology data for PSSA computation
        self.register_buffer("volumes_full", topo.volumes)
        tissue_mask = (~topo.is_blood_pool & ~topo.is_lumen & ~topo.is_sink)
        self.register_buffer("tissue_mask", tissue_mask)

        # Pre-compute flow structure for each PSSA node
        self._build_pssa_structure(topo, pssa_idx_list, tissue_mask)

    def _build_pssa_structure(
        self,
        topo: "PBPKTopology",
        pssa_idx_list: list,
        tissue_mask: Tensor,
    ) -> None:
        """Pre-compute inflow sources and outflow rates for PSSA nodes.

        For each PSSA node, registers buffers:
            pssa_{i}_src_reduced  — source indices in reduced state
            pssa_{i}_src_full    — source indices in full 34-node state
            pssa_{i}_rates       — flow rates per inflow edge
            pssa_{i}_is_tissue   — whether each source is a tissue node
        """
        pssa_set = set(pssa_idx_list)
        device = topo.volumes.device

        flow_src = topo.flow_src.tolist()
        flow_tgt = topo.flow_tgt.tolist()
        flow_rates = topo.flow_rates.tolist()

        q_outs = []
        v_bps = []

        for i, bp_idx in enumerate(pssa_idx_list):
            in_src_full, in_src_reduced, in_rates, in_is_tissue = [], [], [], []

            for j in range(len(flow_src)):
                if flow_tgt[j] == bp_idx:
                    s = flow_src[j]
                    assert s not in pssa_set, (
                        f"PSSA node {topo.node_names[bp_idx]} receives flow from "
                        f"another PSSA node {topo.node_names[s]}. "
                        f"Cannot solve algebraic equations independently."
                    )
                    in_src_full.append(s)
                    in_src_reduced.append(self.full_to_reduced[s].item())
                    in_rates.append(flow_rates[j])
                    in_is_tissue.append(tissue_mask[s].item())

            q_out = sum(
                flow_rates[j] for j in range(len(flow_src))
                if flow_src[j] == bp_idx
            )
            q_outs.append(q_out)
            v_bps.append(topo.volumes[bp_idx].item())

            self.register_buffer(
                f"pssa_{i}_src_reduced",
                torch.tensor(in_src_reduced, dtype=torch.long, device=device),
            )
            self.register_buffer(
                f"pssa_{i}_src_full",
                torch.tensor(in_src_full, dtype=torch.long, device=device),
            )
            self.register_buffer(
                f"pssa_{i}_rates",
                torch.tensor(in_rates, dtype=torch.float32, device=device),
            )
            self.register_buffer(
                f"pssa_{i}_is_tissue",
                torch.tensor(in_is_tissue, dtype=torch.bool, device=device),
            )

        self.register_buffer(
            "pssa_q_out",
            torch.tensor(q_outs, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "pssa_v_bp",
            torch.tensor(v_bps, dtype=torch.float32, device=device),
        )
        self.n_pssa = len(pssa_idx_list)

    def _compute_pssa_amounts(self, y_reduced: Tensor) -> Tensor:
        """Compute algebraic steady-state amounts for the 3 PSSA nodes.

        For each blood pool bp:
            inflow = Σ Q_j × C_out_j   (flow from connected ODE nodes)
            A_bp = V_bp × inflow / Q_out_total

        C_out depends on source type:
            tissue:     A × rbp / (V × Kp)
            blood pool: A / V

        Args:
            y_reduced: [batch, 31] amounts for ODE nodes.

        Returns:
            [batch, 3] amounts for PSSA nodes, ordered by pssa_indices.
        """
        batch = y_reduced.shape[0]
        dp = self.drug_params
        kp_full = self.pbpk_func._expand_kp_to_nodes(dp.kp)  # [batch, 34]

        results = []
        for i in range(self.n_pssa):
            src_r = getattr(self, f"pssa_{i}_src_reduced")
            src_f = getattr(self, f"pssa_{i}_src_full")
            q_in = getattr(self, f"pssa_{i}_rates")
            is_tissue = getattr(self, f"pssa_{i}_is_tissue")
            n_in = len(src_r)

            a_src = y_reduced[:, src_r]           # [batch, n_in]
            v_src = self.volumes_full[src_f]       # [n_in]
            kp_src = kp_full[:, src_f]             # [batch, n_in]

            # rbp for tissue sources, 1.0 for blood pool sources
            rbp_factor = torch.where(
                is_tissue.unsqueeze(0).expand(batch, -1),
                dp.rbp.view(-1, 1).expand(batch, n_in),
                torch.ones(batch, n_in, device=y_reduced.device),
            )

            c_out = a_src * rbp_factor / (v_src.unsqueeze(0) * kp_src + 1e-30)
            total_inflow = (q_in.unsqueeze(0) * c_out).sum(dim=1)  # [batch]
            a_bp = self.pssa_v_bp[i] * total_inflow / (self.pssa_q_out[i] + 1e-30)
            results.append(a_bp)

        return torch.stack(results, dim=1)  # [batch, 3]

    def forward(self, t: Tensor, y_reduced: Tensor) -> Tensor:
        """Compute dy/dt for the reduced 31-dim state.

        Steps:
            1. Clamp ODE state to non-negative.
            2. Compute blood pool amounts algebraically.
            3. Assemble full 34-dim state via scatter.
            4. Evaluate full RHS via PBPKFunc.
            5. Return only ODE node rates.

        Args:
            t: Current time (hours).
            y_reduced: [batch, 31] ODE node amounts.

        Returns:
            [batch, 31] dy/dt for ODE nodes.
        """
        batch = y_reduced.shape[0]
        y_reduced = y_reduced.clamp(min=0.0)

        # Algebraic blood pool amounts
        pssa_amounts = self._compute_pssa_amounts(y_reduced)

        # Assemble full state (out-of-place scatter for autograd safety)
        ode_exp = self.ode_indices.unsqueeze(0).expand(batch, -1)
        pssa_exp = self.pssa_indices.unsqueeze(0).expand(batch, -1)
        y_full = torch.zeros(
            batch, self.n_full, device=y_reduced.device, dtype=y_reduced.dtype,
        )
        y_full = y_full.scatter(1, ode_exp, y_reduced)
        y_full = y_full.scatter(1, pssa_exp, pssa_amounts)

        dydt_full = self.pbpk_func(t, y_full)
        return dydt_full.gather(1, ode_exp)

    def expand_solution(self, solution_reduced: Tensor) -> Tensor:
        """Expand reduced ODE solution to full 34-node state.

        Computes algebraic blood pool amounts at each timepoint.

        Args:
            solution_reduced: [T, batch, 31] solution at evaluation times.

        Returns:
            [T, batch, 34] full state including PSSA nodes.
        """
        T, batch, _ = solution_reduced.shape
        device = solution_reduced.device

        ode_exp = self.ode_indices.unsqueeze(0).expand(batch, -1)
        pssa_exp = self.pssa_indices.unsqueeze(0).expand(batch, -1)

        slices = []
        for t_idx in range(T):
            y_t = solution_reduced[t_idx]
            pssa_t = self._compute_pssa_amounts(y_t)
            row = torch.zeros(batch, self.n_full, device=device, dtype=y_t.dtype)
            row = row.scatter(1, ode_exp, y_t)
            row = row.scatter(1, pssa_exp, pssa_t)
            slices.append(row)

        return torch.stack(slices)  # [T, batch, 34]
