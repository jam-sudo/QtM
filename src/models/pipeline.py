"""End-to-end pipeline: SchNet → Projector → ODE solve.

Wires Module B (encoder), Module C (projector), and Module D (ODE system)
into a single differentiable forward pass. Uses odeint_adjoint with explicit
adjoint_params for correct gradient backpropagation through the ODE solver.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint_adjoint

from src.data.graph_topology import PBPKTopology
from src.models.encoder import SchNetEncoder
from src.models.ode_system import DrugParams, ODEWrapper, PBPKFunc
from src.models.projector import HierarchicalProjector


class QtMPipeline(nn.Module):
    """Full QtM pipeline: SMILES features → PK concentration curves.

    Forward pass:
        1. SchNet encoder: (z, pos, charges, edge_index, batch) → Z_mol
        2. Hierarchical projector: (Z_mol, Kp_baseline) → DrugParams
        3. ODE solve: (y0, DrugParams) → concentration time series

    The ODE is solved with odeint_adjoint for O(1) VRAM backpropagation.
    """

    def __init__(
        self,
        topo: PBPKTopology,
        encoder: SchNetEncoder,
        projector: HierarchicalProjector,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.pbpk_func = PBPKFunc(topo)

        self.stomach_idx = topo.stomach_idx
        self.venous_idx = topo.venous_idx
        self.n_nodes = topo.n_nodes

        # Store volumes for concentration conversion
        self.register_buffer("volumes", topo.volumes)

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        charges: Tensor,
        edge_index: Tensor,
        batch_idx: Tensor,
        kp_baseline: Tensor,
        dose_mg: Tensor,
        t_eval: Tensor,
        solver_method: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ) -> Tuple[Tensor, DrugParams]:
        """Full forward pass: molecules → PK curves.

        Args:
            z: [N_total] atomic numbers.
            pos: [N_total, 3] 3D coordinates.
            charges: [N_total] partial charges.
            edge_index: [2, E] edges.
            batch_idx: [N_total] molecule assignment per atom.
            kp_baseline: [batch, 15] Rodgers & Rowland baselines.
            dose_mg: [batch] dose per molecule (mg).
            t_eval: [T] time points to evaluate (hours).
            solver_method: ODE solver ("dopri5" or "implicit_adams").
            rtol: Relative tolerance for ODE solver.
            atol: Absolute tolerance for ODE solver.

        Returns:
            venous_conc: [T, batch] venous blood concentration (mg/L).
            drug_params: DrugParams for inspection/logging.
        """
        batch_size = int(batch_idx.max().item()) + 1

        # 1. Encode molecules
        z_mol = self.encoder(z, pos, charges, edge_index, batch_idx)  # [batch, 128]

        # 2. Project to PBPK parameters
        drug_params = self.projector(z_mol, kp_baseline)

        # 3. Initial conditions: dose in stomach
        y0 = torch.zeros(batch_size, self.n_nodes, device=z.device, dtype=torch.float32)
        y0[:, self.stomach_idx] = dose_mg

        # 4. Solve ODE with adjoint method
        wrapper = ODEWrapper(self.pbpk_func, drug_params)

        # Collect all trainable params + drug_param tensors for adjoint
        adjoint_params = (
            tuple(self.encoder.parameters())
            + tuple(self.projector.parameters())
            + (drug_params.kp, drug_params.enzyme_affinities, drug_params.ps,
               drug_params.fup, drug_params.rbp, drug_params.peff,
               drug_params.renal_cl, drug_params.particle_radius)
        )

        solution = odeint_adjoint(
            wrapper, y0, t_eval,
            method=solver_method,
            rtol=rtol,
            atol=atol,
            adjoint_params=adjoint_params,
        )  # [T, batch, 34]

        # 5. Extract venous blood concentration
        venous_amounts = solution[:, :, self.venous_idx]  # [T, batch]
        venous_conc = venous_amounts / self.volumes[self.venous_idx]  # [T, batch] mg/L

        return venous_conc, drug_params

    def compute_mass_balance(
        self, solution: Tensor, dose_mg: Tensor
    ) -> Tensor:
        """Compute mass balance error for PINN loss.

        Args:
            solution: [T, batch, 34] ODE solution (amounts).
            dose_mg: [batch] original dose.

        Returns:
            error: [T, batch] relative mass balance error.
        """
        total_mass = solution.sum(dim=-1)  # [T, batch]
        return (total_mass - dose_mg.unsqueeze(0)) / (dose_mg.unsqueeze(0) + 1e-30)
