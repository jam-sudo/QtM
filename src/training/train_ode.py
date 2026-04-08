"""Stage 2: PBPK ODE fine-tuning with time-concentration curves.

Trains the full QtM pipeline (encoder → projector → Neural ODE) on
284 drugs with clinical PK profiles from Sisyphus clinical_pk.json.

Key features:
- Adaptive solver transition: implicit_adams → dopri5
- odeint_adjoint for O(1) VRAM backpropagation
- Annealed PINN loss (data MSLE + mass balance)
- Solver divergence logging (§6.1)
- Gradient accumulation for effective larger batches
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchdiffeq import odeint, odeint_adjoint

from src.data.graph_topology import PBPKTopology, load_topology
from src.models.encoder import SchNetEncoder
from src.models.ode_system import DrugParams, ODEWrapper, PBPKFunc
from src.models.projector import HierarchicalProjector
from src.training.logger import RunLogger
from src.training.loss import annealed_pinn_loss

logger = logging.getLogger(__name__)


def solve_ode_safe(
    wrapper: ODEWrapper,
    y0: Tensor,
    t_eval: Tensor,
    method: str,
    rtol: float,
    atol: float,
    adjoint_params: tuple,
    use_adjoint: bool = True,
) -> Tuple[Optional[Tensor], Optional[str]]:
    """Solve ODE with error handling for divergence.

    Returns (solution, None) on success, (None, error_msg) on failure.
    Divergent molecules are NEVER silently dropped (§6.1).

    Args:
        use_adjoint: If True, use odeint_adjoint (O(1) VRAM, slower).
                     If False, use odeint (stores all states, faster for small batches).
    """
    try:
        opts = {"max_num_steps": 10000}
        if use_adjoint:
            solution = odeint_adjoint(
                wrapper, y0, t_eval,
                method=method, rtol=rtol, atol=atol,
                adjoint_params=adjoint_params,
                options=opts, adjoint_options=opts,
            )
        else:
            solution = odeint(
                wrapper, y0, t_eval,
                method=method, rtol=rtol, atol=atol,
                options=opts,
            )
        # Check for NaN
        if torch.isnan(solution).any():
            return None, "NaN in ODE solution"
        return solution, None
    except RuntimeError as e:
        return None, str(e)


def interpolate_to_obs(
    solution_conc: Tensor,
    t_common: Tensor,
    obs_times: Tensor,
) -> Tensor:
    """Interpolate solution at common grid to observation timepoints.

    Uses linear interpolation between grid points.

    Args:
        solution_conc: [T_common, batch] concentrations at common grid.
        t_common: [T_common] common time grid.
        obs_times: [T_obs] observation times for one molecule.

    Returns:
        [T_obs] interpolated concentrations.
    """
    # Simple linear interpolation
    indices = torch.searchsorted(t_common, obs_times.clamp(max=t_common[-1]))
    indices = indices.clamp(1, len(t_common) - 1)

    t_lo = t_common[indices - 1]
    t_hi = t_common[indices]
    c_lo = solution_conc[indices - 1]
    c_hi = solution_conc[indices]

    frac = (obs_times - t_lo) / (t_hi - t_lo + 1e-8)
    return c_lo + frac * (c_hi - c_lo)


def get_solver_config(epoch: int) -> dict:
    """Adaptive solver transition strategy.

    Epoch 0-5:   dopri5 with very loose tolerances (warm-up)
    Epoch 6-20:  dopri5 with moderate tolerances
    Epoch 21+:   dopri5 with tight tolerances

    Note: implicit_adams in torchdiffeq has functional iteration convergence
    issues. Using dopri5 throughout with tolerance annealing instead.
    """
    if epoch < 5:
        return {"method": "dopri5", "rtol": 1e-2, "atol": 1e-3}
    elif epoch < 20:
        return {"method": "dopri5", "rtol": 1e-3, "atol": 1e-5}
    return {"method": "dopri5", "rtol": 1e-4, "atol": 1e-6}


def train_one_epoch(
    pipeline_modules: dict,
    dataloader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    run_logger: RunLogger,
    accum_steps: int = 4,
    lambda_max: float = 1.0,
    warmup_epochs: int = 20,
    use_adjoint: bool = True,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        pipeline_modules: dict with 'encoder', 'projector', 'pbpk_func',
                          'topo', 'volumes' keys.
        dataloader: Yields batches of PKCurveDataset samples.
        optimizer: The optimizer.
        epoch: Current epoch number.
        device: CUDA device.
        run_logger: For logging divergences.
        accum_steps: Gradient accumulation steps.
        lambda_max: Max mass balance penalty weight.
        warmup_epochs: Annealing warmup epochs.

    Returns:
        Dict of epoch metrics.
    """
    encoder = pipeline_modules["encoder"]
    projector = pipeline_modules["projector"]
    pbpk_func = pipeline_modules["pbpk_func"]
    topo = pipeline_modules["topo"]
    volumes = pipeline_modules["volumes"]

    encoder.train()
    projector.train()

    solver_cfg = get_solver_config(epoch)
    t_common = torch.linspace(0, 48, 100, device=device)

    total_loss = 0.0
    total_data = 0.0
    total_mb = 0.0
    n_batches = 0
    n_diverged = 0
    n_molecules = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        batch_size = int(batch.batch.max().item()) + 1
        n_molecules += batch_size

        # Encode
        z_mol = encoder(batch.z, batch.pos, batch.charges, batch.edge_index, batch.batch)
        drug_params = projector(z_mol, batch.kp_baseline)

        # ODE solve
        y0 = torch.zeros(batch_size, topo.n_nodes, device=device)
        y0[:, topo.stomach_idx] = batch.dose_mg.squeeze(-1)

        wrapper = ODEWrapper(pbpk_func, drug_params)
        adjoint_params = (
            tuple(encoder.parameters()) + tuple(projector.parameters())
            + (drug_params.kp, drug_params.enzyme_affinities, drug_params.ps,
               drug_params.fup, drug_params.rbp, drug_params.peff,
               drug_params.renal_cl, drug_params.particle_radius)
        )

        sol, err = solve_ode_safe(
            wrapper, y0, t_common,
            method=solver_cfg["method"],
            rtol=solver_cfg["rtol"],
            atol=solver_cfg["atol"],
            adjoint_params=adjoint_params,
            use_adjoint=use_adjoint,
        )

        if sol is None:
            n_diverged += batch_size
            for i in range(batch_size):
                run_logger.log_divergence(epoch, batch.mol_id[i], err)
            continue

        # Extract venous concentrations and interpolate
        venous_conc = sol[:, :, topo.venous_idx] / volumes[topo.venous_idx]

        # Per-molecule interpolation to observation times
        losses = []
        for i in range(batch_size):
            obs_t = batch.obs_times[i] if hasattr(batch, 'obs_times') else None
            obs_c = batch.obs_conc[i] if hasattr(batch, 'obs_conc') else None
            if obs_t is None or obs_c is None:
                continue

            pred_at_obs = interpolate_to_obs(venous_conc[:, i], t_common, obs_t)

            # pred_at_obs: [T_obs], obs_c: [T_obs] → reshape to [T_obs, 1] for loss
            loss_dict = annealed_pinn_loss(
                pred_conc=pred_at_obs.unsqueeze(-1),
                obs_conc=obs_c.unsqueeze(-1),
                solution=sol[:, i:i+1, :],
                dose_mg=batch.dose_mg[i:i+1].squeeze(-1),
                epoch=epoch,
                lambda_max=lambda_max,
                warmup_epochs=warmup_epochs,
            )
            losses.append(loss_dict)

        if not losses:
            continue

        batch_loss = torch.stack([l["total"] for l in losses]).mean()
        batch_data = torch.stack([l["data"] for l in losses]).mean()
        batch_mb = torch.stack([l["mass_balance"] for l in losses]).mean()

        # Gradient accumulation
        scaled_loss = batch_loss / accum_steps
        scaled_loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projector.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += batch_loss.item()
        total_data += batch_data.item()
        total_mb += batch_mb.item()
        n_batches += 1

    # Final optimizer step for remaining gradients
    if n_batches % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(projector.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        optimizer.zero_grad()

    divergence_rate = n_diverged / max(n_molecules, 1)
    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "data_loss": total_data / max(n_batches, 1),
        "mass_balance": total_mb / max(n_batches, 1),
        "divergence_rate": divergence_rate,
        "n_molecules": n_molecules,
        "n_diverged": n_diverged,
        "solver": solver_cfg["method"],
    }

    return metrics
