"""Stage 2: PBPK ODE fine-tuning with time-concentration curves.

Trains the full QtM pipeline (encoder → projector → Neural ODE) on
284 drugs with clinical PK profiles from Sisyphus clinical_pk.json.

Key features:
- PSSA (Pseudo-Steady State Approximation) for 3 central blood pools,
  reducing ODE stiffness from λ=1482 to λ≈662 h⁻¹
- Forward sensitivity equations exploiting ODE linearity (dy/dt = M(θ)·y)
  for full 24h gradient flow without backward through solver
- rk4 fixed-step (dt=0.003h) — stable for λ_max < 927 h⁻¹
- Strict parameter clamping in Module C prevents solver blowup
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
from torchdiffeq import odeint

from src.data.graph_topology import PBPKTopology, load_topology
from src.models.encoder import SchNetEncoder
from src.models.linear_sensitivity import solve_ode_with_sensitivity
from src.models.ode_system import DrugParams, ODEWrapper, PBPKFunc, PSSAWrapper
from src.models.projector import HierarchicalProjector
from src.training.logger import RunLogger
from src.training.loss import annealed_pinn_loss

logger = logging.getLogger(__name__)


def solve_ode_windowed(
    wrapper,
    y0: Tensor,
    t_eval: Tensor,
    method: str = "dopri5",
    rtol: float = 1e-3,
    atol: float = 1e-5,
    window_hours: float = 4.0,
    options: dict | None = None,
) -> Tuple[Optional[Tensor], Optional[str]]:
    """Solve ODE using truncated BPTT with time-windowing.

    Splits the time horizon into windows. Within each window, solve with
    odeint (non-adjoint) for short-range gradient flow. Between windows,
    detach the state to limit memory and prevent backward instability.

    This avoids:
    - OOM from non-adjoint on full horizon
    - NaN from adjoint backward on stiff systems
    - Instability from fixed-step solvers

    Args:
        wrapper: ODE function wrapper (ODEWrapper or PSSAWrapper).
        y0: [batch, N] initial state (N=34 for ODEWrapper, 31 for PSSAWrapper).
        t_eval: [T] evaluation timepoints (sorted).
        method: ODE solver method.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        window_hours: Duration of each BPTT window.

    Returns:
        (solution, None) on success, (None, error_msg) on failure.
        solution: [T, batch, N] amounts at t_eval points.
    """
    try:
        device = y0.device
        t_max = t_eval[-1].item()
        batch = y0.shape[0]
        n_states = y0.shape[1]

        # Collect solutions at evaluation points
        all_solutions = [y0.unsqueeze(0)]  # t=0
        y_current = y0

        # Define windows
        window_start = 0.0
        eval_idx = 1  # skip t=0

        while window_start < t_max and eval_idx < len(t_eval):
            window_end = min(window_start + window_hours, t_max)

            # Gather eval points in this window
            window_t = [torch.tensor(window_start, device=device)]
            while eval_idx < len(t_eval) and t_eval[eval_idx].item() <= window_end + 1e-6:
                window_t.append(t_eval[eval_idx])
                eval_idx += 1
            if len(window_t) == 1:
                # No eval points in window — add endpoint
                window_t.append(torch.tensor(window_end, device=device))

            window_t = torch.stack(window_t)

            # Solve this window (with PSSA: ~800 rk4 steps or ~200 dopri5 steps per 4h)
            opts = options if options is not None else {"max_num_steps": 10000}
            window_sol = odeint(
                wrapper, y_current, window_t,
                method=method, rtol=rtol, atol=atol,
                options=opts,
            )  # [len(window_t), batch, N]

            # Check for NaN
            if torch.isnan(window_sol).any():
                return None, f"NaN in window [{window_start:.1f}, {window_end:.1f}]h"

            # Store solutions at eval points (skip the first which is window_start)
            all_solutions.append(window_sol[1:])

            # Detach state for next window (truncated BPTT)
            y_current = window_sol[-1].detach().requires_grad_(True)
            window_start = window_end

        solution = torch.cat(all_solutions, dim=0)  # [T, batch, 34]
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
    """Epoch-dependent solver configuration.

    Epoch 0-9:   rk4 fixed-step (dt=0.005h). Predictable compute graph,
                 ~7s/molecule. Stable for λ_max ≈ 143 h⁻¹ (λdt=0.72 < 2.78).
    Epoch 10+:   dopri5 adaptive (rtol=5e-2). Better accuracy as params
                 stabilize; adaptive step avoids wasted evaluations.
    """
    if epoch < 10:
        return {"method": "rk4", "options": {"step_size": 0.005}}
    return {
        "method": "dopri5",
        "rtol": 5e-2,
        "atol": 1e-3,
        "options": {"max_num_steps": 10000},
    }


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
    adme_dataloader=None,
    adme_head=None,
    adme_weight: float = 0.1,
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
    t_common = torch.linspace(0, 24, 50, device=device)

    total_loss = 0.0
    total_data = 0.0
    total_mb = 0.0
    n_batches = 0
    n_diverged = 0
    n_molecules = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        # Convert float tensors to model dtype (float64 for ODE stability)
        model_dtype = next(encoder.parameters()).dtype
        for key in ['pos', 'charges', 'kp_baseline', 'dose_mg', 'obs_times', 'obs_conc']:
            if hasattr(batch, key):
                attr = getattr(batch, key)
                if isinstance(attr, torch.Tensor) and attr.is_floating_point():
                    setattr(batch, key, attr.to(dtype=model_dtype))

        batch_size = int(batch.batch.max().item()) + 1
        n_molecules += batch_size

        # Encode
        z_mol = encoder(batch.z, batch.pos, batch.charges, batch.edge_index, batch.batch)
        drug_params = projector(z_mol, batch.kp_baseline)

        # ODE solve with PSSA + forward sensitivity equations
        # Exploits linearity of PBPK in y: dy/dt = M(θ)·y
        # Gradient via forward sensitivity S = ∂y/∂θ (no backward through solver)
        wrapper = PSSAWrapper(pbpk_func, topo, drug_params)
        y0 = torch.zeros(batch_size, wrapper.n_reduced, device=device, dtype=next(encoder.parameters()).dtype)
        y0[:, wrapper.stomach_idx_reduced] = batch.dose_mg.squeeze(-1)

        try:
            # dt=0.001 is 2.5× safety margin for max_eig ≈ 1092 across molecules
            # (2.78 / 1092 = 0.00254 rk4 boundary; use 2x smaller)
            sol = solve_ode_with_sensitivity(
                pbpk_func, topo, drug_params, y0, t_common, step_size=0.001,
            )
            err = None
        except RuntimeError as e:
            sol = None
            err = str(e)

        if sol is None or torch.isnan(sol).any():
            n_diverged += batch_size
            for i in range(batch_size):
                run_logger.log_divergence(epoch, batch.mol_id[i], err or "NaN in solution")
            continue

        # Expand to full 34-node state for venous concentration + mass balance
        sol_full = wrapper.expand_solution(sol)
        # C_plasma = C_blood / rbp (clinical PK data reports plasma concentration)
        venous_conc = sol_full[:, :, topo.venous_idx] / (
            volumes[topo.venous_idx] * drug_params.rbp.unsqueeze(0)
        )

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
                solution=sol_full[:, i:i+1, :],
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

        # Gradient accumulation with NaN guard
        scaled_loss = batch_loss / accum_steps
        scaled_loss.backward()

        # Check for NaN gradients (float32 overflow in backward through ODE)
        all_params = list(encoder.parameters()) + list(projector.parameters())
        has_nan_grad = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in all_params
        )
        if has_nan_grad:
            # Discard corrupted gradients — do NOT step optimizer
            optimizer.zero_grad()
            n_diverged += batch_size
            continue

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += batch_loss.item()
        total_data += batch_data.item()
        total_mb += batch_mb.item()
        n_batches += 1

    # Final optimizer step for remaining gradients
    if n_batches % accum_steps != 0:
        all_params = list(encoder.parameters()) + list(projector.parameters())
        has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in all_params)
        if not has_nan:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
            optimizer.step()
        optimizer.zero_grad()

    # ADME multi-task pass (if enabled): 13× more molecules for encoder gradient
    adme_loss_val = 0.0
    if adme_dataloader is not None and adme_head is not None:
        from src.training.loss import adme_multitask_loss
        adme_head.train()
        for adme_batch in adme_dataloader:
            adme_batch = adme_batch.to(device)
            model_dtype = next(encoder.parameters()).dtype
            for key in ['pos', 'charges']:
                if hasattr(adme_batch, key):
                    attr = getattr(adme_batch, key)
                    if isinstance(attr, torch.Tensor) and attr.is_floating_point():
                        setattr(adme_batch, key, attr.to(dtype=model_dtype))

            z_mol = encoder(adme_batch.z, adme_batch.pos, adme_batch.charges,
                           adme_batch.edge_index, adme_batch.batch)
            pred = adme_head(z_mol)
            task_mask = ~torch.isnan(adme_batch.y)
            loss_dict = adme_multitask_loss(pred, adme_batch.y.to(dtype=model_dtype), task_mask)
            adme_loss = loss_dict["total"] * adme_weight
            adme_loss.backward()
            adme_loss_val += loss_dict["total"].item()

        all_params = list(encoder.parameters()) + list(projector.parameters())
        if adme_head is not None:
            all_params += list(adme_head.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
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
        "adme_loss": adme_loss_val,
    }

    return metrics
