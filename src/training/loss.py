"""Annealed PINN loss function (CLAUDE.md §3 Module D).

Total_Loss = MSLE_PK_Data + λ(epoch) × Mass_Balance_Penalty

The mass balance penalty is annealed from 0 to λ_max over a warmup period,
allowing the network to first fit PK data before enforcing physics constraints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def msle_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean Squared Logarithmic Error.

    MSLE = mean((log(1 + pred) - log(1 + target))^2)

    Handles zero concentrations via log1p. Masked entries are excluded.

    Args:
        pred: [T, batch] or [batch, K] predicted values.
        target: Same shape as pred, observed values.
        mask: Same shape, bool. True = valid, False = ignore.

    Returns:
        Scalar loss.
    """
    log_pred = torch.log1p(pred.clamp(min=0))
    log_target = torch.log1p(target.clamp(min=0))
    sq_err = (log_pred - log_target) ** 2

    if mask is not None:
        sq_err = sq_err * mask.float()
        return sq_err.sum() / (mask.float().sum() + 1e-8)
    return sq_err.mean()


def mass_balance_penalty(
    solution: Tensor,
    dose_mg: Tensor,
) -> Tensor:
    """Compute mass balance violation across all timepoints.

    Total mass in the system should equal the dose at all times
    (conservation of mass — drug is redistributed, not created/destroyed).

    Args:
        solution: [T, batch, 34] ODE solution (amounts in mg).
        dose_mg: [batch] original dose.

    Returns:
        Scalar penalty (MSE of relative violation).
    """
    total_mass = solution.sum(dim=-1)  # [T, batch]
    relative_error = (total_mass - dose_mg.unsqueeze(0)) / (dose_mg.unsqueeze(0) + 1e-8)
    return (relative_error ** 2).mean()


def annealed_pinn_loss(
    pred_conc: Tensor,
    obs_conc: Tensor,
    solution: Tensor,
    dose_mg: Tensor,
    epoch: int,
    lambda_max: float = 1.0,
    warmup_epochs: int = 20,
    conc_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute the full annealed PINN loss.

    Args:
        pred_conc: [T, batch] predicted venous concentrations (mg/L).
        obs_conc: [T, batch] observed concentrations (mg/L).
        solution: [T, batch, 34] full ODE solution (amounts).
        dose_mg: [batch] dose per molecule.
        epoch: Current training epoch (for annealing).
        lambda_max: Maximum mass balance penalty weight.
        warmup_epochs: Epochs over which to anneal λ from 0 to λ_max.
        conc_mask: [T, batch] bool mask for valid observations.

    Returns:
        Dict with "total", "data", "mass_balance", "lambda" keys.
    """
    # Data loss (MSLE on PK concentrations)
    data_loss = msle_loss(pred_conc, obs_conc, mask=conc_mask)

    # Mass balance penalty
    mb_penalty = mass_balance_penalty(solution, dose_mg)

    # Annealing schedule: linear ramp
    lambda_epoch = min(lambda_max, lambda_max * epoch / max(warmup_epochs, 1))

    total = data_loss + lambda_epoch * mb_penalty

    return {
        "total": total,
        "data": data_loss.detach(),
        "mass_balance": mb_penalty.detach(),
        "lambda": torch.tensor(lambda_epoch),
    }


def adme_multitask_loss(
    pred: Tensor,
    target: Tensor,
    task_mask: Tensor,
) -> dict[str, Tensor]:
    """Multi-task ADME loss with missing label masking.

    Args:
        pred: [batch, K] predicted ADME values (K tasks).
        target: [batch, K] observed values (NaN for missing).
        task_mask: [batch, K] bool, True where target is valid.

    Returns:
        Dict with "total" and per-task losses.
    """
    # Replace NaN with 0 in target to avoid NaN * 0 = NaN
    safe_target = torch.where(task_mask, target, torch.zeros_like(target))

    # Per-task MSE with masking
    sq_err = (pred - safe_target) ** 2 * task_mask.float()

    # Per-task mean
    counts = task_mask.float().sum(dim=0).clamp(min=1)
    per_task = sq_err.sum(dim=0) / counts  # [K]

    # Equal weight across tasks (only tasks with data)
    active = counts > 0
    total = per_task[active].mean() if active.any() else pred.sum() * 0

    return {
        "total": total,
        "per_task": per_task.detach(),
    }
