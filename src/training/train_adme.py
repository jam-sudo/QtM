"""Stage 1: ADME multi-task pretraining.

Trains the SchNet encoder on ADME property prediction without ODE complexity.
This validates the encoder architecture and transfers molecular knowledge
before Stage 2 ODE fine-tuning.

Training data: CLint (~1,900), fup/PPB (~2,800), Vdss (~900), Caco-2 (~900)
Targets are log-transformed for numerical stability.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.models.encoder import SchNetEncoder
from src.training.logger import RunLogger
from src.training.loss import adme_multitask_loss

logger = logging.getLogger(__name__)


class ADMEHead(nn.Module):
    """Multi-task ADME prediction head.

    Takes Z_mol and outputs K ADME property predictions.
    Missing labels are masked in loss (no gradient contribution).
    """

    def __init__(self, z_dim: int = 128, n_tasks: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_tasks),
        )

    def forward(self, z_mol: Tensor) -> Tensor:
        """Predict K ADME properties.

        Args:
            z_mol: [batch, z_dim] molecular latent.

        Returns:
            [batch, K] predicted ADME values (log-scale).
        """
        return self.head(z_mol)


def train_adme_epoch(
    encoder: SchNetEncoder,
    adme_head: ADMEHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train one epoch of ADME pretraining.

    Args:
        encoder: SchNet encoder.
        adme_head: Multi-task prediction head.
        dataloader: Yields ADMEDataset samples.
        optimizer: Optimizer for encoder + head parameters.
        device: CUDA device.

    Returns:
        Dict of epoch metrics.
    """
    encoder.train()
    adme_head.train()

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)

        z_mol = encoder(batch.z, batch.pos, batch.charges, batch.edge_index, batch.batch)
        pred = adme_head(z_mol)

        loss_dict = adme_multitask_loss(pred, batch.y, batch.target_mask)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(adme_head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
    }


@torch.no_grad()
def eval_adme(
    encoder: SchNetEncoder,
    adme_head: ADMEHead,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate ADME predictions on validation set.

    Returns per-task RMSE and R² where sufficient data exists.
    """
    encoder.eval()
    adme_head.eval()

    all_preds = []
    all_targets = []
    all_masks = []

    for batch in dataloader:
        batch = batch.to(device)
        z_mol = encoder(batch.z, batch.pos, batch.charges, batch.edge_index, batch.batch)
        pred = adme_head(z_mol)
        all_preds.append(pred.cpu())
        all_targets.append(batch.y.cpu())
        all_masks.append(batch.target_mask.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    metrics = {}
    n_tasks = preds.shape[1]
    task_names = ["log_clint", "logit_fup", "log_vdss", "log_papp", "log_thalf"]

    for i in range(n_tasks):
        valid = masks[:, i]
        if valid.sum() < 10:
            continue
        p = preds[valid, i]
        t = targets[valid, i]

        rmse = ((p - t) ** 2).mean().sqrt().item()
        ss_res = ((p - t) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

        name = task_names[i] if i < len(task_names) else f"task_{i}"
        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_r2"] = r2

    return metrics
