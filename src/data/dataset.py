"""PyTorch datasets for ADME multi-task (Stage 1) and PK curve (Stage 2) training.

Both datasets read 3D molecular features from HDF5 and construct PyG Data
objects with edge_index built from distance cutoff (no torch-cluster needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset

from .normalization import NormStats


def _build_radius_graph(pos: Tensor, cutoff: float = 10.0) -> Tensor:
    """Build edge_index from pairwise distances within cutoff.

    Replaces torch_geometric.nn.radius_graph (requires torch-cluster).

    Args:
        pos: [N, 3] atom positions.
        cutoff: Distance cutoff in Ångström.

    Returns:
        edge_index: [2, E] long tensor (directed, no self-loops).
    """
    dist = torch.cdist(pos, pos)  # [N, N]
    mask = (dist < cutoff) & (dist > 0)  # exclude self-loops
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # [2, E]
    return edge_index


class MolecularHDF5Dataset(Dataset):
    """Base dataset reading 3D molecular features from HDF5.

    Each sample becomes a PyG Data object:
        - z: [N] int, atomic numbers
        - pos: [N, 3] float, 3D coordinates
        - charges: [N] float, partial charges
        - edge_index: [2, E] from distance cutoff
        - mol_id: string identifier
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        mol_ids: List[str],
        cutoff: float = 10.0,
        norm_stats: Optional[NormStats] = None,
    ):
        super().__init__()
        self.hdf5_path = str(hdf5_path)
        self.mol_ids = mol_ids
        self.cutoff = cutoff
        self.norm_stats = norm_stats
        self._hdf5_file: Optional[h5py.File] = None

    def _get_hf(self) -> h5py.File:
        """Lazy-open HDF5 (per-worker for DataLoader num_workers > 0)."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def len(self) -> int:
        return len(self.mol_ids)

    def get(self, idx: int) -> Data:
        hf = self._get_hf()
        mol_id = self.mol_ids[idx]
        grp = hf[mol_id]

        coords = torch.tensor(grp["coords"][:], dtype=torch.float32)
        atomic_nums = torch.tensor(grp["atomic_nums"][:], dtype=torch.long)
        charges = torch.tensor(grp["charges"][:], dtype=torch.float32)

        # Normalize charges if stats provided
        if self.norm_stats and "charges" in self.norm_stats.feature_means:
            charges = self.norm_stats.normalize(charges, "charges")

        edge_index = _build_radius_graph(coords, self.cutoff)

        return Data(
            z=atomic_nums,
            pos=coords,
            charges=charges,
            edge_index=edge_index,
            mol_id=mol_id,
        )

    def close(self) -> None:
        if self._hdf5_file is not None:
            self._hdf5_file.close()
            self._hdf5_file = None


class ADMEDataset(MolecularHDF5Dataset):
    """Stage 1: Multi-task ADME prediction dataset.

    Adds ADME labels (fup, CLint, Vdss, etc.) as target tensors.
    Missing labels are NaN — masked in loss computation.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        mol_ids: List[str],
        adme_labels: Dict[str, Dict[str, float]],
        cutoff: float = 10.0,
        norm_stats: Optional[NormStats] = None,
        target_names: Optional[List[str]] = None,
    ):
        """
        Args:
            adme_labels: {mol_id: {property_name: value}}.
                         Missing properties should be absent, not NaN.
            target_names: Ordered list of ADME property names to predict.
        """
        super().__init__(hdf5_path, mol_ids, cutoff, norm_stats)
        self.adme_labels = adme_labels
        self.target_names = target_names or [
            "log_clint", "logit_fup", "log_vdss", "log_papp", "log_thalf",
        ]

    def get(self, idx: int) -> Data:
        data = super().get(idx)
        mol_id = self.mol_ids[idx]
        labels = self.adme_labels.get(mol_id, {})

        # Build target vector — NaN for missing
        # Store as [1, K] so PyG collates to [batch, K] not [batch*K]
        targets = torch.full((1, len(self.target_names)), float("nan"), dtype=torch.float32)
        for i, name in enumerate(self.target_names):
            if name in labels:
                targets[0, i] = labels[name]

        data.y = targets
        data.target_mask = ~torch.isnan(targets)
        return data


class PKCurveDataset(MolecularHDF5Dataset):
    """Stage 2: PK time-concentration curve dataset for ODE training.

    Each sample includes:
        - Molecular features (from parent)
        - dose_mg: dosing amount
        - obs_times: [T] observed timepoints (hours)
        - obs_conc: [T] observed concentrations (mg/L)
        - kp_baseline: [15] pre-computed Kp baselines (Rodgers & Rowland)
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        mol_ids: List[str],
        pk_data: Dict[str, dict],
        kp_baselines: Dict[str, Tensor],
        cutoff: float = 10.0,
        norm_stats: Optional[NormStats] = None,
    ):
        """
        Args:
            pk_data: {mol_id: {"dose_mg": float, "time_h": list, "conc_mg_L": list}}.
            kp_baselines: {mol_id: Tensor[15]} pre-computed Kp baseline values.
        """
        super().__init__(hdf5_path, mol_ids, cutoff, norm_stats)
        self.pk_data = pk_data
        self.kp_baselines = kp_baselines

    def get(self, idx: int) -> Data:
        data = super().get(idx)
        mol_id = self.mol_ids[idx]
        pk = self.pk_data[mol_id]

        dose = pk.get("dose_mg") or 100.0  # default 100mg if missing
        data.dose_mg = torch.tensor([float(dose)], dtype=torch.float32)
        # Pad to 50 timepoints for uniform batching
        max_t = 50
        times = pk["time_h"][:max_t]
        conc = pk["conc_mg_L"][:max_t]
        n_obs = len(times)
        # Pad with zeros + mask
        times_padded = times + [0.0] * (max_t - n_obs)
        conc_padded = conc + [0.0] * (max_t - n_obs)
        data.obs_times = torch.tensor(times_padded, dtype=torch.float32).unsqueeze(0)  # [1, 50]
        data.obs_conc = torch.tensor(conc_padded, dtype=torch.float32).unsqueeze(0)    # [1, 50]
        data.obs_mask = torch.tensor(
            [True] * n_obs + [False] * (max_t - n_obs), dtype=torch.bool
        ).unsqueeze(0)  # [1, 50]
        kp = self.kp_baselines.get(mol_id, torch.ones(15, dtype=torch.float32))
        data.kp_baseline = kp.unsqueeze(0) if kp.dim() == 1 else kp  # [1, 15] for PyG
        return data
