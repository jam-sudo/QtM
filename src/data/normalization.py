"""Train-only normalization firewall (CLAUDE.md §6.2).

All feature statistics (mean, std) are computed exclusively from the
training set. These frozen statistics are then applied to val/test sets.
Computing statistics from the full dataset is PROHIBITED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class NormStats:
    """Frozen normalization statistics computed from training data only.

    Attributes:
        feature_means: {feature_name: mean_value}
        feature_stds: {feature_name: std_value} (floored at 1e-8)
        computed_from: "train" — always. Enforced by construction.
        n_samples: Number of training samples used to compute stats.
    """
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]
    computed_from: str = "train"
    n_samples: int = 0

    def normalize(self, values: Tensor, feature_name: str) -> Tensor:
        """Normalize values using stored train statistics.

        Args:
            values: Raw feature tensor.
            feature_name: Key into feature_means/feature_stds.

        Returns:
            Standardized tensor: (values - mean) / std.
        """
        mean = self.feature_means[feature_name]
        std = self.feature_stds[feature_name]
        return (values - mean) / std

    def denormalize(self, values: Tensor, feature_name: str) -> Tensor:
        """Reverse normalization."""
        mean = self.feature_means[feature_name]
        std = self.feature_stds[feature_name]
        return values * std + mean

    def save(self, path: str | Path) -> None:
        """Save to JSON for reproducibility."""
        data = {
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "computed_from": self.computed_from,
            "n_samples": self.n_samples,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> NormStats:
        """Load from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def compute_norm_stats(
    train_features: Dict[str, np.ndarray],
    min_std: float = 1e-8,
) -> NormStats:
    """Compute normalization statistics from training features ONLY.

    Args:
        train_features: {feature_name: array_of_values} from TRAINING set.
        min_std: Floor for standard deviation to prevent division by zero.

    Returns:
        NormStats with frozen mean/std per feature.
    """
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    n_samples = 0

    for name, values in train_features.items():
        arr = np.asarray(values, dtype=np.float64)
        # Flatten if multi-dimensional (e.g., per-atom charges)
        flat = arr.flatten()
        means[name] = float(np.nanmean(flat))
        stds[name] = float(max(np.nanstd(flat), min_std))
        n_samples = max(n_samples, len(arr))

    return NormStats(
        feature_means=means,
        feature_stds=stds,
        computed_from="train",
        n_samples=n_samples,
    )
