"""Murcko scaffold-aware train/val/test splitting.

Molecules sharing the same Murcko scaffold are assigned to the same split,
preventing structural leakage across partitions (CLAUDE.md §6.2).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _get_scaffold(smiles: str) -> str:
    """Return the canonical Murcko scaffold SMILES for a molecule.

    Falls back to the SMILES itself if scaffold extraction fails
    (e.g., acyclic molecules have no ring scaffold).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return smiles


def create_scaffold_split(
    smiles_list: List[str],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    holdout_smiles: Optional[List[str]] = None,
) -> Dict[str, List[int]]:
    """Split molecules by Murcko scaffold into train/val/test sets.

    Entire scaffold groups are assigned to one split — no scaffold appears
    in more than one partition.

    Args:
        smiles_list: List of SMILES strings to split.
        ratios: (train, val, test) fractions summing to 1.0.
        seed: Random seed for reproducibility.
        holdout_smiles: Optional pre-defined holdout set (forced into test).

    Returns:
        Dict with keys "train", "val", "test", each mapping to a list of
        indices into smiles_list.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"
    n = len(smiles_list)
    rng = np.random.RandomState(seed)

    # Group molecule indices by scaffold
    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = _get_scaffold(smi)
        scaffold_to_indices[scaf].append(i)

    # Pre-assign holdout molecules to test if provided
    holdout_set = set()
    forced_test_indices: List[int] = []
    if holdout_smiles:
        holdout_set = {Chem.MolToSmiles(Chem.MolFromSmiles(s))
                       for s in holdout_smiles if Chem.MolFromSmiles(s) is not None}
        # Find scaffolds containing holdout molecules
        holdout_scaffolds = set()
        for scaf, indices in scaffold_to_indices.items():
            for idx in indices:
                canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_list[idx]))
                if canon and canon in holdout_set:
                    holdout_scaffolds.add(scaf)
                    break

        # Move holdout scaffolds to test
        for scaf in holdout_scaffolds:
            forced_test_indices.extend(scaffold_to_indices.pop(scaf))

    # Sort scaffold groups by size (largest first) for greedy assignment
    scaffold_groups = list(scaffold_to_indices.values())
    rng.shuffle(scaffold_groups)
    scaffold_groups.sort(key=len, reverse=True)

    # Greedy assignment: place each scaffold group into the split
    # that is furthest below its target count
    target_train = int(n * ratios[0])
    target_val = int(n * ratios[1])

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = list(forced_test_indices)

    for group in scaffold_groups:
        # Pick the split most below its target
        train_deficit = target_train - len(train_indices)
        val_deficit = target_val - len(val_indices)
        test_target = n - target_train - target_val
        test_deficit = test_target - len(test_indices)

        deficits = [train_deficit, val_deficit, test_deficit]
        best = int(np.argmax(deficits))

        if best == 0:
            train_indices.extend(group)
        elif best == 1:
            val_indices.extend(group)
        else:
            test_indices.extend(group)

    # Sort indices within each split for determinism
    result = {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
    }

    # Verify no overlap
    all_assigned = set(result["train"]) | set(result["val"]) | set(result["test"])
    assert len(all_assigned) == n, f"Lost molecules: {n - len(all_assigned)}"
    assert len(result["train"]) + len(result["val"]) + len(result["test"]) == n

    return result
