"""Module A: Generate 3D molecular features from SMILES and store in HDF5.

Phase 1 uses RDKit ETKDGv3 for 3D conformer generation and Gasteiger
partial charges. xTB GFN2 features (better charges, HOMO-LUMO gap, dipole)
will be added when xtb-python becomes available.

Output HDF5 schema per molecule:
    /{mol_id}/
        coords:          float32 [N_atoms, 3]   # Ångström
        atomic_nums:     int8    [N_atoms]
        charges:         float32 [N_atoms]       # Gasteiger (Phase 1) / xTB (Phase 2)
        smiles:          string attribute
        n_atoms:         int attribute
        charge_method:   string attribute        # "gasteiger" or "xtb_gfn2"
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger(__name__)


@dataclass
class MolFeatures:
    """Container for computed 3D molecular features."""
    mol_id: str
    smiles: str
    coords: np.ndarray        # [N, 3] float32
    atomic_nums: np.ndarray   # [N] int8
    charges: np.ndarray       # [N] float32
    charge_method: str        # "gasteiger" or "xtb_gfn2"


@dataclass
class FailedMol:
    """Record of a molecule that failed feature generation."""
    mol_id: str
    smiles: str
    error: str


def _compute_features_single(args: Tuple[str, str]) -> MolFeatures | FailedMol:
    """Compute 3D features for a single molecule. Runs in worker process.

    Args:
        args: (mol_id, smiles) tuple.

    Returns:
        MolFeatures on success, FailedMol on failure.
    """
    mol_id, smiles = args

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return FailedMol(mol_id, smiles, "RDKit parse failure")

        # Add hydrogens for 3D embedding, but store heavy atoms only
        mol_h = Chem.AddHs(mol)

        # ETKDGv3 conformer generation — try up to 5 conformers, keep lowest energy
        params = AllChem.ETKDGv3()
        params.numThreads = 1  # CRITICAL: prevent thread contention in multiprocessing
        params.randomSeed = 42
        params.maxAttempts = 5

        cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=5, params=params)
        if len(cids) == 0:
            return FailedMol(mol_id, smiles, "ETKDGv3 embedding failed (0 conformers)")

        # MMFF94 optimization — pick lowest energy conformer
        results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=1, maxIters=500)
        # results: list of (converged, energy) tuples
        energies = []
        for i, (converged, energy) in enumerate(results):
            if converged == 0:  # 0 = converged, 1 = not converged
                energies.append((energy, i))
            else:
                energies.append((energy, i))  # Use even if not converged

        if not energies:
            return FailedMol(mol_id, smiles, "MMFF optimization failed for all conformers")

        best_energy, best_cid = min(energies, key=lambda x: x[0])

        # Extract heavy-atom coordinates from best conformer
        conf = mol_h.GetConformer(best_cid)
        heavy_indices = [i for i, a in enumerate(mol_h.GetAtoms()) if a.GetAtomicNum() > 1]
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in heavy_indices],
            dtype=np.float32
        )
        atomic_nums = np.array(
            [mol_h.GetAtomWithIdx(i).GetAtomicNum() for i in heavy_indices],
            dtype=np.int8
        )

        # Gasteiger partial charges on heavy atoms
        AllChem.ComputeGasteigerCharges(mol_h)
        charges = np.array(
            [float(mol_h.GetAtomWithIdx(i).GetProp("_GasteigerCharge")) for i in heavy_indices],
            dtype=np.float32
        )
        # Replace NaN/Inf charges with 0
        charges = np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0)

        return MolFeatures(
            mol_id=mol_id,
            smiles=smiles,
            coords=coords,
            atomic_nums=atomic_nums,
            charges=charges,
            charge_method="gasteiger",
        )

    except Exception as e:
        return FailedMol(mol_id, smiles, str(e))


def generate_hdf5(
    smiles_dict: Dict[str, str],
    output_path: str | Path,
    max_workers: int = 14,
) -> Tuple[int, List[FailedMol]]:
    """Generate 3D features for all molecules and write to HDF5.

    Args:
        smiles_dict: {mol_id: SMILES} mapping.
        output_path: Path to output HDF5 file.
        max_workers: Number of parallel workers (default 14, leaving 2 threads for OS).

    Returns:
        (n_success, failed_list) tuple.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = list(smiles_dict.items())
    n_total = len(tasks)
    successes = 0
    failures: List[FailedMol] = []

    with h5py.File(output_path, "w") as hf:
        # Store metadata
        hf.attrs["n_molecules"] = n_total
        hf.attrs["charge_method"] = "gasteiger"
        hf.attrs["conformer_method"] = "ETKDGv3_MMFF94"

        failed_grp = hf.create_group("_failed")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_features_single, t): t[0] for t in tasks}

            for i, future in enumerate(as_completed(futures)):
                result = future.result()

                if isinstance(result, MolFeatures):
                    grp = hf.create_group(result.mol_id)
                    grp.create_dataset(
                        "coords", data=result.coords,
                        compression="gzip", compression_opts=4,
                    )
                    grp.create_dataset(
                        "atomic_nums", data=result.atomic_nums,
                        compression="gzip", compression_opts=4,
                    )
                    grp.create_dataset(
                        "charges", data=result.charges,
                        compression="gzip", compression_opts=4,
                    )
                    grp.attrs["smiles"] = result.smiles
                    grp.attrs["n_atoms"] = len(result.atomic_nums)
                    grp.attrs["charge_method"] = result.charge_method
                    successes += 1
                else:
                    failures.append(result)
                    failed_grp.attrs[result.mol_id] = result.error
                    logger.warning(
                        f"Failed [{result.mol_id}]: {result.error}"
                    )

                if (i + 1) % 100 == 0 or (i + 1) == n_total:
                    logger.info(f"Progress: {i+1}/{n_total} ({successes} ok, {len(failures)} failed)")

        hf.attrs["n_success"] = successes
        hf.attrs["n_failed"] = len(failures)

    logger.info(f"Done: {successes}/{n_total} succeeded, {len(failures)} failed")
    return successes, failures
