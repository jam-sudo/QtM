"""Precompute Rodgers & Rowland Kp baselines for all training molecules.

Ported from Sisyphus ivive.py (lines 296-433). Computes tissue:plasma
partition coefficients for 15 tissues using logP, pKa, compound_type,
and tissue compositions from reference_man.yaml.

Kp_baseline is used by the residual architecture in projector.py:
    Kp_predicted = Kp_baseline * exp(tanh(Delta_Kp))

IMPORTANT (§6.2): Kp_baseline must be derived from literature or training
set data only. Test set PK data must NOT inform Kp_baseline.

Usage:
    python scripts/precompute_kp_baseline.py \
        --smiles-csv data/train_smiles.csv \
        --output data/kp_baselines.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.data.graph_topology import KP_TISSUE_NAMES

logger = logging.getLogger(__name__)

# Plasma composition (standard values)
# Source: Rodgers & Rowland (2005), Table II
PLASMA_FN = 0.0023    # neutral lipid
PLASMA_FP = 0.0013    # phospholipid
PLASMA_FW = 0.945     # water
PLASMA_PH = 7.4       # blood pH


@dataclass
class TissueComp:
    fn: float
    fp: float
    fw: float
    pH: float


def _lipid_partition(comp: TissueComp, p: float) -> float:
    """fn * P + fp * (0.3P + 0.7)"""
    return comp.fn * p + comp.fp * (0.3 * p + 0.7)


def compute_kp_rr(
    logp: float,
    pka: Optional[float],
    compound_type: str,
    tissue: TissueComp,
    plasma: TissueComp,
) -> float:
    """Rodgers & Rowland Kp calculation.

    Neutral: Kp = (fw_t + fn_t*P + fp_t*(0.3P+0.7)) / (fw_p + fn_p*P + fp_p*(0.3P+0.7))
    Acid: uses distribution coefficient D with ionization ratio
    Base: adds phospholipid binding for cationic species

    Source: Rodgers & Rowland, Pharm Res (2005) 22:1495; (2006) 23:56.
    """
    p = 10 ** logp

    if compound_type == "neutral" or pka is None:
        num = tissue.fw + _lipid_partition(tissue, p)
        den = plasma.fw + _lipid_partition(plasma, p)
        kp = num / den if den > 0 else 1.0

    elif compound_type == "acid":
        d_t = p / (1.0 + 10 ** (tissue.pH - pka))
        d_p = p / (1.0 + 10 ** (plasma.pH - pka))
        ion_r = (1.0 + 10 ** (tissue.pH - pka)) / (1.0 + 10 ** (plasma.pH - pka))
        num = tissue.fw * ion_r + _lipid_partition(tissue, d_t)
        den = plasma.fw + _lipid_partition(plasma, d_p)
        kp = num / den if den > 0 else 1.0

    elif compound_type in ("base", "zwitterion"):
        ion_r = (1.0 + 10 ** (pka - tissue.pH)) / (1.0 + 10 ** (pka - plasma.pH))
        pl_binding = tissue.fp * max(ion_r - 1.0, 0.0)
        num = tissue.fw * ion_r + _lipid_partition(tissue, p) + pl_binding
        den = plasma.fw + _lipid_partition(plasma, p)
        kp = num / den if den > 0 else 1.0

    else:
        num = tissue.fw + _lipid_partition(tissue, p)
        den = plasma.fw + _lipid_partition(plasma, p)
        kp = num / den if den > 0 else 1.0

    return float(np.clip(kp, 0.01, 50.0))


def berezhkovskiy_correction(kp: float, fup: float) -> float:
    """Berezhkovskiy (2004) correction: Kp_bz = Kp / (1 + (Kp-1)*fup)."""
    denom = 1.0 + (kp - 1.0) * fup
    if denom < 1e-10:
        return kp
    return float(np.clip(kp / denom, 0.01, 50.0))


def classify_compound(smiles: str) -> Tuple[str, Optional[float]]:
    """Heuristic compound type classification from SMILES.

    Returns (compound_type, pKa_estimate).
    Simple rule-based: checks for carboxylic acids, amines, etc.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "neutral", None

    smarts_acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")  # -COOH
    smarts_base = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")  # primary/secondary amine

    has_acid = mol.HasSubstructMatch(smarts_acid)
    has_base = mol.HasSubstructMatch(smarts_base)

    if has_acid and has_base:
        return "zwitterion", 7.0  # approximate
    elif has_acid:
        return "acid", 4.0  # typical carboxylic acid
    elif has_base:
        return "base", 9.0  # typical amine
    else:
        return "neutral", None


def load_tissue_compositions(yaml_path: str | Path) -> Dict[str, TissueComp]:
    """Extract tissue compositions from reference_man.yaml."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    comps = {}
    for node in cfg["nodes"]:
        comp_data = node.get("composition")
        if comp_data:
            # Use lookup_name if available (for perm-limited tissues)
            name = node.get("lookup_name", node["name"])
            comps[name] = TissueComp(
                fn=comp_data["fn"],
                fp=comp_data["fp"],
                fw=comp_data["fw"],
                pH=comp_data.get("pH", 7.0),
            )
    return comps


def compute_all_kp_baselines(
    smiles: str,
    tissue_comps: Dict[str, TissueComp],
    fup: float = 0.5,
) -> Dict[str, float]:
    """Compute Kp baseline for all 15 tissues for one molecule.

    Args:
        smiles: SMILES string.
        tissue_comps: {tissue_name: TissueComp} from YAML.
        fup: Fraction unbound in plasma (for Berezhkovskiy correction).

    Returns:
        {tissue_name: Kp_corrected} for all 15 KP_TISSUE_NAMES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {t: 1.0 for t in KP_TISSUE_NAMES}

    logp = Descriptors.MolLogP(mol)
    compound_type, pka = classify_compound(smiles)
    plasma = TissueComp(fn=PLASMA_FN, fp=PLASMA_FP, fw=PLASMA_FW, pH=PLASMA_PH)

    result = {}
    for tissue_name in KP_TISSUE_NAMES:
        tc = tissue_comps.get(tissue_name)
        if tc is None:
            result[tissue_name] = 1.0
            continue
        kp_rr = compute_kp_rr(logp, pka, compound_type, tc, plasma)
        kp_bz = berezhkovskiy_correction(kp_rr, fup)
        result[tissue_name] = kp_bz

    return result


def compute_kp_baselines_batch(
    smiles_list: List[str],
    yaml_path: str | Path,
    fup_values: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """Compute Kp baselines for a batch of molecules.

    Args:
        smiles_list: List of SMILES strings.
        yaml_path: Path to reference_man.yaml.
        fup_values: Optional per-molecule fup. Defaults to 0.5 for all.

    Returns:
        List of {tissue_name: Kp} dicts, one per molecule.
    """
    tissue_comps = load_tissue_compositions(yaml_path)

    results = []
    for i, smi in enumerate(smiles_list):
        fup = fup_values[i] if fup_values else 0.5
        kp = compute_all_kp_baselines(smi, tissue_comps, fup=fup)
        results.append(kp)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Precompute Kp baselines")
    parser.add_argument("--yaml", default="configs/reference_man.yaml")
    parser.add_argument("--smiles-csv", required=True, help="CSV with 'smiles' column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--fup-col", default=None, help="Column name for fup values")
    args = parser.parse_args()

    # Load SMILES
    import pandas as pd
    df = pd.read_csv(args.smiles_csv)
    smiles_list = df["smiles"].tolist()
    fup_values = df[args.fup_col].tolist() if args.fup_col and args.fup_col in df.columns else None

    logger.info(f"Computing Kp baselines for {len(smiles_list)} molecules...")
    baselines = compute_kp_baselines_batch(smiles_list, args.yaml, fup_values)

    # Write output
    rows = []
    for smi, kp in zip(smiles_list, baselines):
        row = {"smiles": smi}
        row.update(kp)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    logger.info(f"Saved to {args.output}")
