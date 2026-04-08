"""Prepare all training data for QtM pipeline.

Loads Sisyphus datasets, runs scaffold split, generates HDF5 molecular
features, computes train-only normalization stats, and Kp baselines.

Outputs:
    data/molecules.h5      — 3D coords + charges for all molecules
    data/splits.json       — scaffold-aware train/val/test indices
    data/norm_stats.json   — train-only normalization statistics
    data/kp_baselines.csv  — Rodgers & Rowland Kp baselines (train molecules)
    data/adme_labels.json  — merged ADME labels for Stage 1
    data/pk_data.json      — CT curve data for Stage 2

Usage:
    python scripts/prepare_data.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.normalization import NormStats, compute_norm_stats
from src.data.quantum_generator import generate_hdf5
from src.data.scaffold_split import create_scaffold_split
from scripts.precompute_kp_baseline import (
    compute_kp_baselines_batch,
    load_tissue_compositions,
)
from src.data.graph_topology import KP_TISSUE_NAMES

logger = logging.getLogger(__name__)


def load_clinical_pk(path: str) -> dict:
    """Load clinical_pk.json → {drug_name: {smiles, dose_mg, time_h, conc_mg_L}}."""
    with open(path) as f:
        raw = json.load(f)

    pk_data = {}
    for name, drug in raw["drugs"].items():
        ct = drug.get("ct_curve")
        if not ct or "time_h" not in ct:
            continue
        smi = drug.get("smiles")
        if not smi:
            continue
        pk_data[name] = {
            "smiles": smi,
            "dose_mg": drug.get("dose_mg", 100.0),
            "time_h": ct["time_h"],
            "conc_mg_L": ct["conc_mg_L"],
        }
    return pk_data


def load_adme_datasets(cfg: dict) -> dict:
    """Load and merge ADME datasets → {smiles: {property: value}}.

    Properties are log-transformed for numerical stability:
    - log_clint: log10(CLint + 1)
    - logit_fup: log(fup / (1-fup))
    - log_vdss: log10(Vdss)
    - log_thalf: log10(t_half)
    """
    labels = {}

    # CLint from clint_expanded_v2.csv
    clint_path = cfg["data"].get("clint_csv")
    if clint_path and Path(clint_path).exists():
        df = pd.read_csv(clint_path)
        smi_col = [c for c in df.columns if "smiles" in c.lower() or "canon" in c.lower()]
        val_col = [c for c in df.columns if "clint" in c.lower()]
        if smi_col and val_col:
            for _, row in df.iterrows():
                smi = str(row[smi_col[0]])
                val = row[val_col[0]]
                if pd.notna(val) and val > 0:
                    labels.setdefault(smi, {})["log_clint"] = float(np.log10(val + 1))
        logger.info(f"CLint: {sum(1 for v in labels.values() if 'log_clint' in v)} molecules")

    # Vdss from vdss_v2_training.csv
    vdss_path = cfg["data"].get("vdss_csv")
    if vdss_path and Path(vdss_path).exists():
        df = pd.read_csv(vdss_path)
        smi_col = [c for c in df.columns if "smiles" in c.lower() or "canon" in c.lower()]
        val_col = [c for c in df.columns if "vdss" in c.lower() or "log_vdss" in c.lower()]
        if smi_col and val_col:
            for _, row in df.iterrows():
                smi = str(row[smi_col[0]])
                val = row[val_col[0]]
                if pd.notna(val):
                    labels.setdefault(smi, {})["log_vdss"] = float(val)
        logger.info(f"Vdss: {sum(1 for v in labels.values() if 'log_vdss' in v)} molecules")

    # PBPK features (fup, clint from mmpk)
    pbpk_path = cfg["data"].get("pbpk_features_csv")
    if pbpk_path and Path(pbpk_path).exists():
        df = pd.read_csv(pbpk_path)
        for _, row in df.iterrows():
            smi = str(row.get("smiles", ""))
            if not smi or smi == "nan":
                continue
            fup = row.get("fup")
            clint = row.get("clint")
            if pd.notna(fup) and 0 < fup < 1:
                fup_clamped = max(min(fup, 0.999), 0.001)
                labels.setdefault(smi, {})["logit_fup"] = float(
                    np.log(fup_clamped / (1 - fup_clamped))
                )
            if pd.notna(clint) and clint > 0:
                labels.setdefault(smi, {}).setdefault("log_clint", float(np.log10(clint + 1)))
        logger.info(f"PBPK features: {len(df)} rows processed")

    return labels


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    # ── 1. Load PK data (Stage 2) ────────────────────────────
    logger.info("Loading clinical PK data...")
    pk_data = load_clinical_pk(cfg["data"]["clinical_pk_json"])
    logger.info(f"  {len(pk_data)} drugs with CT curves")

    # ── 2. Load ADME labels (Stage 1) ────────────────────────
    logger.info("Loading ADME datasets...")
    adme_labels = load_adme_datasets(cfg)
    logger.info(f"  {len(adme_labels)} molecules with ADME labels")

    # ── 3. Collect all unique SMILES ─────────────────────────
    all_smiles = set()
    pk_smiles_map = {}  # drug_name → smiles
    for name, data in pk_data.items():
        all_smiles.add(data["smiles"])
        pk_smiles_map[name] = data["smiles"]
    for smi in adme_labels.keys():
        all_smiles.add(smi)

    smiles_list = sorted(all_smiles)
    smiles_to_id = {smi: f"mol_{i:05d}" for i, smi in enumerate(smiles_list)}
    logger.info(f"  Total unique molecules: {len(smiles_list)}")

    # ── 4. Scaffold split ────────────────────────────────────
    logger.info("Running scaffold split...")
    split_cfg = cfg.get("split", {})
    ratios = tuple(split_cfg.get("ratios", [0.8, 0.1, 0.1]))
    seed = split_cfg.get("seed", 42)

    # Load holdout if available
    holdout_smiles = None
    holdout_path = cfg["data"].get("holdout_json")
    if holdout_path and Path(holdout_path).exists():
        with open(holdout_path) as f:
            holdout = json.load(f)
        if "holdout_drugs" in holdout:
            holdout_smiles = [d.get("smiles") for d in holdout["holdout_drugs"] if d.get("smiles")]
            logger.info(f"  Holdout set: {len(holdout_smiles)} drugs")

    splits = create_scaffold_split(smiles_list, ratios=ratios, seed=seed,
                                   holdout_smiles=holdout_smiles)
    logger.info(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    # Save splits
    splits_out = {
        "smiles_list": smiles_list,
        "smiles_to_id": smiles_to_id,
        "train_indices": splits["train"],
        "val_indices": splits["val"],
        "test_indices": splits["test"],
    }
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits_out, f)
    logger.info(f"  Saved splits to {out_dir / 'splits.json'}")

    # ── 5. Generate HDF5 features ────────────────────────────
    logger.info("Generating HDF5 molecular features...")
    smiles_dict = {smiles_to_id[smi]: smi for smi in smiles_list}
    hdf5_path = out_dir / "molecules.h5"
    n_ok, failed = generate_hdf5(
        smiles_dict, hdf5_path,
        max_workers=cfg.get("hdf5", {}).get("max_workers", 14),
    )
    logger.info(f"  HDF5: {n_ok} succeeded, {len(failed)} failed")

    # Remove failed molecules from splits
    failed_ids = {f.mol_id for f in failed}
    failed_smiles = {smiles_dict[fid] for fid in failed_ids if fid in smiles_dict}
    for split_name in ["train", "val", "test"]:
        splits[split_name] = [i for i in splits[split_name] if smiles_list[i] not in failed_smiles]

    # ── 6. Compute normalization stats (train only) ──────────
    logger.info("Computing normalization stats (train set only)...")
    train_charges = []
    with h5py.File(hdf5_path, "r") as hf:
        for idx in splits["train"]:
            mol_id = smiles_to_id[smiles_list[idx]]
            if mol_id in hf:
                train_charges.append(hf[mol_id]["charges"][:])

    if train_charges:
        norm_stats = compute_norm_stats({"charges": np.concatenate(train_charges)})
        norm_stats.save(out_dir / "norm_stats.json")
        logger.info(f"  Charge stats: mean={norm_stats.feature_means['charges']:.4f}, "
                     f"std={norm_stats.feature_stds['charges']:.4f}")

    # ── 7. Compute Kp baselines (train molecules) ────────────
    logger.info("Computing Kp baselines...")
    train_smiles = [smiles_list[i] for i in splits["train"]]

    # Get fup values where available
    fup_values = []
    for smi in train_smiles:
        labels = adme_labels.get(smi, {})
        logit_fup = labels.get("logit_fup")
        if logit_fup is not None:
            fup = 1.0 / (1.0 + np.exp(-logit_fup))
            fup_values.append(float(fup))
        else:
            fup_values.append(0.5)  # default

    baselines = compute_kp_baselines_batch(
        train_smiles, cfg["data"]["physiology_yaml"], fup_values
    )

    # Save as CSV
    rows = []
    for smi, kp in zip(train_smiles, baselines):
        row = {"smiles": smi}
        row.update({f"kp_{t}": v for t, v in kp.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "kp_baselines.csv", index=False)
    logger.info(f"  Kp baselines: {len(rows)} molecules")

    # ── 8. Save ADME labels and PK data ──────────────────────
    with open(out_dir / "adme_labels.json", "w") as f:
        # Map SMILES → mol_id for dataset
        adme_by_id = {}
        for smi, labels in adme_labels.items():
            mol_id = smiles_to_id.get(smi)
            if mol_id:
                adme_by_id[mol_id] = labels
        json.dump(adme_by_id, f)
    logger.info(f"  ADME labels: {len(adme_by_id)} molecules")

    pk_by_id = {}
    for name, data in pk_data.items():
        mol_id = smiles_to_id.get(data["smiles"])
        if mol_id:
            pk_by_id[mol_id] = {
                "drug_name": name,
                "dose_mg": data["dose_mg"],
                "time_h": data["time_h"],
                "conc_mg_L": data["conc_mg_L"],
            }
    with open(out_dir / "pk_data.json", "w") as f:
        json.dump(pk_by_id, f)
    logger.info(f"  PK data: {len(pk_by_id)} drugs with CT curves")

    # ── Summary ──────────────────────────────────────────────
    logger.info("\n=== Data Preparation Complete ===")
    logger.info(f"  Molecules: {len(smiles_list)} total, {n_ok} with features")
    logger.info(f"  Split: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")
    logger.info(f"  ADME labels: {len(adme_by_id)} molecules")
    logger.info(f"  PK curves: {len(pk_by_id)} drugs")
    logger.info(f"  Kp baselines: {len(rows)} molecules")
    logger.info(f"  Output dir: {out_dir}")


if __name__ == "__main__":
    main()
