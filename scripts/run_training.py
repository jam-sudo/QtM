"""Master training script for Project Sisyphus-Q.

Stage 1: ADME multi-task pretraining (encoder warm-up)
Stage 2: PBPK ODE fine-tuning (full pipeline)

Usage:
    python scripts/run_training.py --config configs/default.yaml [--stage 1|2|both]
    python scripts/run_training.py --config configs/default.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.data.dataset import ADMEDataset, PKCurveDataset
from src.data.graph_topology import KP_TISSUE_NAMES, load_topology
from src.data.normalization import NormStats
from src.models.encoder import SchNetEncoder
from src.models.ode_system import PBPKFunc
from src.models.projector import HierarchicalProjector
from src.training.logger import RunLogger
from src.training.loss import adme_multitask_loss
from src.training.train_adme import ADMEHead, eval_adme, train_adme_epoch
from src.training.train_ode import train_one_epoch

logger = logging.getLogger(__name__)


def load_splits_and_ids(data_dir: Path) -> dict:
    with open(data_dir / "splits.json") as f:
        return json.load(f)


def run_stage1(cfg: dict, device: torch.device, dry_run: bool = False):
    """Stage 1: ADME multi-task pretraining."""
    logger.info("=" * 60)
    logger.info("STAGE 1: ADME Multi-task Pretraining")
    logger.info("=" * 60)

    data_dir = Path("data")
    s1 = cfg["stage1"]
    enc_cfg = cfg["encoder"]

    # Load splits
    splits_data = load_splits_and_ids(data_dir)
    smiles_list = splits_data["smiles_list"]
    smiles_to_id = splits_data["smiles_to_id"]
    train_ids = [smiles_to_id[smiles_list[i]] for i in splits_data["train_indices"]]
    val_ids = [smiles_to_id[smiles_list[i]] for i in splits_data["val_indices"]]

    # Load ADME labels
    with open(data_dir / "adme_labels.json") as f:
        adme_labels = json.load(f)

    # Filter to molecules that have labels AND exist in HDF5
    import h5py
    with h5py.File(data_dir / "molecules.h5", "r") as hf:
        hdf5_keys = set(hf.keys())
    train_ids = [mid for mid in train_ids if mid in adme_labels and mid in hdf5_keys]
    val_ids = [mid for mid in val_ids if mid in adme_labels and mid in hdf5_keys]
    logger.info(f"Train: {len(train_ids)} | Val: {len(val_ids)} molecules with ADME labels + HDF5")

    if not train_ids:
        logger.warning("No training molecules with ADME labels. Skipping Stage 1.")
        return None

    # Load norm stats
    norm_stats = None
    if (data_dir / "norm_stats.json").exists():
        norm_stats = NormStats.load(data_dir / "norm_stats.json")

    # Datasets
    target_names = s1.get("target_names", ["log_clint", "logit_fup", "log_vdss", "log_papp", "log_thalf"])
    train_ds = ADMEDataset(
        data_dir / "molecules.h5", train_ids, adme_labels,
        cutoff=enc_cfg["cutoff"], norm_stats=norm_stats, target_names=target_names,
    )
    val_ds = ADMEDataset(
        data_dir / "molecules.h5", val_ids, adme_labels,
        cutoff=enc_cfg["cutoff"], norm_stats=norm_stats, target_names=target_names,
    )

    train_loader = DataLoader(train_ds, batch_size=s1["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=s1["batch_size"])

    # Models
    encoder = SchNetEncoder(
        hidden_channels=enc_cfg["hidden_channels"],
        num_interactions=enc_cfg["num_interactions"],
        num_gaussians=enc_cfg["num_gaussians"],
        cutoff=enc_cfg["cutoff"],
    ).to(device)

    adme_head = ADMEHead(
        z_dim=enc_cfg["hidden_channels"],
        n_tasks=len(target_names),
    ).to(device)

    optimizer = AdamW(
        list(encoder.parameters()) + list(adme_head.parameters()),
        lr=s1["lr"], weight_decay=s1.get("weight_decay", 1e-5),
    )

    n_epochs = 5 if dry_run else s1["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    run_logger = RunLogger(
        cfg["logging"]["run_log"], seed=cfg["split"]["seed"],
        hparams={"stage": 1, **s1},
    )

    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        metrics = train_adme_epoch(encoder, adme_head, train_loader, optimizer, device)
        scheduler.step()

        val_metrics = eval_adme(encoder, adme_head, val_loader, device)
        all_metrics = {**metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}

        run_logger.log_epoch(epoch, all_metrics)
        logger.info(f"Epoch {epoch:3d} | loss={metrics['loss']:.4f} | "
                     + " | ".join(f"{k}={v:.3f}" for k, v in val_metrics.items() if "r2" in k))

        # Save best
        val_loss = metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), ckpt_dir / "encoder_pretrained.pt")
            torch.save(adme_head.state_dict(), ckpt_dir / "adme_head.pt")

    run_logger.finalize("success", {"best_val_loss": best_val_loss})
    logger.info(f"Stage 1 complete. Best val loss: {best_val_loss:.4f}")

    # Cleanup dataset handles
    train_ds.close()
    val_ds.close()
    return encoder


def run_stage2(cfg: dict, device: torch.device, encoder: SchNetEncoder | None = None, dry_run: bool = False):
    """Stage 2: PBPK ODE fine-tuning."""
    logger.info("=" * 60)
    logger.info("STAGE 2: PBPK ODE Fine-tuning")
    logger.info("=" * 60)

    data_dir = Path("data")
    s2 = cfg["stage2"]
    enc_cfg = cfg["encoder"]
    proj_cfg = cfg["projector"]

    # Load topology
    topo = load_topology(cfg["data"]["physiology_yaml"])
    topo = topo.to(device)

    # Load splits
    splits_data = load_splits_and_ids(data_dir)
    smiles_list = splits_data["smiles_list"]
    smiles_to_id = splits_data["smiles_to_id"]

    # Load PK data
    with open(data_dir / "pk_data.json") as f:
        pk_data = json.load(f)

    # Filter train molecules with PK curves AND in HDF5
    import h5py as _h5
    with _h5.File(data_dir / "molecules.h5", "r") as hf:
        hdf5_keys = set(hf.keys())
    train_pk_ids = [
        smiles_to_id[smiles_list[i]]
        for i in splits_data["train_indices"]
        if smiles_to_id[smiles_list[i]] in pk_data and smiles_to_id[smiles_list[i]] in hdf5_keys
    ]
    logger.info(f"Train molecules with CT curves + HDF5: {len(train_pk_ids)}")

    if not train_pk_ids:
        logger.warning("No training molecules with PK curves. Cannot run Stage 2.")
        return

    # Load Kp baselines
    import pandas as pd
    kp_df = pd.read_csv(data_dir / "kp_baselines.csv")
    kp_baselines = {}
    for _, row in kp_df.iterrows():
        smi = row["smiles"]
        mol_id = smiles_to_id.get(smi)
        if mol_id:
            kp_vals = torch.tensor(
                [row.get(f"kp_{t}", 1.0) for t in KP_TISSUE_NAMES],
                dtype=torch.float32,
            )
            kp_baselines[mol_id] = kp_vals

    # Load norm stats
    norm_stats = None
    if (data_dir / "norm_stats.json").exists():
        norm_stats = NormStats.load(data_dir / "norm_stats.json")

    # Dataset
    train_ds = PKCurveDataset(
        data_dir / "molecules.h5", train_pk_ids, pk_data, kp_baselines,
        cutoff=enc_cfg["cutoff"], norm_stats=norm_stats,
    )
    train_loader = DataLoader(train_ds, batch_size=s2["batch_size"], shuffle=True)

    # Models
    if encoder is None:
        encoder = SchNetEncoder(
            hidden_channels=enc_cfg["hidden_channels"],
            num_interactions=enc_cfg["num_interactions"],
            num_gaussians=enc_cfg["num_gaussians"],
            cutoff=enc_cfg["cutoff"],
        ).to(device)
        # Try loading pretrained
        ckpt = Path(cfg["logging"]["checkpoint_dir"]) / "encoder_pretrained.pt"
        if ckpt.exists():
            encoder.load_state_dict(torch.load(ckpt, map_location=device))
            logger.info(f"Loaded pretrained encoder from {ckpt}")
    else:
        encoder = encoder.to(device)

    projector = HierarchicalProjector(
        z_dim=proj_cfg["z_dim"], hidden_dim=proj_cfg["hidden_dim"],
    ).to(device)

    pbpk_func = PBPKFunc(topo).to(device)

    # Optimizer: separate LRs for encoder (lower) and projector
    param_groups = [
        {"params": projector.parameters(), "lr": s2["projector_lr"]},
        {"params": encoder.parameters(), "lr": s2["encoder_lr"]},
    ]
    optimizer = AdamW(param_groups, weight_decay=s2.get("weight_decay", 1e-5))

    n_epochs = 3 if dry_run else s2["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    run_logger = RunLogger(
        cfg["logging"]["run_log"], seed=cfg["split"]["seed"],
        hparams={"stage": 2, **s2},
    )

    # Freeze encoder for first N epochs
    freeze_epochs = s2.get("freeze_encoder_epochs", 10)

    pipeline_modules = {
        "encoder": encoder,
        "projector": projector,
        "pbpk_func": pbpk_func,
        "topo": topo,
        "volumes": topo.volumes,
    }

    for epoch in range(n_epochs):
        # Freeze/unfreeze encoder
        for p in encoder.parameters():
            p.requires_grad = (epoch >= freeze_epochs)

        metrics = train_one_epoch(
            pipeline_modules, train_loader, optimizer, epoch, device, run_logger,
            accum_steps=s2.get("accum_steps", 4),
            lambda_max=cfg["loss"]["lambda_max"],
            warmup_epochs=cfg["loss"]["warmup_epochs"],
            use_adjoint=False,  # non-adjoint: faster backward, VRAM ok for batch≤8
        )
        scheduler.step()

        run_logger.log_epoch(epoch, metrics)
        logger.info(
            f"Epoch {epoch:3d} | loss={metrics['loss']:.4f} | "
            f"data={metrics['data_loss']:.4f} | mb={metrics['mass_balance']:.6f} | "
            f"div={metrics['divergence_rate']:.2%} | solver={metrics['solver']}"
        )

        # Save checkpoint
        if (epoch + 1) % cfg["logging"].get("save_every_epochs", 10) == 0 or epoch == n_epochs - 1:
            ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
            }, ckpt_dir / f"stage2_epoch{epoch:03d}.pt")

    final_metrics = {
        "final_loss": metrics["loss"],
        "final_divergence_rate": metrics["divergence_rate"],
    }
    run_logger.finalize("success", final_metrics)
    train_ds.close()
    logger.info("Stage 2 complete.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="QtM Training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal epochs for testing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    encoder = None
    if args.stage in ("1", "both"):
        encoder = run_stage1(cfg, device, dry_run=args.dry_run)

    if args.stage in ("2", "both"):
        run_stage2(cfg, device, encoder=encoder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
