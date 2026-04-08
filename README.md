# QtM (Quantum-to-Macro)

End-to-end differentiable PBPK system that predicts human pharmacokinetic curves directly from molecular structure.

SMILES &rarr; 3D conformer &rarr; SchNet encoder &rarr; PBPK parameter projector &rarr; 34-compartment Neural ODE &rarr; plasma concentration-time curve

## Architecture

```
                  Module A (offline)              Module B                Module C                   Module D
                 ┌──────────────┐            ┌──────────────┐       ┌──────────────────┐      ┌─────────────────────┐
  SMILES ──────► │ RDKit ETKDGv3│──► HDF5 ──►│ SchNet (3×CF)│──────►│ Hierarchical     │─────►│ 34-compartment ODE  │──► C(t)
                 │ 3D + charges │            │ Z_mol [128]  │       │ Projector → 30θ  │      │ PSSA + rk4 solver   │
                 └──────────────┘            └──────────────┘       └──────────────────┘      └─────────────────────┘
```

| Module | Description | Parameters |
|--------|-------------|------------|
| **A** Quantum Generator | RDKit ETKDGv3 conformer + Gasteiger charges &rarr; HDF5 | Offline |
| **B** SchNet Encoder | 3-layer continuous-filter convolution, E(3)-invariant | 254,080 |
| **C** Hierarchical Projector | Tissue-grouped attention &rarr; 30 drug-dependent ODE parameters | 18,846 |
| **D** PBPK ODE | 34-node multigraph (ICRP Reference Man, 70 kg), 5 flux types | 0 (physics) |

**Total trainable parameters: 272,926**

### PBPK Graph Topology

34 compartments: 15 organs, 7 blood pools (3 algebraic via PSSA), 8 GI lumen segments, 4 sinks.
54 edges: 31 flow, 4 diffusion, 8 transit, 8 absorption, 3 clearance.

Physiology source: ICRP Reference Man (cardiac output 390 L/h, body weight 70 kg).

### 30 Predicted Drug Parameters

| Group | Count | Output activation | Range |
|-------|-------|-------------------|-------|
| Tissue partition coefficients (Kp) | 15 | Residual: Kp_baseline &times; exp(tanh(&Delta;)), clamp | [1.0, 50] |
| CYP enzyme affinities | 5 | Sigmoid | [0.001, 0.1] &mu;L/min/pmol |
| Permeability-surface area products | 4 | Sigmoid | [1, 50] L/h |
| Fraction unbound in plasma (fup) | 1 | Sigmoid | [0.01, 1.0] |
| Blood:plasma ratio (rbp) | 1 | Sigmoid | [0.5, 2.0] |
| Effective permeability (Peff) | 1 | Sigmoid | [0.01, 10] &times;10<sup>-4</sup> cm/s |
| Renal clearance | 1 | Sigmoid | [0.01, 20] L/h |
| Particle radius | 1 | Sigmoid | [5, 50] &mu;m |
| Solubility | 1 | Softplus | [0, &infin;) mg/mL |

Solubility is predicted but not used in the ODE (reserved for future multi-task loss).

Kp baselines computed via Rodgers & Rowland method with Berezhkovskiy correction.

### Pseudo-Steady State Approximation (PSSA)

Venous blood, arterial blood, and portal vein are converted from differential to algebraic equations:

```
A_bp = V_bp * sum(Q_in * C_out_src) / Q_out_total
```

This eliminates eigenvalues up to &lambda;=1482 h<sup>-1</sup>, reducing the stiffness ratio from ~1500:1 to ~120:1 while preserving the full 34-node topology.

## Training

Two-stage curriculum:

### Stage 1: ADME Multi-task Pretraining

Encoder warm-up on 3,308 molecules with scalar ADME labels (CLint, fup, Vdss, Caco-2 Papp, t&frac12;).
Trains SchNet encoder to produce meaningful molecular representations before ODE fine-tuning.

- Batch size 64, lr 1e-3, cosine annealing, 100 epochs
- Best fup R&sup2; = 0.65

### Stage 2: PBPK ODE Fine-tuning

Full pipeline (encoder &rarr; projector &rarr; ODE) on clinical concentration-time curves (265 total, 245 in train split).

- Encoder frozen (default config: `freeze_encoder_epochs` = `epochs` = 10)
- Projector lr 1e-4, encoder lr 1e-5 (active when unfrozen), gradient accumulation (effective batch = 8)
- rk4 fixed-step (dt=0.005h), truncated BPTT with 0.5h windows
- Loss: MSLE(C_plasma_pred, C_plasma_obs) + &lambda;(epoch) &times; mass balance penalty

### Loss Function

```
Total = MSLE_PK + lambda(epoch) * Mass_Balance_Penalty
lambda(epoch) = min(lambda_max, lambda_max * epoch / warmup_epochs)
MSLE = mean( (log(1 + pred) - log(1 + obs))^2 )
```

Note: default config has `warmup_epochs=20` with `stage2.epochs=10`, so &lambda; peaks at 0.45 (never reaches &lambda;_max=1.0).

## Dataset

| Source | Molecules | Usage |
|--------|-----------|-------|
| Clinical PK curves (Sisyphus) | 265 | Stage 2 ODE fine-tuning |
| ADME labels (CLint, fup, Vdss, Papp, t&frac12;) | 3,308 | Stage 1 pretraining |
| Total unique molecules | 3,554 | |
| Scaffold-aware split | 2,843 / 355 / 356 | Train / Val / Test |

Data leakage prevention: Murcko scaffold split, train-only normalization, Kp baseline isolation.

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.x (CPU or CUDA)
- PyTorch Geometric 2.7+
- torchdiffeq 0.2+
- RDKit 2023+
- h5py, pandas, pyyaml

### Data Preparation

Requires access to Sisyphus reference data (`clinical_pk.json`, ADME datasets).

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

Outputs: `data/molecules.h5`, `data/splits.json`, `data/norm_stats.json`, `data/kp_baselines.csv`, `data/adme_labels.json`, `data/pk_data.json`

### Training

```bash
# Stage 1 only (ADME pretraining)
python -m scripts.run_training --config configs/default.yaml --stage 1

# Stage 2 only (ODE fine-tuning, requires Stage 1 checkpoint)
python -m scripts.run_training --config configs/default.yaml --stage 2 --device cpu

# Both stages sequentially
python -m scripts.run_training --config configs/default.yaml --stage both --device cpu

# Quick test run (Stage 1: 5 epochs, Stage 2: 3 epochs)
python -m scripts.run_training --config configs/default.yaml --dry-run --device cpu
```

Stage 2 is recommended to run on CPU (`--device cpu`). With batch_size=1, the 31-dimensional ODE solves are ~8x faster on CPU than GPU due to CUDA kernel launch overhead.

## Project Structure

```
QtM/
  configs/
    default.yaml              # All hyperparameters
    reference_man.yaml         # 34-node PBPK physiology (ICRP Reference Man)
  src/
    data/
      quantum_generator.py     # Module A: SMILES -> 3D + charges -> HDF5
      graph_topology.py        # YAML -> PBPKTopology (fixed PyTorch tensors)
      dataset.py               # ADMEDataset (Stage 1), PKCurveDataset (Stage 2)
      scaffold_split.py        # Murcko scaffold-aware train/val/test split
      normalization.py         # Train-only normalization statistics
    models/
      encoder.py               # Module B: SchNet (no torch-cluster dependency)
      projector.py             # Module C: Hierarchical parameter projection
      ode_system.py            # Module D: Batched ODE RHS + PSSAWrapper (DAE)
      pipeline.py              # End-to-end wiring (B -> C -> D)
    training/
      train_adme.py            # Stage 1: ADME multi-task pretraining
      train_ode.py             # Stage 2: PBPK ODE fine-tuning
      loss.py                  # Annealed PINN loss (MSLE + mass balance)
      logger.py                # Run logging (JSON lines, divergence tracking)
  scripts/
    prepare_data.py            # Master data preparation pipeline
    precompute_kp_baseline.py  # Rodgers & Rowland Kp calculation
    run_training.py            # Master training script (Stage 1 + 2)
```

## Known Limitations

- **Kp floor at 1.0**: Required for rk4 solver stability. Prevents modeling BBB-restricted drugs or tissues with Kp < 1.0. Future work: forward sensitivity equations or implicit solver.
- **PSSA mass balance**: Algebraic blood pool approximation introduces ~1-5% mass conservation error.
- **Truncated BPTT (0.5h windows)**: Limits long-range temporal gradient flow. Gradients only propagate within each 0.5h window; cross-window signal comes from the PSSA algebraic path only.
- **Enzyme affinity floor**: Minimum CLint &asymp; 1.4 L/h excludes very metabolically stable compounds (t&frac12; > 24h).
- **Float32 backward overflow**: Backward through >150 rk4 steps overflows float32. The 0.5h window (100 steps) keeps gradients within representable range.

## Experimental Integrity

Per CLAUDE.md sections 6.1 and 6.2:

- **All runs logged**: every training run recorded with UUID, seed, hyperparameters, epoch metrics, divergence rate.
- **No post-hoc subset selection**: metrics reported on full pre-defined test set.
- **Solver divergence transparency**: failed molecules logged individually (molecule ID + error), divergence rate included in epoch metrics.
- **Scaffold-aware split**: Murcko scaffold grouping prevents structural leakage between train/val/test.
- **Normalization firewall**: all feature statistics computed from training set only.
- **Kp baseline isolation**: Rodgers & Rowland baselines derived from literature/training data only.

## References

- Rodgers T, Rowland M. Physiologically based pharmacokinetic modelling 2: predicting the tissue distribution of acids, very weak bases, neutrals and zwitterions. J Pharm Sci. 2006;95(6):1238-1257.
- Berezhkovskiy LM. Volume of distribution at steady state for a linear pharmacokinetic system with peripheral elimination. J Pharm Sci. 2004;93(6):1628-1640.
- Schutt KT, et al. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. NeurIPS. 2017.
- Chen RTQ, et al. Neural Ordinary Differential Equations. NeurIPS. 2018.
