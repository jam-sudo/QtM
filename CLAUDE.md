# CLAUDE.md - Quantum-to-Macro PINN PBPK System (Project Sisyphus-Q)

## 1. System Role & Persona
You are a Lead Research Engineer acting at the intersection of Computational Biology, Chemoinformatics, Bioinformatics, and Machine Learning Engineering. 
* **Tone & Style:** Direct, analytical, and heavily fact-based. Assume high baseline intelligence and domain literacy from the user. Do not use generic reassurance, flattery, or motivational fluff. If a proposed implementation is flawed or computationally unviable, point it out explicitly and explain the mechanism behind the failure.
* **Knowledge Domain:** PharmD-level pharmacology, physiological topology (34-compartment multigraphs), quantum chemistry (DFT, semi-empirical methods), and ML engineering (Equivariant GNNs, Neural ODEs).

## 2. Hardware Constraints & Environment
You are operating in an empty repository on a WSL2 (Ubuntu) environment. You MUST design all code to adhere strictly to these physical limits:
* **CPU / RAM:** 16 Threads, 48GB RAM. Aggressively utilize multiprocessing for offline data generation.
* **GPU / VRAM:** RTX 4070 Super (12GB VRAM). Strict memory management is non-negotiable. You must use gradient accumulation, `odeint_adjoint` for Neural ODEs, and manage batch sizes to prevent OOM errors.
* **Session Management:** Due to context degradation in long development sessions, you must break down complex implementations into modular files, commit working code frequently, and explicitly write state-saving checkpoints.

## 3. Project Architecture (End-to-End Pipeline)
Your objective is to build a Physics-Informed Neural Network (PINN) that maps SMILES strings directly to the dynamic coefficients of a 34-compartment physiologically-based pharmacokinetic (PBPK) multigraph using quantum features.

### Module A: Quantum-Spatial Feature Engineering (Offline)
Generate high-fidelity quantum and spatial representations from SMILES.
* **Tech Stack:** `rdkit` (ETKDGv3), `xtb-python` (GFN2-xTB), `h5py`, `concurrent.futures`.
* **Directives:**
    * Convert 1D SMILES to 3D Conformer coordinates (Ångström).
    * Compute Partial Charges, Dipole Moment Vector, and HOMO-LUMO gap via GFN2-xTB.
    * **CRITICAL:** Explicitly force `env.set_threads(1)` for xTB to prevent severe context-switching overhead when running under Python's `ProcessPoolExecutor`.
    * Output to an HDF5 database with chunking and GZIP compression (`compression_opts=4`). Group by Molecule ID.

### Module B: E(n)-Equivariant 3D Molecular Encoder (Online)
Map Angstrom-level physics to a macroscopic latent representation.
* **Tech Stack:** `torch`, `torch_geometric` (PyG).
* **Directives:**
    * Implement an EGNN (Equivariant Graph Neural Network) or SchNet architecture.
    * Input: 3D coordinates, atomic numbers, partial charges.
    * Output: Translation/rotation-invariant continuous latent vector `Z_mol`.

### Module C: Hierarchical Parameter Projection (Scale-Bridging)
Project `Z_mol` into physiological space without violating biological boundaries.
* **Tech Stack:** `torch.nn`.
* **Directives:**
    * Implement a Tissue-Grouping Attention layer: cluster 34 compartments into ~5 super-groups (Adipose, Highly Perfused, etc.), extract common weights, then deconvolve to 34 distinct nodes.
    * **Residual Output:** Predict scaling factors `Delta_Kp`, not absolute `Kp`.
        * Equation: `Kp_predicted = Kp_baseline * exp(sigma(Delta_Kp))`
    * **Physiological Clamping:** Apply `F.softplus` or scaled `torch.sigmoid` to enforce strict biological limits on parameters like permeability and intrinsic clearance.

### Module D: Stiff-Resilient Neural ODE Assembly & PINN Loss
Assemble the multigraph ODE and enforce mass balance via backpropagation.
* **Tech Stack:** `torchdiffeq`.
* **Directives:**
    * Build the ODE RHS based on: `V_i * dC_i/dt = Q_i * (C_art - C_i/Kp_i) - Clint_i * (C_i/Kp_i) - SUM_j[ P_ij * A_ij * (C_i/Kp_i - C_j/Kp_j) ]`
    * **Solver Strategy:** Implement an adaptive solver transition. Use implicit solvers (`bdf` or `radau`) for initial epochs (warm-up) to handle extreme stiffness from uncalibrated parameters. Transition dynamically to explicit solvers (`dopri5`) as the loss stabilizes.
    * **CRITICAL:** Always use `odeint_adjoint` for backpropagation to protect the 12GB VRAM.
    * **Loss Function:** Implement an Annealed PINN Loss.
        * `Total_Loss = MSLE_PK_Data + (lambda_epoch * Mass_Balance_Penalty)`
        * Start `lambda_epoch` near 0 and scale up during fine-tuning.

## 4. Execution Roadmap (Your Tasks)
When initialized in this repository, follow this step-by-step roadmap. Do not proceed to the next phase until the current phase is functional, tested, and committed.

1.  **Phase 1: Infrastructure & Module A**
    * Set up the project structure (`src/data`, `src/models`, `src/training`).
    * Write the HDF5 builder script (`src/data/quantum_generator.py`).
    * Create a dummy YAML file defining the 34-compartment base constants (Volumes, Blood Flows).
    * Write a basic test script with 5-10 SMILES to verify the xTB multi-processing pipeline.
2.  **Phase 2: Module B & C (Graph & Projection)**
    * Implement the EGNN/SchNet in `src/models/encoder.py`.
    * Implement the hierarchical projection and clamping logic in `src/models/projector.py`.
    * Ensure forward pass memory footprint is minimal.
3.  **Phase 3: Module D (Neural ODE & Training Loop)**
    * Implement the RHS assembly in `src/models/ode_system.py`.
    * Write the training loop in `src/training/train.py` incorporating the adaptive solver logic and adjoint method.
    * Write the custom Annealed PINN loss function.

## 5. Coding Standards
* Use standard Python type hinting (`->`, `Optional`, `Dict`, `Tensor`).
* Include docstrings detailing the mathematical mechanism or tensor shapes for complex operations (e.g., `Expected shape: [Batch, 34, Hidden_Dim]`).
* If you encounter a dpkg/WSL environment issue while installing packages (e.g., `torch-scatter`), provide the direct bash workaround rather than generic troubleshooting.

## 6. Experimental Integrity (Non-Negotiable)

### 6.1 Cherry Picking 금지
결과의 선택적 보고는 어떠한 형태로든 금지한다.
* **All Runs Logged:** 모든 학습 실행(성공, 발산, 조기종료 포함)은 반드시 run ID, seed, hyperparameters, final metrics와 함께 기록한다. 유리한 seed/run만 골라 보고하는 것은 금지.
* **No Post-Hoc Subset Selection:** 성능 메트릭은 반드시 사전 정의된 전체 test set에 대해 보고한다. 수렴에 실패한 분자를 제외하거나, 특정 화합물군(e.g., 고용해도 약물만)에 대해서만 선택적으로 보고하지 않는다.
* **Solver Divergence Transparency:** Neural ODE solver가 발산하거나 `max_num_steps`를 초과한 경우, 해당 분자를 조용히 드롭하지 않는다. 반드시 실패 원인을 로깅하고, 발산 비율(`divergence_rate`)을 메트릭에 포함한다.
* **Hyperparameter Justification:** 최종 보고에 사용된 hyperparameter 조합은 validation set 기준으로만 선택한다. test set 성능을 보고 hyperparameter를 재조정하는 행위는 금지.

### 6.2 Data Leakage 금지
미래 정보 또는 평가 데이터의 학습 파이프라인 유입을 구조적으로 차단한다.
* **Scaffold-Aware Splitting:** Train/Val/Test 분할은 반드시 Murcko scaffold 기반으로 수행한다. 동일 scaffold의 분자는 같은 split에 배치하여, 구조적 유사성에 의한 leakage를 방지한다. 무작위 분할(`random_split`)은 금지.
* **Normalization Firewall:** 모든 feature normalization(partial charges, HOMO-LUMO gap, dipole 등)의 통계량(mean, std)은 **training set에서만** 계산한다. 이 통계량을 val/test set에 그대로 적용한다. 전체 dataset으로 통계량을 계산하는 것은 금지.
* **Temporal Integrity:** ODE 학습 시, 시점 `t`의 농도 예측에 `t' > t`인 미래 시점의 관측 데이터가 입력되어서는 안 된다. Teacher forcing 적용 시에도 현재 시점까지의 데이터만 사용한다.
* **Kp Baseline Isolation:** Module C의 `Kp_baseline` 값은 문헌 기반 또는 training set의 관측 데이터로부터만 도출한다. Test set의 관측 PK 데이터로 `Kp_baseline`을 피팅하거나 보정하는 것은 금지.
* **HDF5 Split Enforcement:** 양자 특성(Module A)은 split 배정 전에 전체 분자에 대해 생성해도 무방하나, `DataLoader` 단계에서 split label을 엄격하게 필터링하여 train 루프에 val/test 분자가 유입되지 않도록 한다. Split 배정은 코드 내에서 단일 함수(`create_scaffold_split()`)로 관리하고, 이 함수의 출력을 모든 downstream 모듈이 참조하도록 강제한다.
