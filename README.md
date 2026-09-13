# Autonomous High-Throughput Single-Atom Catalysis Platform with MACE

An end-to-end computational pipeline designed to autonomously synthesize Single-Atom Catalyst (SAC) architectures, sample phase space using ab initio calculations (GPAW PBE-D3 + Langevin AIMD), fine-tune Machine Learning Interatomic Potentials (MACE), and screen reaction kinetics via Climbing-Image Nudged Elastic Band (CI-NEB) and long-timescale molecular dynamics.

---

## Key Features

- **Zero-Prerequisite Substrate Synthesis:** Builds graphene supercells analytically, punches divacancy pores, coordinates target metals, and runs adaptive BFGS pre-relaxations.
- **10 Realistic Motifs:** Systematically covers saturated 4-coordinated ($M\text{-N}_x\text{C}_{4-x}$) and single-vacancy 3-coordinated ($M\text{-N}_x\text{C}_{3-x}$) cavities without unphysical pore-collapse artifacts.
- **Phase-Space Sampling:** Resolves out-of-distribution (OOD) extrapolation errors with targeted $z$-approach scans, $\mathrm{H-H}$ bond cleavage grids, and $NVT$ AIMD snapshots.
- **Automated MACE Fine-Tuning:** Benchmarks zero-shot foundation models (MACE-MP-0), fine-tunes a local potential, and quantifies performance improvements.
- **Catalytic Reaction Profiling:** Automated CI-NEB across all 10 motifs to extract activation barriers ($E_a$), reaction energies ($\Delta E$), and harmonic free energy corrections ($\Delta G^\ddagger$ at $T, P$).
- **Active-Learning Dynamics:** Production MD with automated force-norm tripwires that dump high-uncertainty configurations back to DFT.

---

### Computational Parameters & Convergence Notes

- **Supercell Architecture**: A 4×4 graphene supercell (31–33 atoms depending on H₂ adsorption state) 
  with a 15 Å vacuum spacing along the z-axis is employed as the default benchmark. 
  This setup yields an inter-site Pt–Pt separation of ~9.8 Å, which comfortably accommodates 
  the local interaction cutoff of the MACE foundation model (r_cut = 5.0 Å) while drastically 
  accelerating high-throughput screening.
- **DFT Parameters (GPAW)**: Plane-wave / grid cutoff of 350 eV with PBE-D3 dispersion corrections. 
  This configuration provides smooth, conservative force gradients (numerical noise < 0.05 eV/Å) 
  suitable for training equivariant message-passing potentials while allowing the full 
  autonomous pipeline to run efficiently on standard multi-core workstations.
- **Production Scaling**: For ultra-high precision sub-chemical accuracy (< 0.02 eV barrier shifts), 
  parameters can be readily scaled to 5×5 supercells and 450+ eV cutoffs directly in `config.yaml`.

## Directory Setup

```bash
mkdir -p Autonomous_Pipeline_SAC
cd Autonomous_Pipeline_SAC
# Place config.yaml, pipeline_engine.py, run_ht_pipeline.py, and README.md here




#Installation

Ensure your Python environment contains ASE, GPAW with plane-wave support, PyYAML, Matplotlib, and MACE:

conda create -n sac_mlip python=3.10 -y
conda activate sac_mlip

# Install dependencies
pip install ase pyyaml matplotlib
pip install mace-torch
pip install gpaw

#Usage
1. Standard Unattended Run

Runs all phases from in-silico synthesis to publication figures using settings in config.yaml:

python -u run_ht_pipeline.py > pipeline.log 2>&1 &
tail -f pipeline.log

#2. Targeted Execution via CLI

Control which phases run using command-line arguments without modifying configuration files:

# Skip DFT sampling if catalysis.db is already populated
python run_ht_pipeline.py --skip synth dft aimd

# Run only the CI-NEB catalytic screening and figure generation
python run_ht_pipeline.py --only neb figures

# Run only model training and validation
python run_ht_pipeline.py --only train validate


#Workflow Diagram

[ config.yaml ] 
         │
         ▼
  [ Phase 0 ] In-Silico Substrate Synthesis (Pristine Slab + BFGS Pre-relaxation)
         │
         ▼
  [ Phase 1 ] Enumerate 10 Physical Motifs (M-N4, M-N3C1, ..., M-C3)
         │
         ▼
  [ Phase 2 & 3 ] High-Throughput DFT Sampling (GPAW PBE-D3 + Langevin AIMD)
         │
         ▼
  [ Phase 4 ] Export Stratified Train / Validation Datasets (mace_train.xyz, mace_val.xyz)
         │
         ▼
  [ Phase 5 ] Evaluate Foundation Model Zero-Shot Baseline (MACE-MP-0)
         │
         ▼
  [ Phase 6 ] Programmatic MACE Fine-Tuning & Error Quantization
         │
         ▼
  [ Phase 7 ] CI-NEB Catalytic Barrier Screening + Delta G(T, P)
         │
         ▼
  [ Phase 8 ] Active-Learning-Guarded Production Molecular Dynamics (20 ps)
         │
         ▼
  [ Phase 9 ] Publication-Grade Figure Generation (figures/parity_plot.png, etc.)


  ##Data Inspection

Query the generated SQLite database using standard ASE tools:


# Summary of all evaluated configurations
ase db catalysis.db

# Query specific motifs or reactive stretches
ase db catalysis.db "motif=Pt-N3,d_bond>1.5"

# Launch web GUI
ase db catalysis.db -w