# ActiveML — Spin-Aware Active Space Prediction for Quantum Computing

ML pipeline to predict CASSCF active spaces for transition metal complexes.
Replaces AutoCAS for near-term quantum hardware (VQE/ADAPT-VQE).

## Project
Part of qHPC-GREEN (NHR project, PC2 Paderborn).
Supervisor: Dr. Werner Dobrautz

## Structure
- `scripts/` — HPC generation and feature extraction scripts
- `notebooks/` — Analysis and model training notebooks
- `models/` — Saved model files

## Dataset
Available separately: https://github.com/YOUR_USERNAME/activeml-dataset

## Pipeline
1. `gen_300.py` / `gen_4d5d_metals.py` — generate CASSCF dataset
2. `extract_tier2.py` — add Tier 2 features to existing data
3. `casci_full_single.py` — orbital identification
4. `train_model.ipynb` — train Random Forest

## Status
- Phase 1: Universal mononuclear predictor (3d + 4d + 5d metals)
- Phase 2: Non-innocent ligands + reaction pathways (planned)
