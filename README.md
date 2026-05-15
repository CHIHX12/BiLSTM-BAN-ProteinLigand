# BiLSTM-Powered Bilinear Attention for Protein–Ligand Prediction

> **Official implementation** of the paper published on bioRxiv:
> Chih-Yang Cheng, Yi-An Chen, Feng-Yin Li, Suyong Re (2026)
> [https://doi.org/10.64898/2026.05.10.724184](https://doi.org/10.64898/2026.05.10.724184)
>
> License: [CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## Abstract

Rapid and accurate prediction of protein–ligand bindings is essential for drug discovery. While generative AI has driven rapid advancements in structure-based approaches, sequence-based methods remain significantly faster and more cost-effective. Here, we present a weakly supervised deep learning framework integrating **graph convolutional networks (GCN)** for molecular encoding and **bidirectional long short-term memory (BiLSTM)** for protein modeling. Leveraging a **bilinear attention network (BAN)**, this model learns protein–ligand pairwise interactions without requiring three-dimensional structural supervision. Trained solely on affinity labels from the publicly available BindingDB dataset, the model classifies binders and non-binders with **AUROC of 0.96** and **AUPRC of 0.95**. The model generates interpretable attention maps that serve as a "GPS" to locate binding sites — pinpointing key contact residues confirmed by crystal structures, despite no structural training data. Our method functions as a scalable filter for giga-scale libraries with direct structural insights into the protein–ligand interface.

---

## Overview

This repository implements and compares two protein encoder architectures inside the **DrugBAN** framework for drug-target interaction (DTI) prediction:

| Model | Drug Encoder | Protein Encoder | Key Property |
|-------|-------------|-----------------|--------------|
| **CNN** (baseline) | GCN | CNN | Standard DrugBAN |
| **BiLSTM** (ours) | GCN | BiLSTM | Sequential context + attention |

### Core Idea

The model is trained **only on binding affinity labels** (does drug X bind protein Y? yes/no).
It never sees binding site coordinates during training.

After training, the **BiLSTM attention weights** reveal which protein residues the model focuses on.
When many different drug molecules consistently attend to the same residues, those residues are the predicted active site.

**Key finding:** BiLSTM attention weights spatially cluster at the true binding pocket — CNN attention scatters across the surface.

---

## Performance

Results on the **BindingDB overlap_with_mask** test split:

| Model | Best Epoch | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|-------|-----------|-------|-------|----|-------------|-------------|----------|
| CNN (baseline) | 90 | 0.9603 | 0.9485 | 0.9066 | 0.9070 | 0.9062 | 0.9067 |
| **BiLSTM (ours)** | **94** | **0.9641** | **0.9542** | **0.9118** | **0.8897** | **0.9303** | **0.9065** |

---

## How to Read Attention Colors in PyMOL

| Color | Attention Frequency | Interpretation |
|-------|--------------------|-|
| RED sticks | >75% of drugs focus here | High-confidence active site residues |
| ORANGE sticks | 50–75% | Moderate-confidence |
| YELLOW sticks | 25–50% | Supporting region |
| GRAY | <25% | Background |

The RED/ORANGE/YELLOW residues form a spatially **dense cluster in 3D** — this cluster IS the predicted binding pocket.

### Role of fpocket (optional validation)

fpocket detects geometrically druggable cavities from the **protein 3D structure alone** — no ligand, no ML.
Used here as an **independent second opinion**: if the geometric pocket (fpocket) overlaps with the attention-predicted pocket (BiLSTM RED), two independent methods agree → high confidence.

---

## Quick Start

**Have pre-trained weights? Skip to Step 3.**

```bash
# Step 1: Clone this repository
git clone https://github.com/CHIHX12/BiLSTM-BAN-ProteinLigand.git
cd BiLSTM-BAN-ProteinLigand

# Step 2: Set up the environment (one-time)
bash setup.sh
conda activate drugban

# Step 3: Generate attention vs binding site comparison (8 proteins)
python repro_comparison/run_all_proteins.py

# Step 4: Run fpocket on the 8 protein PDB files (requires fpocket installed)
# See STEP_BY_STEP.md for commands

# Step 5: Generate PyMOL pocket visualization scripts
python repro_comparison/generate_pocket_pmls.py
```

For batch drug-protein screening:
```bash
python predict_batch.py --drug examples/drugs.txt --protein examples/proteins.fasta
```

For the full walkthrough see **[QUICKSTART.md](QUICKSTART.md)** and **[STEP_BY_STEP.md](STEP_BY_STEP.md)**.

---

## Pre-trained Weights

Model checkpoints are hosted separately (GitHub LFS / Zenodo):

| Model | Checkpoint | AUROC |
|-------|-----------|-------|
| CNN baseline | `result/DrugBAN/best_model_epoch_90.pth` | 0.9603 |
| BiLSTM (ours) | `result/DrugBAN_BiLSTM/best_model_epoch_94.pth` | 0.9641 |

> Checkpoints are included in this repository under `result/`.

---

## Directory Layout

```
BiLSTM-BAN-ProteinLigand/
|
+-- ban.py / configs.py / dataloader.py    Core model code
+-- domain_adaptator.py / main.py
+-- models.py / trainer.py / utils.py
|
+-- configs/                               All experiment configs (YAML)
|   +-- DrugBAN.yaml                       CNN baseline
|   +-- DrugBAN_BiLSTM.yaml               BiLSTM model (main contribution)
|   +-- DrugBAN_*_DA.yaml                 Domain adaptation variants
|
+-- result/
|   +-- DrugBAN/best_model_epoch_90.pth   CNN best checkpoint
|   +-- DrugBAN_BiLSTM/best_model_epoch_94.pth  BiLSTM best checkpoint
|
+-- datasets/
|   +-- bindingdb/                         Training data (overlap_with_mask splits)
|
+-- P-L/                                   Protein-Ligand 3D structures (8 proteins)
|   +-- 1981-2000/{1aq1,1qfs}/
|   +-- 2001-2010/{1oxg,1t48,2qk8,3nzc}/
|   +-- 2011-2019/{5hls,6g3q}/
|
+-- repro_comparison/                      Reproducibility scripts + all results
|   +-- run_all_proteins.py               Step 3: attention frequency analysis
|   +-- generate_pocket_pmls.py           Step 5: PyMOL sphere visualization
|   +-- SUMMARY.csv                       Quantitative results table
|   +-- with_binding_site/               Results: trained WITH binding site info
|   +-- without_binding_site/            Results: trained WITHOUT binding site info
|   +-- without_binding_site_fpocket/    Results: WITHOUT + fpocket validation (FINAL)
|
+-- examples/                              Demo input files for predict_batch.py
+-- screening/                             High-throughput screening pipeline
```

---

## Requirements

See `requirements.txt` and `environment.yml` for exact versions. Key packages:

```
torch==2.2.1
dgl==2.1.0          # GCN drug encoder
dgllife==0.3.2
numpy, pandas, scikit-learn, matplotlib
yacs, tqdm
```

Optional (for Steps 4–6):
```
fpocket   -- geometric pocket detection (Linux: apt install fpocket)
PyMOL     -- 3D visualization (pymol-open-source via conda)
```

One-click environment setup:
```bash
bash setup.sh        # Linux / macOS
setup.bat            # Windows
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cheng2026bilstm,
  title   = {{BiLSTM}-Powered Bilinear Attention for Protein--Ligand Prediction},
  author  = {Cheng, Chih-Yang and Chen, Yi-An and Li, Feng-Yin and Re, Suyong},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.10.724184},
  url     = {https://www.biorxiv.org/content/10.64898/2026.05.10.724184},
  note    = {Preprint. CC-BY-NC-ND 4.0}
}
```

---

## Acknowledgements

This work builds on the [DrugBAN](https://github.com/peizhenbai/DrugBAN) framework by Bai et al. (2023).
