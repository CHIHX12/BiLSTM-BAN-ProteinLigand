# Step-by-Step Reproducibility Guide

Run all commands from the `GCN-BILSTM-BAN/` root directory.

---

## Step 1: Train the models (optional -- skip if using pre-trained weights)

### CNN baseline (GCN + CNN)
```bash
python main.py --cfg configs/DrugBAN.yaml
# Checkpoint saved to: result/DrugBAN/best_model_epoch_*.pth
```

### BiLSTM model (GCN + BiLSTM)
```bash
python main.py --cfg configs/DrugBAN_BiLSTM.yaml
# Checkpoint saved to: result/DrugBAN_BiLSTM/best_model_epoch_*.pth
```

Pre-trained weights already included:
- `result/DrugBAN/best_model_epoch_90.pth`       (CNN, epoch 90)
- `result/DrugBAN_BiLSTM/best_model_epoch_94.pth` (BiLSTM, epoch 94)

---

## Step 2: (Already done) Verify training results

```bash
# View quantitative metrics
cat result/DrugBAN/test_markdowntable.txt        # CNN test results
cat result/DrugBAN_BiLSTM/test_markdowntable.txt # BiLSTM test results
```

---

## Step 3: Generate attention frequency analysis (8 proteins)

This script loads both models, runs inference on 8 proteins, and computes
which residues receive high attention. Output goes to
`repro_comparison/without_binding_site/`.

```bash
python repro_comparison/run_all_proteins.py
```

Expected output per protein (in `repro_comparison/without_binding_site/{PDB}/`):
- `{PDB}_comparison.png`    -- attention frequency heatmap (BiLSTM vs CNN)
- `{PDB}_BiLSTM_predict.pml` -- PyMOL coloring: RED/ORANGE/YELLOW = attention clusters
- `{PDB}_CNN_predict.pml`

Quantitative summary: `repro_comparison/SUMMARY.csv`

**How to interpret `*_predict.pml` directly (no fpocket needed):**
Open in PyMOL and look for a **spatially dense cluster of RED/ORANGE/YELLOW sticks**.
That cluster is the predicted binding pocket — identified from affinity labels alone.
BiLSTM produces one compact cluster near the true pocket; CNN scatters residues globally.

> Pre-computed results already exist in `without_binding_site/` — Step 3 only needs
> re-running if you retrain the model or add new proteins.

---

## Step 4: Run fpocket on the 8 proteins (optional — results pre-computed)

fpocket detects geometrically druggable cavities from protein **3D structure only**
(no ligand, no ML, fully independent of the model).
It is used here as **independent validation**: does the geometric pocket overlap
with the BiLSTM attention cluster?

```bash
# Install fpocket if not present
# https://github.com/Discngine/fpocket

# Run for each protein (PDB files are in repro_comparison/without_binding_site/{PDB}/)
for PDB in 1AQ1 1OXG 1QFS 1T48 2QK8 3NZC 5HLS 6G3Q; do
  pdb_lower=$(echo $PDB | tr '[:upper:]' '[:lower:]')
  fpocket -f repro_comparison/without_binding_site/$PDB/${pdb_lower}_protein.pdb \
          -o repro_comparison/without_binding_site_fpocket/$PDB/
done
```

Note: fpocket results are already pre-computed in
`repro_comparison/without_binding_site_fpocket/`.
Each `{PDB}_{model}_fpocket.pml` contains the residue selections.

---

## Step 5: Generate PyMOL pocket sphere visualization

Reads `*_fpocket.pml` files and generates enhanced `*_pocket.pml` files with:
- DBSCAN spatial clustering of attention residues
- Magenta consensus sphere (attention in fpocket overlap)
- Gold/Red spheres per spatial cluster
- Blue reference sphere (fpocket center)

```bash
python repro_comparison/generate_pocket_pmls.py
```

Output: `repro_comparison/without_binding_site_fpocket/{PDB}/{PDB}_{model}_pocket.pml`

---

## Step 6: Visualize in PyMOL (Windows or Linux)

1. Copy the `repro_comparison/without_binding_site_fpocket/` folder to Windows
2. Open PyMOL
3. File -> Run Script -> select `{PDB}_BiLSTM_pocket.pml`

**What you see:**

| Color | Meaning |
|-------|---------|
| MAGENTA sticks | Consensus residues (BiLSTM RED + fpocket) |
| RED sticks | BiLSTM attention >75% |
| ORANGE sticks | BiLSTM attention 50-75% |
| YELLOW sticks | BiLSTM attention 25-50% |
| BLUE sticks | fpocket only |
| CYAN sticks | Ligand (spatial reference) |
| MAGENTA sphere | Consensus centroid (near pocket by definition) |
| GOLD sphere(s) | BiLSTM attention clusters (should overlap BLUE) |
| BLUE sphere | fpocket reference center |

**Key comparison:**
- BiLSTM: GOLD sphere overlaps BLUE sphere
- CNN: RED sphere far from BLUE sphere

---

---

## Direct Prediction (no re-training needed)

### Mode A: Single drug-protein pair

```bash
python predict.py \
  --model BiLSTM \
  --drug "CCOc1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1" \
  --protein "MENFQKVEKIGE..." \
  --output result.csv
```

Output:
```
Binding probability : 0.5962
Predicted label     : BIND  (threshold 0.5)
Top-10 attention residues: [262, 100, 261, 110, 111, ...]
```

### Mode B: Batch prediction from CSV

Prepare `input.csv`:
```
SMILES,Protein,Y
CCO,MKVL...,0
CC(=O)Oc1ccccc1C(=O)O,MENFQ...,0
```

```bash
python predict.py --model BiLSTM --input input.csv --output predictions.csv
```

### Mode C: High-throughput screening (drug discovery)

**1 drug vs many protein targets** (find which receptors a drug hits):
```bash
# proteins.txt: one amino acid sequence per line
python predict.py \
  --model BiLSTM --screen \
  --drug "CCOc1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1" \
  --protein_list proteins.txt \
  --output screening_results.csv
```

**1 protein vs many drug candidates** (virtual screening for a specific receptor):
```bash
# drugs.txt: one SMILES per line
python predict.py \
  --model BiLSTM --screen \
  --protein "MENFQKVEKIGE..." \
  --drug_list drugs.txt \
  --output virtual_screen.csv
```

Results are **sorted by binding probability** (highest first).  
Use this for rapid lead compound identification.

---

## Adding a New Protein

1. Add protein PDB + ligand SDF to `P-L/{year}/{pdb_lower}/`
2. Add protein sequence + drug pairs to `datasets/bindingdb/overlap_with_mask/`
3. Add the protein ID to `PROTEINS` dict in `repro_comparison/run_all_proteins.py`
4. Run Steps 3-6

---

## Configs Reference

| Config file | Description |
|-------------|-------------|
| `DrugBAN.yaml` | CNN baseline (GCN drug + CNN protein) |
| `DrugBAN_BiLSTM.yaml` | BiLSTM model (GCN drug + BiLSTM protein) |
| `DrugBAN_BiLSTM_DA.yaml` | BiLSTM + domain adaptation |
| `DrugBAN_BiLSTM_Non_DA.yaml` | BiLSTM without domain adaptation |
| `DrugBAN_DA.yaml` | CNN + domain adaptation |

All configs use `datasets/bindingdb` as the data source.
