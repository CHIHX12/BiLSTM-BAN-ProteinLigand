# Quick Start Guide — Drug-Protein Binding Predictor

> **Who is this for?**
> Anyone who wants to predict whether a drug molecule will bind to a protein target —
> without any programming background.

---

## What does this tool do?

It answers one simple question:

> **"Will this drug interact with this protein — YES or NO?"**

You give it:
1. A **drug** (as a text code called SMILES)
2. A **protein sequence** (a long string of letters like `MENFQKVEK...`)

It gives you back:
- **BIND** — the drug is predicted to interact with the protein
- **NO BIND** — no significant interaction predicted
- A **probability score** between 0 and 100 %

---

## Step 0 — One-time setup (install once, never again)

You need **Miniconda** (a free tool for managing Python environments).

### 0-A  Install Miniconda

Download and install from:
```
https://docs.conda.io/en/latest/miniconda.html
```
Choose the version that matches your operating system (Windows / Linux / Mac).
Use all default options during installation.

### 0-B  Open a terminal

- **Windows:** search for "Anaconda Prompt" in the Start menu and open it
- **Linux / Mac:** open the "Terminal" app

### 0-C  Navigate to the project folder

Type the following command (replace the path with where you saved this folder):

```bash
cd /path/to/this-folder
```

For example, if you downloaded it to your Desktop:

```bash
# Windows
cd C:\Users\YourName\Desktop\GCN-BILSTM-BAN

# Linux / Mac
cd ~/Desktop/GCN-BILSTM-BAN
```

### 0-D  One-click environment setup (recommended)

We provide setup scripts that automatically install everything for you.

**Linux / Mac:**
```bash
bash setup.sh
```

**Windows (Anaconda Prompt):**
```
setup.bat
```

The script will:
1. Check that Miniconda is installed
2. Create the `drugban` Python environment
3. Install all required packages (PyTorch, RDKit, DGL, etc.)
4. Verify the installation is complete

This takes **5–15 minutes** the first time (it downloads packages).
Once it finishes, you will see `Setup complete!` in the terminal.

> **Manual alternative:** If the script does not work, run this instead:
> ```bash
> conda env create -f environment.yml
> ```

### 0-E  Activate the environment

Every time you open a new terminal, run:

```bash
conda activate drugban
```

You will see `(drugban)` appear at the start of your terminal prompt.
This means the environment is active and you are ready.

---

## Step 1 — Get the drug SMILES

SMILES is a standard text code that describes a drug molecule.
Every approved drug has one.

**How to find it:**

1. Go to **https://pubchem.ncbi.nlm.nih.gov/**
2. Type the drug name in the search box (e.g. `Ibuprofen`, `Aspirin`, `Metformin`)
3. Click on the matching compound
4. Look for **"Isomeric SMILES"** or **"Canonical SMILES"** on the page
5. Copy the full SMILES text

**Example SMILES for common drugs:**

| Drug | SMILES |
|------|--------|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Ibuprofen | `CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O` |
| Metformin | `CN(C)C(=N)NC(=N)N` |
| Caffeine | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` |

---

## Step 2 — Get the protein sequence

The protein sequence is the "blueprint" of the target protein.
It is a long string of capital letters (A, C, D, E, F, ...).

**How to find it:**

1. Go to **https://www.uniprot.org/**
2. Search for the protein name (e.g. `EGFR human`, `CDK2 human`)
3. Click on the matching result (look for "Reviewed / Swiss-Prot" entries)
4. Click the **"Sequence"** tab
5. Click **"Copy sequence"**
6. Paste it — the letters only (not the line starting with `>`)

---

## Step 3 — Run the predictor (easy interactive mode)

Make sure you have activated the environment (`conda activate drugban`).
Then run:

```bash
python predict_simple.py
```

The tool will ask you two questions:
1. Paste the drug SMILES → press Enter
2. Paste the protein sequence → press Enter (for multi-line FASTA, press Enter twice)

That is all.  The result appears immediately.

**Example output:**

```
============================================================
  PREDICTION RESULT
============================================================

  Drug   : CC(=O)Oc1ccccc1C(=O)O
  Protein: MRPSGTAGAALLALLAALCPASRALEEKKVC... (length = 1210 aa)

  Binding probability  : 71.0%
  Confidence bar       : [############################----]
  Confidence level     : High

  >>> PREDICTION : BIND <<<

  Interpretation: The model predicts this drug INTERACTS with
  the protein.  This is a computational prediction — wet-lab
  validation is always recommended.
============================================================
```

The result is also saved automatically to `last_prediction.txt` in this folder.

---

## Step 4 (optional) — Batch mode: test many pairs at once

If you have a list of drugs or proteins, you can test them all at once with `predict_batch.py`.
It automatically figures out the right mode from your input files.

### Three modes — auto-detected

| Your input | Mode | What it does |
|------------|------|--------------|
| 1 drug + many proteins | **1:N** | Screens one drug against all targets |
| Many drugs + 1 protein | **N:1** | Virtual screen of all drugs against one target |
| Many drugs + many proteins | **N:M** | All combinations (N × M predictions) |

### Prepare input files

**Drug file** (one SMILES per line, `.txt` or `.smi`):

```
# my_drugs.txt
caffeine    CN1C=NC2=C1C(=O)N(C(=O)N2C)C
aspirin     CC(=O)Oc1ccccc1C(=O)O
ibuprofen   CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O
```

- Lines starting with `#` are treated as comments and ignored
- Tab-separated: first column = name, second column = SMILES
- Or just one SMILES per line (no name needed)

**Protein file** — plain text (one sequence per line, `.txt`):

```
# my_proteins.txt
CDK2    MENFQKVEKIGEGTYGVVYKARNKLTGE...
COX2    MLARALLLLCAVLASHARPALPHPGSA...
```

**Protein file** — FASTA format (`.fasta` or `.fa`):

```
>CDK2_human
MENFQKVEKIGEGTYGVVYKARNKLTGE
VVALKKIRLDTETEGVPSTAIRE
>COX2_human
MLARALLLLCAVLASHARPALPHPGSAAAQ
PTEGASQSPADSGAERAPS
```

### Run the batch predictor

```bash
# 1 drug vs many proteins (auto-detected as 1:N)
python predict_batch.py --ligands one_drug.txt --receptors proteins.fasta

# Many drugs vs 1 protein (auto-detected as N:1)
python predict_batch.py --ligands my_drugs.txt --receptors one_protein.txt

# Many drugs vs many proteins — all combinations
python predict_batch.py --ligands my_drugs.txt --receptors my_proteins.fasta

# Entire folders of files
python predict_batch.py --ligands drug_folder/ --receptors protein_folder/

# Multi-chain FASTA: join all chains into one protein complex
python predict_batch.py --ligands my_drugs.txt --receptors complex.fasta --concat_chains

# Force 1:1 paired matching (drug[1] vs protein[1], etc.)
python predict_batch.py --ligands my_drugs.txt --receptors my_proteins.txt --mode paired
```

### What happens to bad entries?

The script is forgiving:
- Invalid SMILES → **skipped** with a warning (does not crash)
- Sequence with unrecognized characters → **auto-cleaned** (non-AA letters removed)
- Too-short sequences → **skipped** with a warning

Valid entries are still processed normally.

### Output

Results are saved to a CSV file (auto-named with timestamp, e.g. `batch_results_20240506_1430.csv`).
Open it in Excel.

| Column | Meaning |
|--------|---------|
| `ligand_name` | Drug identifier |
| `receptor_name` | Protein identifier |
| `smiles` | The drug SMILES that was used |
| `protein_seq` | First 40 letters of the protein sequence |
| `binding_prob` | Binding probability (0.0 – 1.0) |
| `prediction` | `BIND` or `NO_BIND` |
| `confidence` | Very High / High / Medium / Low / Very Low |

Results are sorted from highest to lowest binding probability.
A log file is also saved alongside the CSV.

---

## Understanding the output

**Confidence levels:**

| Probability | Confidence | Meaning |
|-------------|------------|---------|
| 85 – 100 % | Very High | Strong predicted binder |
| 60 – 85 % | High | Likely binder |
| 40 – 60 % | Medium | Uncertain — could go either way |
| 20 – 40 % | Low | Weak evidence of binding |
| 0 – 20 % | Very Low | Very unlikely to bind |

---

## Model performance (on held-out test set)

The model was trained on 70 000+ drug-protein pairs from BindingDB.
Independent test results:

| Metric | Value | Plain English |
|--------|-------|---------------|
| AUROC | 0.964 | Overall ranking quality (1.0 = perfect) |
| Accuracy | 90.7 % | Correct predictions out of all predictions |
| Sensitivity | 89.0 % | Of real binders → correctly identified as BIND |
| Specificity | 93.0 % | Of real non-binders → correctly identified as NO BIND |

---

## Troubleshooting

**"conda: command not found"**
→ Miniconda is not installed or not on your PATH.
Re-install from https://docs.conda.io/en/latest/miniconda.html

**"ModuleNotFoundError: No module named 'torch'"**
→ You forgot to activate the environment.
Run: `conda activate drugban`

**"SMILES parse error" or "invalid molecule"**
→ The SMILES string is invalid.  Copy it again from PubChem carefully.
Make sure you copy the entire string without extra spaces.

**"RuntimeError: CUDA out of memory"**
→ Not enough GPU memory.  The tool will automatically fall back to CPU.
It will be slower (~30 seconds per pair) but still correct.

**The protein sequence contains unexpected characters**
→ Make sure you only paste the amino acid letters (A–Z).
Remove any numbers, spaces, or `>` header lines.
`predict_batch.py` handles this automatically — it cleans the sequence for you.

**"No valid ligands" or "No valid receptors"**
→ Check that your file format is correct.
Drug files must contain SMILES strings; protein files must contain amino acid sequences.
See the examples in Step 4 above.

---

## Need the advanced guide?

See `STEP_BY_STEP.md` for:
- Training a new model from scratch
- Running on your own dataset
- Generating 3D binding site visualizations in PyMOL
- High-throughput virtual screening (thousands of compounds)
