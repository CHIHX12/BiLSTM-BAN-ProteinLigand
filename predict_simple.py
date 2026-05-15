#!/usr/bin/env python3
"""
predict_simple.py  —  Beginner-friendly prediction tool
========================================================

PURPOSE
-------
Answer one question:  "Will this drug bind to this protein?"

INPUT
-----
  - One drug   (as a SMILES string  — a text code for the molecule)
  - One protein (as an amino acid sequence — the letters A, C, D, E, F, G, ...)

OUTPUT
------
  BIND     → the model predicts the drug DOES interact with this protein
  NO BIND  → the model predicts NO interaction

HOW TO RUN
----------
  python predict_simple.py

  The script will ask you to paste the drug SMILES and protein sequence
  interactively.  No command-line flags needed.

WHERE TO GET SMILES
-------------------
  1. Go to https://pubchem.ncbi.nlm.nih.gov/
  2. Search your drug name (e.g. "Ibuprofen")
  3. On the compound page, look for "Isomeric SMILES" or "Canonical SMILES"
  4. Copy the whole SMILES string and paste it here

WHERE TO GET PROTEIN SEQUENCE
------------------------------
  1. Go to https://www.uniprot.org/
  2. Search your protein name (e.g. "EGFR human")
  3. On the protein page, click "Sequence" tab
  4. Copy the amino acid sequence (the long string of letters)
  5. Paste it here — do NOT include the FASTA header line (the line starting with >)
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")


def get_smiles(prompt: str, example: str) -> str:
    """Ask the user for a SMILES string.  Keep all characters — SMILES uses (, ), =, #, digits, etc."""
    print()
    print(prompt)
    print(f"  Example: {example[:80]}{'...' if len(example) > 80 else ''}")
    print()
    value = input("  Paste here and press Enter: ").strip()
    # Only remove leading/trailing whitespace and collapse internal newlines
    return "".join(value.split())   # joins any multi-line paste into one string


def get_protein(prompt: str, example: str) -> str:
    """Ask the user for a protein sequence.
    Accepts single-line paste OR multi-line FASTA.
    User signals end of multi-line input by pressing Enter on an empty line.
    """
    print()
    print(prompt)
    print(f"  Example: {example[:80]}{'...' if len(example) > 80 else ''}")
    print()
    print("  Paste the sequence below, then press Enter (or Enter twice for multi-line):")
    collected = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            # Empty line = user finished multi-line input
            if collected:
                break
            continue
        collected.append(line)
        # If this looks like a single-line sequence (not a FASTA header), stop after first line
        if collected and not collected[0].startswith(">") and len(collected) == 1:
            # Peek: if it's long enough to be a full sequence, accept immediately
            candidate = "".join(c for c in collected[0] if c.isalpha())
            if len(candidate) >= 10:
                break
    # Drop FASTA header lines, keep only letters
    seq_lines = [ln for ln in collected if not ln.startswith(">")]
    return "".join(c for c in "".join(seq_lines) if c.isalpha())


def run_prediction(smiles: str, protein: str) -> None:
    """Load model and print prediction result."""
    print()
    print("=" * 60)
    print("  Loading model ... (first run may take ~10 seconds)")
    print("=" * 60)

    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader

    ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, ROOT)

    from configs import get_cfg_defaults
    from models import DrugBAN
    from dataloader import DTIDataset
    from utils import graph_collate_func

    # Always use the best model (BiLSTM)
    cfg_path   = os.path.join(ROOT, "configs/DrugBAN_BiLSTM.yaml")
    ckpt_path  = os.path.join(ROOT, "result/DrugBAN_BiLSTM/best_model_epoch_94.pth")
    threshold  = 0.1511   # optimised on validation set (AUROC 0.9641)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    model = DrugBAN(**cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    use_features = cfg.PROTEIN.get("USE_BILSTM", False)

    df = pd.DataFrame([{"SMILES": smiles, "Protein": protein, "Y": 0}])
    dataset = DTIDataset(
        list(range(len(df))),
        df,
        max_drug_nodes=290,
        max_protein_length=1200,
        use_features=use_features,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=graph_collate_func,
    )

    with torch.no_grad():
        for batch in loader:
            v_d, v_p, _ = batch
            if isinstance(v_d, (list, tuple)):
                v_d = tuple(x.to(device) for x in v_d)
            else:
                v_d = v_d.to(device)
            if isinstance(v_p, (list, tuple)):
                v_p = tuple(x.to(device) for x in v_p)
            else:
                v_p = v_p.to(device)

            _, _, score, attn = model(v_d, v_p, mode="eval")
            prob = float(torch.sigmoid(score).squeeze().cpu())

    label = "BIND" if prob >= threshold else "NO BIND"

    # Confidence description
    if prob >= 0.85:
        confidence = "Very High"
    elif prob >= 0.60:
        confidence = "High"
    elif prob >= 0.40:
        confidence = "Medium"
    elif prob >= 0.20:
        confidence = "Low"
    else:
        confidence = "Very Low"

    bar_filled = int(prob * 40)
    bar = "[" + "#" * bar_filled + "-" * (40 - bar_filled) + "]"

    print()
    print("=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print()
    print(f"  Drug   : {smiles[:60]}{'...' if len(smiles) > 60 else ''}")
    print(f"  Protein: {protein[:40]}... (length = {len(protein)} aa)")
    print()
    print(f"  Binding probability  : {prob:.1%}")
    print(f"  Confidence bar       : {bar}")
    print(f"  Confidence level     : {confidence}")
    print()
    print(f"  >>> PREDICTION : {label} <<<")
    print()

    if label == "BIND":
        print("  Interpretation: The model predicts this drug INTERACTS with")
        print("  the protein.  Probability above the decision threshold of")
        print(f"  {threshold:.0%}.  This is a computational prediction — wet-lab")
        print("  validation is always recommended.")
    else:
        print("  Interpretation: The model predicts NO significant interaction.")
        print(f"  Probability below the decision threshold of {threshold:.0%}.")

    print()
    print("=" * 60)
    print()

    # Save result
    out_path = os.path.join(ROOT, "last_prediction.txt")
    with open(out_path, "w") as f:
        f.write(f"Drug SMILES : {smiles}\n")
        f.write(f"Protein seq : {protein[:40]}... (len={len(protein)})\n")
        f.write(f"Probability : {prob:.4f}\n")
        f.write(f"Result      : {label}\n")
        f.write(f"Confidence  : {confidence}\n")
    print(f"  Result saved to: {out_path}")
    print()


def main():
    print()
    print("=" * 60)
    print("  Drug-Protein Binding Predictor")
    print("  GCN-BiLSTM-BAN  |  Simple Mode")
    print("=" * 60)
    print()
    print("  This tool predicts whether a drug molecule will BIND")
    print("  to a target protein.")
    print()
    print("  You need two pieces of information:")
    print("    1. Drug SMILES  (get from https://pubchem.ncbi.nlm.nih.gov/)")
    print("    2. Protein sequence  (get from https://www.uniprot.org/)")
    print()

    smiles = get_smiles(
        "STEP 1 — Paste the drug SMILES string:",
        "CC(=O)Oc1ccccc1C(=O)O",       # Aspirin
    )
    if not smiles:
        print("ERROR: Drug SMILES cannot be empty.")
        sys.exit(1)

    protein = get_protein(
        "STEP 2 — Paste the protein amino acid sequence (letters only, no > header):",
        "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALK...",
    )
    if not protein:
        print("ERROR: Protein sequence cannot be empty.")
        sys.exit(1)

    # Basic sanity checks
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid  = set(protein.upper()) - valid_aa
    if invalid:
        print(f"WARNING: Unexpected characters in protein sequence: {invalid}")
        print("         Standard amino acid letters are: A C D E F G H I K L M N P Q R S T V W Y")
        ans = input("         Continue anyway? (y/n): ").strip().lower()
        if ans != "y":
            sys.exit(1)

    run_prediction(smiles, protein.upper())


if __name__ == "__main__":
    main()
