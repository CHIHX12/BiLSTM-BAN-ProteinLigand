#!/usr/bin/env python3
"""
predict_batch.py — Smart batch Drug-Protein Binding Predictor
=============================================================

Automatically detects mode from your input files:

  MODE 1:N  — 1 ligand  vs many receptors  (find which proteins a drug hits)
  MODE N:1  — many ligands vs 1 receptor   (virtual screen against one target)
  MODE N:M  — many ligands vs many receptors (all combinations predicted)

Supported input formats:
  .txt    one entry per line (SMILES or amino-acid sequence)
  .fasta / .fa / .fas  FASTA format (multiple entries with > headers)
  folder  reads all .txt / .fasta / .fa inside the folder

FASTA multi-chain:
  By default each >entry is treated as a separate protein.
  Use --concat_chains to join all chains in one file into one protein.

Usage examples:
  # 1 drug file + 1 protein file (auto-detects 1:N, N:1, or N:M)
  python predict_batch.py --ligands drugs.txt --receptors proteins.fasta

  # Folders (reads all files inside)
  python predict_batch.py --ligands ligand_folder/ --receptors receptor_folder/

  # Force paired mode (drug[1] vs protein[1], drug[2] vs protein[2], ...)
  python predict_batch.py --ligands drugs.txt --receptors proteins.txt --mode paired

  # Concatenate all chains in a multi-chain FASTA as one protein
  python predict_batch.py --ligands drug.smi --receptors complex.fasta --concat_chains
"""

import sys, os, re, warnings, argparse, logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Constants ────────────────────────────────────────────────────────────────
VALID_AA      = set("ACDEFGHIKLMNPQRSTVWY")
RARE_AA       = set("XUBZO")          # warn but allow
SMILES_CHARS  = set("=#@[]\\/()+-.%") # chars that only appear in SMILES
MIN_PROTEIN_LEN = 10
MAX_PROTEIN_LEN = 5000                # model truncates at 1200; warn above
MIN_SMILES_LEN  = 2

FASTA_EXTENSIONS = {".fasta", ".fa", ".fas", ".faa"}
SMILES_EXTENSIONS = {".smi", ".smiles"}
TEXT_EXTENSIONS   = {".txt"}

# ── Logging setup ────────────────────────────────────────────────────────────
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("batch")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ═════════════════════════════════════════════════════════════════════════════
# FILE CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def classify_text(text: str) -> str:
    """Return 'SMILES', 'PROTEIN', or 'UNKNOWN' for a single text entry."""
    text = text.strip()
    if not text:
        return "UNKNOWN"

    # Strong SMILES indicators: special characters not in amino-acid alphabet
    smiles_score = sum(1 for c in text if c in SMILES_CHARS)
    # SMILES always has non-alpha characters (rings, bonds, etc.)
    non_alpha = sum(1 for c in text if not c.isalpha())
    total = len(text)

    if smiles_score > 0 or non_alpha / total > 0.05:
        return "SMILES"

    # Check if it looks like a protein (only AA letters)
    upper = text.upper()
    aa_frac = sum(1 for c in upper if c in VALID_AA | RARE_AA) / len(upper)
    if aa_frac > 0.90:
        return "PROTEIN"

    return "UNKNOWN"


def classify_file(path: Path) -> str:
    """Guess whether a file contains SMILES (drugs) or protein sequences."""
    suffix = path.suffix.lower()
    if suffix in FASTA_EXTENSIONS:
        return "PROTEIN"
    if suffix in SMILES_EXTENSIONS:
        return "SMILES"

    # Read first few non-empty, non-comment, non-header lines
    samples = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(">"):
                    continue
                samples.append(line)
                if len(samples) >= 5:
                    break
    except Exception:
        return "UNKNOWN"

    if not samples:
        return "UNKNOWN"

    votes = [classify_text(s) for s in samples]
    smiles_votes = votes.count("SMILES")
    protein_votes = votes.count("PROTEIN")
    if smiles_votes > protein_votes:
        return "SMILES"
    if protein_votes > smiles_votes:
        return "PROTEIN"
    return "UNKNOWN"


# ═════════════════════════════════════════════════════════════════════════════
# PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_fasta(path: Path, concat_chains: bool = False) -> list:
    """
    Parse a FASTA file.  Returns list of dicts:
      {'name': str, 'seq': str, 'source': str}

    If concat_chains=True, all sequences in the file are joined into one entry
    named after the file stem. Useful for multi-chain protein complexes.
    """
    entries = []
    current_name = None
    current_lines = []

    def flush():
        if current_name is not None:
            seq = "".join(current_lines)
            entries.append({"name": current_name, "seq": seq, "source": str(path)})

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if line.startswith(">"):
                flush()
                current_name = line[1:].split()[0] or f"entry_{len(entries)+1}"
                current_lines = []
            else:
                # Remove stop codons, spaces, digits (line numbers)
                cleaned = "".join(c for c in line if c.isalpha())
                current_lines.append(cleaned)
    flush()

    if not entries:
        return []

    if concat_chains:
        combined_seq = "".join(e["seq"] for e in entries)
        return [{"name": path.stem, "seq": combined_seq, "source": str(path)}]

    return entries


def parse_txt(path: Path) -> list:
    """
    Parse a plain-text file: one entry per line.
    Lines starting with # are comments. Empty lines are skipped.
    Tab-separated files: first column = name, second = entry.
    Returns list of dicts: {'name': str, 'raw': str, 'source': str}
    """
    entries = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:  # utf-8-sig strips BOM
        for i, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\r\n").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                name, raw = parts[0].strip(), parts[1].strip()
            else:
                name = f"line{i}"
                raw = parts[0].strip()
            entries.append({"name": name, "raw": raw, "source": str(path)})
    return entries


def load_file(path: Path, file_type: str = "auto",
              concat_chains: bool = False) -> list:
    """
    Load a single file and return a list of entries with 'name', 'raw', 'type'.
    file_type: 'SMILES', 'PROTEIN', or 'auto' (detect from extension + content)
    """
    suffix = path.suffix.lower()
    entries = []

    if suffix in FASTA_EXTENSIONS:
        for e in parse_fasta(path, concat_chains):
            entries.append({"name": e["name"], "raw": e["seq"],
                            "type": "PROTEIN", "source": str(path)})
    else:
        for e in parse_txt(path):
            name = e["name"]
            # Prefix generic "lineN" names with file stem for clarity in folder mode
            if name.startswith("line") and name[4:].isdigit():
                name = f"{path.stem}_{name}"
            entries.append({"name": name, "raw": e["raw"],
                            "type": file_type, "source": str(path)})

    if file_type == "auto":
        detected = classify_file(path)
        for e in entries:
            if e["type"] == "auto":
                e["type"] = detected

    return entries


def load_folder(folder: Path, concat_chains: bool = False,
                hint: str = "auto") -> tuple:
    """
    Load all files in a folder.  Auto-classifies files as SMILES or PROTEIN.
    Returns (ligand_entries, receptor_entries).
    hint: 'ligands', 'receptors', or 'auto'
    """
    supported = FASTA_EXTENSIONS | SMILES_EXTENSIONS | TEXT_EXTENSIONS
    files = sorted(f for f in folder.iterdir()
                   if f.is_file() and f.suffix.lower() in supported)

    ligands, receptors = [], []
    for f in files:
        ftype = classify_file(f)
        entries = load_file(f, ftype, concat_chains)
        if ftype == "SMILES":
            ligands.extend(entries)
        elif ftype == "PROTEIN":
            receptors.extend(entries)
        # UNKNOWN files are skipped with no crash

    return ligands, receptors


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_smiles(raw: str) -> tuple:
    """Returns (cleaned_smiles_or_None, list_of_warning_strings)."""
    from rdkit import Chem
    warnings_out = []
    text = raw.strip()
    if not text:
        return None, ["Empty SMILES"]
    if len(text) < MIN_SMILES_LEN:
        return None, [f"SMILES too short ({len(text)} chars)"]

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None, [f"RDKit could not parse SMILES: '{text[:60]}'"]
    return text, warnings_out


def validate_protein(raw: str) -> tuple:
    """Returns (cleaned_sequence_or_None, list_of_warning_strings)."""
    warnings_out = []
    # Keep only letters (handles spaces, digits, asterisks)
    seq = "".join(c for c in raw.upper() if c.isalpha())

    if len(seq) < MIN_PROTEIN_LEN:
        return None, [f"Protein sequence too short ({len(seq)} aa, min={MIN_PROTEIN_LEN})"]

    rare = set(seq) & RARE_AA
    if rare:
        warnings_out.append(f"Non-standard amino acids found: {rare} "
                            f"(kept — model may handle them)")

    invalid = set(seq) - VALID_AA - RARE_AA
    if invalid:
        warnings_out.append(f"Unrecognized characters: {invalid} — removed")
        seq = "".join(c for c in seq if c in VALID_AA | RARE_AA)

    if len(seq) > MAX_PROTEIN_LEN:
        warnings_out.append(f"Sequence very long ({len(seq)} aa) — "
                            f"model truncates to first 1200 aa")

    if len(seq) < MIN_PROTEIN_LEN:
        return None, [f"After cleaning, sequence is too short ({len(seq)} aa)"]

    return seq, warnings_out


def validate_entries(entries: list, etype: str, logger) -> list:
    """
    Validate and clean a list of entries in-place.
    etype: 'SMILES' or 'PROTEIN'
    Returns list of valid entries only.
    """
    valid = []
    for e in entries:
        if etype == "SMILES":
            cleaned, warns = validate_smiles(e["raw"])
        else:
            cleaned, warns = validate_protein(e["raw"])

        for w in warns:
            logger.warning(f"  [{e['name']}] {w}")

        if cleaned is None:
            logger.warning(f"  SKIPPED [{e['name']}] from {e['source']}")
        else:
            e["cleaned"] = cleaned
            e["warnings"] = warns
            valid.append(e)

    return valid


# ═════════════════════════════════════════════════════════════════════════════
# PAIR BUILDING
# ═════════════════════════════════════════════════════════════════════════════

def build_pairs(ligands: list, receptors: list, mode: str, logger) -> list:
    """
    Build (ligand, receptor) pairs based on mode.
    mode: 'auto' | 'combinations' | 'paired'
    Returns list of dicts with all pair info.
    """
    n_lig = len(ligands)
    n_rec = len(receptors)

    # Detect mode
    if mode == "auto":
        if n_lig == 1 and n_rec > 1:
            mode = "combinations"
            logger.info(f"Auto-detected mode: 1 ligand × {n_rec} receptors (Mode 1:N)")
        elif n_lig > 1 and n_rec == 1:
            mode = "combinations"
            logger.info(f"Auto-detected mode: {n_lig} ligands × 1 receptor (Mode N:1)")
        elif n_lig == n_rec and n_lig > 1:
            mode = "combinations"
            logger.info(f"Auto-detected mode: {n_lig} ligands × {n_rec} receptors "
                        f"= {n_lig*n_rec} pairs (Mode N:M, all combinations). "
                        f"Use --mode paired for 1:1 matching.")
        else:
            mode = "combinations"
            logger.info(f"Mode: {n_lig} × {n_rec} = {n_lig*n_rec} pairs (all combinations)")

    pairs = []
    if mode == "paired":
        if n_lig != n_rec:
            logger.error(f"--mode paired requires equal counts. "
                         f"Got {n_lig} ligands vs {n_rec} receptors.")
            sys.exit(1)
        logger.info(f"Paired mode: {n_lig} pairs (ligand[i] vs receptor[i])")
        for lig, rec in zip(ligands, receptors):
            pairs.append({"lig_name": lig["name"], "smiles": lig["cleaned"],
                          "rec_name": rec["name"], "sequence": rec["cleaned"]})
    else:  # combinations
        for lig in ligands:
            for rec in receptors:
                pairs.append({"lig_name": lig["name"], "smiles": lig["cleaned"],
                              "rec_name": rec["name"], "sequence": rec["cleaned"]})

    logger.info(f"Total pairs to predict: {len(pairs)}")
    return pairs


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

def run_predictions(pairs: list, model_name: str, batch_size: int,
                    logger) -> "pd.DataFrame":
    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    from configs import get_cfg_defaults
    from models import DrugBAN
    from dataloader import DTIDataset
    from utils import graph_collate_func

    MODEL_CONFIGS = {
        "BiLSTM": {
            "cfg":   os.path.join(ROOT, "configs/DrugBAN_BiLSTM.yaml"),
            "ckpt":  os.path.join(ROOT, "result/DrugBAN_BiLSTM/best_model_epoch_94.pth"),
            "threshold": 0.1511,
        },
        "CNN": {
            "cfg":   os.path.join(ROOT, "configs/DrugBAN.yaml"),
            "ckpt":  os.path.join(ROOT, "result/DrugBAN/best_model_epoch_90.pth"),
            "threshold": 0.3100,
        },
    }

    info = MODEL_CONFIGS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model: {model_name}  |  device: {device}")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(info["cfg"])
    cfg.freeze()
    threshold = info["threshold"]

    model = DrugBAN(**cfg).to(device)
    state = torch.load(info["ckpt"], map_location=device)
    model.load_state_dict(state)
    model.eval()

    use_features = cfg.PROTEIN.get("USE_BILSTM", False)

    df_in = pd.DataFrame({
        "SMILES":  [p["smiles"]   for p in pairs],
        "Protein": [p["sequence"] for p in pairs],
        "Y": 0,
    })

    dataset = DTIDataset(
        list(range(len(df_in))), df_in,
        max_drug_nodes=290, max_protein_length=1200,
        use_features=use_features,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=graph_collate_func)

    all_probs = []
    n_batches = (len(pairs) + batch_size - 1) // batch_size
    logger.info(f"Running inference on {len(pairs)} pairs in {n_batches} batches...")

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            v_d, v_p, _ = batch
            v_d = tuple(x.to(device) for x in v_d) if isinstance(v_d, (list,tuple)) else v_d.to(device)
            v_p = tuple(x.to(device) for x in v_p) if isinstance(v_p, (list,tuple)) else v_p.to(device)
            _, _, score, _ = model(v_d, v_p, mode="eval")
            prob = torch.sigmoid(score).squeeze(-1).cpu().numpy()
            if prob.ndim == 0:
                prob = np.array([float(prob)])
            all_probs.extend(prob.tolist())
            if i % 10 == 0 or i == n_batches:
                logger.info(f"  batch {i}/{n_batches} done")

    probs = np.array(all_probs)
    labels = (probs >= threshold).astype(int)

    def confidence(p):
        if p >= 0.85: return "Very High"
        if p >= 0.60: return "High"
        if p >= 0.40: return "Medium"
        if p >= 0.20: return "Low"
        return "Very Low"

    df_out = pd.DataFrame({
        "ligand_name":   [p["lig_name"] for p in pairs],
        "receptor_name": [p["rec_name"] for p in pairs],
        "smiles":        [p["smiles"]   for p in pairs],
        "protein_seq":   [p["sequence"][:40] + "..." for p in pairs],
        "binding_prob":  [f"{p:.4f}" for p in probs],
        "prediction":    ["BIND" if l else "NO_BIND" for l in labels],
        "confidence":    [confidence(p) for p in probs],
    })

    return df_out.sort_values("binding_prob", ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Smart batch drug-protein binding predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ligands",  required=True,
                   help="Ligand file (.txt/.smi) or folder containing ligand files")
    p.add_argument("--receptors", required=True,
                   help="Receptor file (.txt/.fasta/.fa) or folder containing receptor files")
    p.add_argument("--mode", default="auto",
                   choices=["auto", "combinations", "paired"],
                   help="Pairing mode (default: auto-detect)")
    p.add_argument("--concat_chains", action="store_true",
                   help="Concatenate all chains in a FASTA file into one protein")
    p.add_argument("--model", default="BiLSTM", choices=["BiLSTM", "CNN"],
                   help="Model to use (default: BiLSTM)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: auto-named with timestamp)")
    # When run with no arguments, print friendly usage instead of cryptic error
    if len(sys.argv) == 1:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║        GCN-BiLSTM-BAN  —  Batch Prediction Tool                 ║
╚══════════════════════════════════════════════════════════════════╝

  You must provide at least --ligands and --receptors.

  ── Quick examples ────────────────────────────────────────────────

  # 1 drug vs many proteins  (put 1 SMILES in drugs.txt)
  python predict_batch.py --ligands drugs.txt --receptors proteins.fasta

  # Many drugs vs 1 protein
  python predict_batch.py --ligands drugs.txt --receptors one_protein.txt

  # Many drugs vs many proteins  (all N×M combinations)
  python predict_batch.py --ligands drugs.txt --receptors proteins.fasta

  # Use entire folders of files
  python predict_batch.py --ligands drug_folder/ --receptors protein_folder/

  # Multi-chain FASTA: merge all chains into one protein
  python predict_batch.py --ligands drugs.txt --receptors complex.fasta --concat_chains

  ── Input file formats ────────────────────────────────────────────

  Drug file (.txt or .smi) — one SMILES per line:
    caffeine    CN1C=NC2=C1C(=O)N(C(=O)N2C)C
    aspirin     CC(=O)Oc1ccccc1C(=O)O

  Protein file (.txt) — one sequence per line:
    CDK2    MENFQKVEKIGEGTYGVVYKARNKLT...

  Protein file (.fasta / .fa) — FASTA format:
    >CDK2_human
    MENFQKVEKIGEGTYGVVYKARNKLT...

  ── Output ────────────────────────────────────────────────────────

  Results saved to:  batch_results_YYYYMMDD_HHMMSS.csv  (open in Excel)
  Log saved to:      batch_results_YYYYMMDD_HHMMSS_log.txt

  ── All options ───────────────────────────────────────────────────
""")
        p.print_help()
        sys.exit(0)

    return p.parse_args()


def load_path(path_str: str, concat_chains: bool, logger,
              hint: str = "auto") -> list:
    """Load entries from file or folder."""
    p = Path(path_str)
    if not p.exists():
        logger.error(f"Path not found: {p}")
        sys.exit(1)

    if p.is_dir():
        logger.info(f"Loading folder: {p}")
        ligands, receptors = load_folder(p, concat_chains)
        if hint == "ligands":
            return ligands or receptors  # fallback
        if hint == "receptors":
            return receptors or ligands
        return ligands + receptors  # caller will re-separate
    else:
        detected_type = classify_file(p)
        logger.info(f"Loading file: {p}  (detected type: {detected_type})")
        return load_file(p, detected_type, concat_chains)


def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.output or f"batch_results_{ts}.csv"
    log_path = out_csv.replace(".csv", "_log.txt")

    logger = setup_logger(log_path)
    logger.info("=" * 60)
    logger.info("  GCN-BiLSTM-BAN  —  Batch Prediction")
    logger.info("=" * 60)

    # ── Load ligands ──────────────────────────────────────────
    logger.info(f"\nLoading LIGANDS from: {args.ligands}")
    lig_p = Path(args.ligands)
    if lig_p.is_dir():
        lig_raw, _ = load_folder(lig_p, args.concat_chains)
        if not lig_raw:
            logger.error("No ligand files found in folder. "
                         "Make sure files contain SMILES (e.g. drugs.txt, drugs.smi)")
            sys.exit(1)
    else:
        ftype = classify_file(lig_p)
        lig_raw = load_file(lig_p, ftype, args.concat_chains)

    logger.info(f"  Loaded {len(lig_raw)} raw ligand entries")
    ligands = validate_entries(lig_raw, "SMILES", logger)
    logger.info(f"  Valid ligands: {len(ligands)} / {len(lig_raw)}")

    if not ligands:
        logger.error("No valid ligands. Check your file format.")
        sys.exit(1)

    # ── Load receptors ────────────────────────────────────────
    logger.info(f"\nLoading RECEPTORS from: {args.receptors}")
    rec_p = Path(args.receptors)
    if rec_p.is_dir():
        _, rec_raw = load_folder(rec_p, args.concat_chains)
        if not rec_raw:
            logger.error("No receptor files found in folder. "
                         "Make sure files contain protein sequences (.fasta, .txt)")
            sys.exit(1)
    else:
        ftype = classify_file(rec_p)
        rec_raw = load_file(rec_p, ftype, args.concat_chains)

    logger.info(f"  Loaded {len(rec_raw)} raw receptor entries")
    receptors = validate_entries(rec_raw, "PROTEIN", logger)
    logger.info(f"  Valid receptors: {len(receptors)} / {len(rec_raw)}")

    if not receptors:
        logger.error("No valid receptors. Check your file format.")
        sys.exit(1)

    # ── Build pairs ───────────────────────────────────────────
    logger.info("")
    pairs = build_pairs(ligands, receptors, args.mode, logger)

    # ── Run predictions ───────────────────────────────────────
    logger.info("")
    df = run_predictions(pairs, args.model, args.batch_size, logger)

    # ── Save output ───────────────────────────────────────────
    df.to_csv(out_csv, index=False)

    n_bind   = (df["prediction"] == "BIND").sum()
    n_nobind = (df["prediction"] == "NO_BIND").sum()

    logger.info("")
    logger.info("=" * 60)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total pairs predicted : {len(df)}")
    logger.info(f"  BIND                  : {n_bind}  ({n_bind/len(df)*100:.1f}%)")
    logger.info(f"  NO_BIND               : {n_nobind}  ({n_nobind/len(df)*100:.1f}%)")
    logger.info("")
    logger.info("  Top 5 binding candidates:")
    for _, row in df.head(5).iterrows():
        logger.info(f"    [{row['confidence']:>9}]  {float(row['binding_prob']):.4f}  "
                    f"{row['ligand_name']} ↔ {row['receptor_name']}")
    logger.info("")
    logger.info(f"  Results saved to : {out_csv}")
    logger.info(f"  Full log saved to: {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
