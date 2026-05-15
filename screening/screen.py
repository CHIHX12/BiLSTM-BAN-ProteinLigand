#!/usr/bin/env python3
"""
HTS Drug-Protein Binding Screening Engine
==========================================
High-throughput prediction of drug-protein binding affinity.
Computes N × M pair predictions from ligand SMILES and protein sequences.

Usage:
    python screen.py [options]

Options:
    --model       bilstm (default) | cnn
    --ligands     Path to ligands file   (default: input/ligands/ligands.txt)
    --proteins    Path to proteins file  (default: input/proteins/proteins.txt)
    --output-dir  Root output folder     (default: output/)
    --batch-size  Inference batch size   (default: 32, reduce if GPU OOM)
    --top         Save top-N results     (default: 50)
    --threshold   Binding threshold      (default: 0.5)
    --no-binding-sites  Skip binding site extraction
    --skip-validation   Skip input validation (use if already validated)
    --multi-chain first|concat|error     FASTA multi-chain handling (default: first)

Example:
    python screen.py --model bilstm --top 100 --threshold 0.6
    python screen.py --model cnn --batch-size 16  # reduce batch if OOM
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Add GCN-BILSTM-BAN-bak to path ────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_PARENT  = os.path.dirname(_HERE)   # GCN-BILSTM-BAN-bak/
sys.path.insert(0, _PARENT)
sys.path.insert(0, _HERE)

warnings.filterwarnings("ignore")

# ── Model imports ──────────────────────────────────────────────────────────
from models import DrugBAN
from configs import get_cfg_defaults
from utils import integer_label_protein, encode_sequence_with_features

try:
    import dgl
    from dgllife.utils import (
        smiles_to_bigraph,
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
    )
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

# ── Preprocessing imports ──────────────────────────────────────────────────
from preprocess import InputLoader, SMILESValidator, ProteinValidator
from preprocess.smiles_validator import ValidationStatus as SMILESStatus
from preprocess.protein_validator import ValidationStatus as ProtStatus

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
MAX_DRUG_NODES     = 290
MAX_PROTEIN_LENGTH = 1200
ATOM_FEAT_DIM      = 74   # CanonicalAtomFeaturizer dim (74) + virtual-bit (1) = 75

MODEL_CONFIGS = {
    "bilstm": {
        "checkpoint": os.path.join(_PARENT, "result", "DrugBAN_BiLSTM", "best_model_epoch_94.pth"),
        "cfg_yaml":   os.path.join(_PARENT, "configs", "DrugBAN_BiLSTM.yaml"),
        "description": "GCN (drug) + BiLSTM (protein) — recommended",
    },
    "cnn": {
        "checkpoint": os.path.join(_PARENT, "result", "DrugBAN", "best_model_epoch_90.pth"),
        "cfg_yaml":   os.path.join(_PARENT, "configs", "DrugBAN.yaml"),
        "description": "GCN (drug) + CNN (protein) — baseline",
    },
}


# ─────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────
def _setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("HTS")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG and above — full details)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────
# Drug graph encoding
# ─────────────────────────────────────────────────────────────────────────
def _build_drug_graph(smiles: str) -> Optional[object]:
    """Convert SMILES to a padded DGL graph ready for GCN.  Returns None on failure."""
    if not DGL_AVAILABLE:
        raise RuntimeError("DGL / DGLLife not installed.")
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
    fc = partial(smiles_to_bigraph, add_self_loop=True)

    try:
        g = fc(smiles=smiles,
               node_featurizer=atom_featurizer,
               edge_featurizer=bond_featurizer)
        if g is None:
            return None

        actual_feats   = g.ndata.pop("h")
        n_actual       = actual_feats.shape[0]
        n_virtual      = MAX_DRUG_NODES - n_actual

        if n_virtual < 0:
            return None   # Too large (validated upstream, but guard anyway)

        # Mark real atoms: last bit = 0
        actual_feats = torch.cat(
            [actual_feats, torch.zeros(n_actual, 1)], dim=1)
        g.ndata["h"] = actual_feats

        # Pad with virtual nodes: last bit = 1
        virtual_feats = torch.cat(
            [torch.zeros(n_virtual, ATOM_FEAT_DIM),
             torch.ones(n_virtual, 1)], dim=1)
        g.add_nodes(n_virtual, {"h": virtual_feats})
        g = g.add_self_loop()
        return g
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# Protein encoding
# ─────────────────────────────────────────────────────────────────────────
def _encode_protein_bilstm(
    seq: str,
    max_length: int = MAX_PROTEIN_LENGTH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (p_idx, p_feat, p_len) as CPU tensors (unsqueezed for batch dim)."""
    p_idx, p_feat, p_len = encode_sequence_with_features(seq, max_length=max_length)
    return (
        torch.LongTensor(p_idx),    # [L]
        torch.FloatTensor(p_feat),  # [L, feature_dim]
        torch.LongTensor([p_len]),  # [1]
    )


def _encode_protein_cnn(
    seq: str,
    max_length: int = MAX_PROTEIN_LENGTH,
) -> torch.Tensor:
    """Returns integer-encoded protein tensor [L]."""
    p = integer_label_protein(seq, max_length=max_length)
    return torch.LongTensor(p)   # [L]


# ─────────────────────────────────────────────────────────────────────────
# Batch collation helpers
# ─────────────────────────────────────────────────────────────────────────
def _collate_batch(
    drug_graphs: list,
    protein_tensors: list,
    use_bilstm_protein: bool,
    device: torch.device,
) -> Tuple:
    """Stack a mini-batch and move to device."""
    bg_d = dgl.batch(drug_graphs).to(device)

    if use_bilstm_protein:
        # Each element is (p_idx[L], p_feat[L,d], p_len[1])
        idx_list  = [t[0] for t in protein_tensors]
        feat_list = [t[1] for t in protein_tensors]
        len_list  = [t[2] for t in protein_tensors]
        v_p = (
            torch.stack(idx_list).to(device),    # [B, L]
            torch.stack(feat_list).to(device),   # [B, L, d]
            torch.cat(len_list).to(device),      # [B]
        )
    else:
        v_p = torch.stack(protein_tensors).to(device)  # [B, L]

    return bg_d, v_p


# ─────────────────────────────────────────────────────────────────────────
# Binding site extraction from BAN attention
# ─────────────────────────────────────────────────────────────────────────
def _extract_binding_sites(
    att: torch.Tensor,           # [B, h_out, N_drug, L_protein]  raw logits
    actual_lengths: List[int],   # real protein lengths (no padding)
    seq_list: List[str],         # amino acid sequences (one per batch item)
    threshold_pct: float = 0.80, # top-N percentile → binding site
) -> List[List[Dict]]:
    """
    Convert BAN attention maps to per-residue binding site scores.

    Strategy:
      1. Average over BAN heads (dim 1) and drug nodes (dim 2) → [B, L]
      2. Min-max normalize per sample (0–1 range)
      3. Mask padding positions
      4. Label residues above threshold as binding sites

    Returns:
        List of lists: outer = batch items, inner = residue dicts
        Each residue dict: {pos, aa, score, is_binding_site}
        Only non-padding residues are included.
    """
    # [B, L]
    att_mean = att.mean(dim=[1, 2]).cpu().float()

    results = []
    for b_idx in range(att_mean.size(0)):
        scores = att_mean[b_idx]                # [L]
        real_len = actual_lengths[b_idx]
        seq = seq_list[b_idx]

        # Clip to actual protein length
        scores = scores[:real_len]

        # Min-max normalize → [0, 1]
        s_min, s_max = scores.min().item(), scores.max().item()
        if abs(s_max - s_min) < 1e-9:
            norm_scores = torch.zeros_like(scores)
        else:
            norm_scores = (scores - s_min) / (s_max - s_min)

        # Compute threshold value
        threshold_val = float(torch.quantile(norm_scores, threshold_pct))

        residues = []
        for pos in range(real_len):
            aa  = seq[pos] if pos < len(seq) else "X"
            sc  = float(norm_scores[pos].item())
            residues.append({
                "pos":             pos + 1,          # 1-indexed
                "aa":              aa,
                "attention_score": round(sc, 6),
                "is_binding_site": sc >= threshold_val,
            })
        results.append(residues)

    return results


# ─────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────
def load_model(model_type: str, device: torch.device, logger: logging.Logger) -> DrugBAN:
    """Load DrugBAN model from checkpoint."""
    cfg_info = MODEL_CONFIGS[model_type]
    checkpoint_path = cfg_info["checkpoint"]
    cfg_yaml_path   = cfg_info["cfg_yaml"]

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Please ensure the model has been trained and the checkpoint exists."
        )

    logger.info(f"Loading config: {cfg_yaml_path}")
    cfg = get_cfg_defaults()
    if os.path.isfile(cfg_yaml_path):
        cfg.merge_from_file(cfg_yaml_path)
    else:
        logger.warning(f"Config YAML not found at {cfg_yaml_path}. Using defaults.")

    logger.info(f"Building DrugBAN ({cfg_info['description']})...")
    model = DrugBAN(**cfg)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)

    # Checkpoints may be saved as full state dict or wrapped
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Parameters: {n_params:,}")
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────
def _write_results_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_heatmap_csv(path: str, results: List[dict],
                        drug_names: List[str], protein_names: List[str]) -> None:
    """Write a drug × protein probability matrix."""
    # Build lookup
    lookup: Dict[Tuple[str, str], float] = {}
    for r in results:
        lookup[(r["drug_name"], r["protein_name"])] = r["binding_prob"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + protein_names)
        for d in drug_names:
            row = [d] + [
                f"{lookup.get((d, p), 'N/A'):.4f}" for p in protein_names
            ]
            writer.writerow(row)


def _write_binding_sites_csv(path: str, binding_site_rows: List[dict]) -> None:
    if not binding_site_rows:
        return
    fieldnames = ["drug_name", "protein_name", "binding_prob",
                  "residue_pos", "residue_aa",
                  "attention_score", "is_binding_site"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(binding_site_rows)


def _write_summary(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ─────────────────────────────────────────────────────────────────────────
# Main screening function
# ─────────────────────────────────────────────────────────────────────────
def run_screening(args: argparse.Namespace) -> int:
    """Main screening loop. Returns exit code (0=success)."""

    # ── Output directory setup ─────────────────────────────────────────
    run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "run_log.txt")
    logger   = _setup_logging(log_path)

    logger.info("=" * 65)
    logger.info("  HTS Drug-Protein Binding Screening")
    logger.info(f"  Run ID: {run_id}")
    logger.info("=" * 65)

    global_start = time.perf_counter()

    # ── Device detection ───────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        logger.info(f"Device: GPU — {gpu_name} ({gpu_mem} GB VRAM)")
    else:
        device = torch.device("cpu")
        logger.warning(
            "GPU not detected. Running on CPU.\n"
            "  ⚠  CPU mode is significantly slower than GPU.\n"
            "  ⚠  Estimated time: ~1–3 min per 1,000 pairs on a modern CPU.\n"
            "  ⚠  For large screens (>10,000 pairs), GPU is strongly recommended."
        )

    logger.info(f"Model:      {args.model}  ({MODEL_CONFIGS[args.model]['description']})")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Threshold:  {args.threshold}")
    logger.info(f"Top-N:      {args.top}")
    logger.info(f"Binding sites: {'Enabled' if not args.no_binding_sites else 'Disabled'}")

    # ── Load and validate inputs ───────────────────────────────────────
    loader = InputLoader()

    logger.info(f"\nLoading ligands:  {args.ligands}")
    ligand_entries = loader.load_ligands(args.ligands)
    logger.info(f"Loading proteins: {args.proteins}")
    protein_entries = loader.load_proteins(args.proteins)
    logger.info(f"Raw entries — Ligands: {len(ligand_entries)}, Proteins: {len(protein_entries)}")

    # Validation
    if not args.skip_validation:
        logger.info("\nValidating inputs...")
        t0 = time.perf_counter()

        smiles_validator  = SMILESValidator()
        protein_validator = ProteinValidator(multi_chain_mode=args.multi_chain)

        smiles_records  = smiles_validator.validate(ligand_entries)
        protein_records = protein_validator.validate(protein_entries)

        smiles_errors  = [r for r in smiles_records  if not r.is_usable]
        protein_errors = [r for r in protein_records if not r.is_usable]

        if smiles_errors:
            logger.warning(f"  {len(smiles_errors)} ligand(s) will be SKIPPED (invalid SMILES):")
            for r in smiles_errors:
                logger.warning(f"    ✗ {r.original_name}: {'; '.join(r.errors)}")

        if protein_errors:
            logger.warning(f"  {len(protein_errors)} protein(s) will be SKIPPED (invalid sequence):")
            for r in protein_errors:
                logger.warning(f"    ✗ {r.original_name}: {'; '.join(r.errors)}")

        # Keep only usable entries (with cleaned data)
        valid_ligands  = [(r.original_name, r.clean_smiles)
                          for r in smiles_records if r.is_usable]
        valid_proteins = [(r.original_name, r.clean_sequence)
                          for r in protein_records if r.is_usable]

        t_val = time.perf_counter() - t0
        logger.info(
            f"  Validation done ({t_val:.1f}s). "
            f"Usable — Ligands: {len(valid_ligands)}, Proteins: {len(valid_proteins)}"
        )
    else:
        logger.info("Skipping validation (--skip-validation).")
        valid_ligands  = ligand_entries
        valid_proteins = protein_entries

    if not valid_ligands:
        logger.error("No valid ligands after validation. Aborting.")
        return 1
    if not valid_proteins:
        logger.error("No valid proteins after validation. Aborting.")
        return 1

    # ── Pre-compute all encodings ──────────────────────────────────────
    logger.info(f"\nPre-computing drug graph encodings ({len(valid_ligands)} ligands)...")
    t0 = time.perf_counter()

    drug_names:  List[str]         = []
    drug_graphs: List             = []
    skip_drugs: List[str]          = []

    for name, smiles in valid_ligands:
        g = _build_drug_graph(smiles)
        if g is None:
            logger.warning(f"  ✗ Cannot build graph for {name} (SMILES: {smiles}). Skipping.")
            skip_drugs.append(name)
        else:
            drug_names.append(name)
            drug_graphs.append(g)

    logger.info(f"  Drug graphs: {len(drug_graphs)} built, {len(skip_drugs)} failed. ({time.perf_counter()-t0:.1f}s)")

    logger.info(f"\nPre-computing protein encodings ({len(valid_proteins)} proteins)...")
    t0 = time.perf_counter()

    use_bilstm = (args.model == "bilstm")
    protein_names:   List[str]   = []
    protein_tensors: List        = []
    protein_seqs:    List[str]   = []   # For binding site AA mapping
    protein_lengths: List[int]   = []   # Actual lengths (not padded)

    for name, seq in valid_proteins:
        # Trim to max_length for length tracking
        real_len = min(len(seq), MAX_PROTEIN_LENGTH)
        if use_bilstm:
            tens = _encode_protein_bilstm(seq, MAX_PROTEIN_LENGTH)
            # tens[2] contains actual length (capped at MAX_PROTEIN_LENGTH)
            act_len = int(tens[2].item())
        else:
            tens    = _encode_protein_cnn(seq, MAX_PROTEIN_LENGTH)
            act_len = real_len

        protein_names.append(name)
        protein_tensors.append(tens)
        protein_seqs.append(seq[:real_len])   # trimmed sequence
        protein_lengths.append(act_len)

    logger.info(f"  Protein tensors: {len(protein_tensors)} encoded. ({time.perf_counter()-t0:.1f}s)")

    # ── Pair generation and inference ─────────────────────────────────
    total_pairs = len(drug_names) * len(protein_names)
    logger.info(f"\nStarting inference: {len(drug_names)} × {len(protein_names)} = {total_pairs:,} pairs")

    # Rough time estimate
    est_sec = total_pairs / (80 if device.type == "cuda" else 8)  # rough pairs/sec
    est_min = est_sec / 60
    if est_min > 1:
        logger.info(f"  Estimated time: ~{est_min:.0f} min (rough estimate)")

    # Load model
    logger.info("\nLoading model...")
    t0 = time.perf_counter()
    model, cfg = load_model(args.model, device, logger)
    logger.info(f"  Model ready. ({time.perf_counter()-t0:.1f}s)")

    # Results storage
    all_results:       List[dict] = []
    binding_site_rows: List[dict] = []

    # Build flat pair list: [(drug_idx, protein_idx), ...]
    # Iterate drug-outer, protein-inner → better cache locality for protein tensors
    pair_list = [(di, pi)
                 for di in range(len(drug_names))
                 for pi in range(len(protein_names))]

    batch_size = args.batch_size
    n_batches  = (len(pair_list) + batch_size - 1) // batch_size

    logger.info(f"  {n_batches} batches × {batch_size} pairs each")

    t_infer_start = time.perf_counter()
    processed = 0

    with torch.no_grad():
        for batch_idx in range(n_batches):
            batch_pairs = pair_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            # Gather batch data
            b_drug_graphs   = [drug_graphs[di]       for di, _  in batch_pairs]
            b_protein_tens  = [protein_tensors[pi]   for _,  pi in batch_pairs]
            b_drug_names    = [drug_names[di]         for di, _  in batch_pairs]
            b_protein_names = [protein_names[pi]      for _,  pi in batch_pairs]
            b_protein_seqs  = [protein_seqs[pi]       for _,  pi in batch_pairs]
            b_protein_lens  = [protein_lengths[pi]    for _,  pi in batch_pairs]

            # Collate and send to device
            try:
                bg_d, v_p = _collate_batch(b_drug_graphs, b_protein_tens,
                                           use_bilstm, device)
            except Exception as e:
                logger.warning(f"  Batch {batch_idx+1}/{n_batches} collation error: {e}. Skipping batch.")
                continue

            # Forward pass
            try:
                outputs = model(bg_d, v_p, mode="eval")
                _, _, score, att = outputs   # [B,1] or [B,2], [B,h,N,L]
            except Exception as e:
                logger.warning(f"  Batch {batch_idx+1}/{n_batches} inference error: {e}. Skipping batch.")
                continue

            # Convert scores to probabilities
            n_class = cfg["DECODER"]["BINARY"]
            if n_class == 1:
                # Binary cross-entropy mode: sigmoid
                prob = torch.sigmoid(score).squeeze(-1).cpu().float()   # [B]
            else:
                # Cross-entropy mode: softmax, take class-1 probability
                prob = torch.softmax(score, dim=1)[:, 1].cpu().float()  # [B]

            prob_np = prob.numpy()

            # Record results
            for i, (di, pi) in enumerate(batch_pairs):
                p = float(prob_np[i])
                all_results.append({
                    "rank":          0,  # filled later
                    "drug_name":     b_drug_names[i],
                    "protein_name":  b_protein_names[i],
                    "binding_prob":  round(p, 6),
                    "bind_predict":  "BIND" if p >= args.threshold else "NO_BIND",
                    "confidence":    _confidence_label(p),
                })

            # Binding site extraction (only for BIND predictions above threshold)
            if not args.no_binding_sites and att is not None:
                bind_mask  = prob_np >= args.threshold
                bind_idxs  = [i for i, m in enumerate(bind_mask) if m]

                if bind_idxs:
                    att_bind = att[bind_idxs]           # [K, h, N, L]
                    seqs_bind = [b_protein_seqs[i]  for i in bind_idxs]
                    lens_bind = [b_protein_lens[i]  for i in bind_idxs]

                    residue_data = _extract_binding_sites(
                        att_bind, lens_bind, seqs_bind,
                        threshold_pct=0.80
                    )

                    for k_idx, orig_i in enumerate(bind_idxs):
                        d_name = b_drug_names[orig_i]
                        p_name = b_protein_names[orig_i]
                        p_val  = float(prob_np[orig_i])
                        for res in residue_data[k_idx]:
                            binding_site_rows.append({
                                "drug_name":       d_name,
                                "protein_name":    p_name,
                                "binding_prob":    round(p_val, 6),
                                "residue_pos":     res["pos"],
                                "residue_aa":      res["aa"],
                                "attention_score": res["attention_score"],
                                "is_binding_site": res["is_binding_site"],
                            })

            processed += len(batch_pairs)

            # Progress log every ~10% or at least every 50 batches
            log_every = max(1, n_batches // 10)
            if (batch_idx + 1) % log_every == 0 or batch_idx == n_batches - 1:
                elapsed  = time.perf_counter() - t_infer_start
                pct      = processed / total_pairs * 100
                rate     = processed / elapsed if elapsed > 0 else 0
                eta_sec  = (total_pairs - processed) / rate if rate > 0 else 0
                logger.info(
                    f"  [{batch_idx+1:>5}/{n_batches}]  "
                    f"{processed:>7,}/{total_pairs:,} pairs  "
                    f"({pct:5.1f}%)  "
                    f"{rate:5.0f} pairs/s  "
                    f"ETA: {_fmt_seconds(eta_sec)}"
                )

    t_infer = time.perf_counter() - t_infer_start
    logger.info(f"\nInference complete. {len(all_results):,} pairs in {_fmt_seconds(t_infer)}")

    # ── Sort and rank ──────────────────────────────────────────────────
    all_results.sort(key=lambda x: x["binding_prob"], reverse=True)
    for rank_idx, row in enumerate(all_results, start=1):
        row["rank"] = rank_idx

    # ── Write outputs ──────────────────────────────────────────────────
    logger.info(f"\nSaving results to: {output_dir}")

    # Full results CSV
    full_csv_path = os.path.join(output_dir, "results_full.csv")
    _write_results_csv(full_csv_path, all_results)
    logger.info(f"  Full results: {full_csv_path}  ({len(all_results):,} rows)")

    # Top-N CSV
    top_n_results   = all_results[:args.top]
    top_csv_path    = os.path.join(output_dir, f"results_top{args.top}.csv")
    _write_results_csv(top_csv_path, top_n_results)
    logger.info(f"  Top {args.top}: {top_csv_path}")

    # Heatmap CSV
    heatmap_path = os.path.join(output_dir, "heatmap.csv")
    _write_heatmap_csv(heatmap_path, all_results, drug_names, protein_names)
    logger.info(f"  Heatmap: {heatmap_path}")

    # Binding sites CSV
    if binding_site_rows:
        bs_path = os.path.join(output_dir, "binding_sites.csv")
        _write_binding_sites_csv(bs_path, binding_site_rows)
        # Summary stats
        n_bind_pairs = len({(r["drug_name"], r["protein_name"])
                            for r in binding_site_rows})
        n_sites      = sum(1 for r in binding_site_rows if r["is_binding_site"])
        logger.info(f"  Binding sites: {bs_path}  ({n_bind_pairs} pairs, {n_sites:,} predicted sites)")
    else:
        logger.info("  Binding sites: none extracted (no pairs above threshold).")

    # Summary report
    global_elapsed = time.perf_counter() - global_start
    n_bind    = sum(1 for r in all_results if r["bind_predict"] == "BIND")
    n_nobind  = len(all_results) - n_bind
    n_high    = sum(1 for r in all_results if r["binding_prob"] >= 0.9)

    # Top ligands by avg prob
    d_probs: Dict[str, List[float]] = {}
    for r in all_results:
        d_probs.setdefault(r["drug_name"], []).append(r["binding_prob"])
    top_drugs = sorted(d_probs.items(),
                       key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)[:5]

    # Top proteins by bind count
    p_counts: Dict[str, int] = {}
    for r in all_results:
        if r["bind_predict"] == "BIND":
            p_counts[r["protein_name"]] = p_counts.get(r["protein_name"], 0) + 1
    top_proteins = sorted(p_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]

    summary = _build_summary(
        run_id=run_id,
        model_type=args.model,
        ligand_count=len(drug_names),
        protein_count=len(protein_names),
        total_pairs=len(all_results),
        n_bind=n_bind,
        n_nobind=n_nobind,
        n_high=n_high,
        threshold=args.threshold,
        top_drugs=top_drugs,
        top_proteins=top_proteins,
        elapsed=global_elapsed,
        output_dir=output_dir,
    )

    summary_path = os.path.join(output_dir, "run_summary.txt")
    _write_summary(summary_path, summary)

    print("\n" + summary)
    logger.info(f"\n  Summary saved to: {summary_path}")
    logger.info(f"  Total elapsed: {_fmt_seconds(global_elapsed)}")
    logger.info("  Done.")

    return 0


# ─────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────
def _confidence_label(prob: float) -> str:
    if prob >= 0.90:   return "HIGH"
    if prob >= 0.70:   return "MEDIUM"
    if prob >= 0.50:   return "LOW"
    return "VERY_LOW"


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, mn = divmod(m, 60)
    return f"{h}h {mn:02d}m {sec:02d}s"


def _build_summary(
    run_id, model_type, ligand_count, protein_count,
    total_pairs, n_bind, n_nobind, n_high, threshold,
    top_drugs, top_proteins, elapsed, output_dir
) -> str:
    bind_pct = n_bind / total_pairs * 100 if total_pairs else 0
    lines = [
        "=" * 65,
        "  HTS Drug-Protein Binding Screening — Run Summary",
        "=" * 65,
        f"  Run ID:       {run_id}",
        f"  Model:        {model_type.upper()} ({MODEL_CONFIGS[model_type]['description']})",
        f"  Threshold:    {threshold}",
        f"  Elapsed time: {_fmt_seconds(elapsed)}",
        "",
        "  Input Statistics",
        "  ─────────────────────────────────────",
        f"  Ligands:      {ligand_count}",
        f"  Proteins:     {protein_count}",
        f"  Total pairs:  {total_pairs:,}",
        "",
        "  Prediction Results",
        "  ─────────────────────────────────────",
        f"  BIND (≥{threshold}):     {n_bind:>8,}  ({bind_pct:.1f}%)",
        f"  NO_BIND (<{threshold}):  {n_nobind:>8,}  ({100-bind_pct:.1f}%)",
        f"  High confidence (≥0.9): {n_high:>6,}",
    ]

    if top_drugs:
        lines += ["", "  Top 5 Ligands (by mean binding probability)"]
        lines.append("  ─────────────────────────────────────")
        for rank, (name, probs) in enumerate(top_drugs, 1):
            avg = sum(probs) / len(probs)
            lines.append(f"  {rank}. {name:<30}  avg_prob={avg:.4f}")

    if top_proteins:
        lines += ["", "  Top 5 Proteins (by number of BIND predictions)"]
        lines.append("  ─────────────────────────────────────")
        for rank, (name, cnt) in enumerate(top_proteins, 1):
            lines.append(f"  {rank}. {name:<30}  bind_count={cnt}")

    lines += [
        "",
        "  Output Files",
        "  ─────────────────────────────────────",
        f"  {output_dir}/results_full.csv    ← All predictions",
        f"  {output_dir}/results_top*.csv    ← Top-N ranked",
        f"  {output_dir}/heatmap.csv         ← Matrix for Excel heatmap",
        f"  {output_dir}/binding_sites.csv   ← Predicted binding residues",
        f"  {output_dir}/run_log.txt         ← Full execution log",
        "=" * 65,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HTS Drug-Protein Binding Screening",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",
                   choices=["bilstm", "cnn"],
                   default="bilstm",
                   help="Protein encoder: bilstm (recommended) or cnn (baseline)")
    p.add_argument("--ligands",
                   default=os.path.join(_HERE, "input", "ligands", "ligands.txt"),
                   help="Path to ligands.txt")
    p.add_argument("--proteins",
                   default=os.path.join(_HERE, "input", "proteins", "proteins.txt"),
                   help="Path to proteins.txt")
    p.add_argument("--output-dir",
                   default=os.path.join(_HERE, "output"),
                   help="Root output directory (timestamped subfolder created automatically)")
    p.add_argument("--batch-size",
                   type=int,
                   default=32,
                   help="Inference batch size. Reduce to 8–16 if GPU out-of-memory")
    p.add_argument("--top",
                   type=int,
                   default=50,
                   help="Save top-N results in results_top*.csv")
    p.add_argument("--threshold",
                   type=float,
                   default=0.5,
                   help="Probability threshold for BIND/NO_BIND classification")
    p.add_argument("--no-binding-sites",
                   action="store_true",
                   help="Skip binding site (attention) extraction. Speeds up screening.")
    p.add_argument("--skip-validation",
                   action="store_true",
                   help="Skip input validation (use only if validate_inputs.py already passed)")
    p.add_argument("--multi-chain",
                   choices=["first", "concat", "error"],
                   default="first",
                   help="How to handle multi-chain FASTA proteins")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Validate required tool availability
    if not DGL_AVAILABLE:
        print("[ERROR] DGL / DGLLife is required. Install with:")
        print("  conda install -c dglteam dgl")
        print("  pip install dgllife")
        sys.exit(3)

    sys.exit(run_screening(args))
