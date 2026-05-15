#!/usr/bin/env python3
"""
cnn_saliency_5hls.py
====================
CNN per-residue importance via Gradient x Input saliency.
This is the correct method for CNN — not attention scores.

Attention scores are on CNN *output* positions (windows).
Gradients are on CNN *input* embeddings (individual residues).

Run from DrugBAN_BiLSTM root:
  python repro_comparison/cnn_saliency_5hls.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from models import DrugBAN
from dataloader import DTIDataset
from utils import graph_collate_func

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
LIG_FILE   = os.path.join(ROOT_DIR, "P-L/2011-2019/5hls/5hls_ligand.sdf")
PROT_FILE  = os.path.join(ROOT_DIR, "P-L/2011-2019/5hls/5hls_protein.pdb")
CONTACT_CUTOFF = 5.0
PDB_OFFSET = 42   # FASTA index 0 → PDB resi 42

CNN_CFG  = os.path.join(ROOT_DIR, "configs/DrugBAN.yaml")
CNN_CKPT = os.path.join(ROOT_DIR, "result/DrugBAN/best_model_epoch_90.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Direct contact residues ──────────────────────────────────────
def load_pocket_indices():
    from collections import OrderedDict
    def read_sdf_coords(p):
        with open(p) as f: lines = f.readlines()
        n = int(lines[3].split()[0])
        return np.array([[float(l.split()[i]) for i in range(3)] for l in lines[4:4+n]])

    def read_protein(p):
        res_atoms, ca_order, ca_seen = OrderedDict(), [], set()
        with open(p) as f:
            for line in f:
                if not line.startswith("ATOM"): continue
                rn = int(line[22:26].strip())
                an = line[12:16].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                if rn not in res_atoms: res_atoms[rn] = []
                res_atoms[rn].append([x, y, z])
                if an == "CA" and rn not in ca_seen:
                    ca_order.append(rn); ca_seen.add(rn)
        return res_atoms, ca_order

    lig = read_sdf_coords(LIG_FILE)
    res_atoms, ca_order = read_protein(PROT_FILE)
    pdb_to_idx = {r: i for i, r in enumerate(ca_order)}
    contact_pdb = []
    for rn, atoms in res_atoms.items():
        if rn not in pdb_to_idx: continue
        d = np.sqrt(((np.array(atoms)[:, None] - lig[None])**2).sum(2)).min()
        if d <= CONTACT_CUTOFF:
            contact_pdb.append(rn)
    pocket_idx = set(pdb_to_idx[r] for r in contact_pdb)
    print(f"Contact residues (<=5A): {sorted(contact_pdb)}")
    return pocket_idx, ca_order


# ── Gradient x Input saliency for CNN ───────────────────────────
def compute_cnn_saliency(df, seq_len):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(CNN_CFG)

    dataset = DTIDataset(
        df.index.values, df,
        max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
        use_features=False, use_selfies=False,
        selfies_vocab=None,
        max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
        use_drug_features=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=graph_collate_func)

    model = DrugBAN(**cfg).to(DEVICE)
    state = torch.load(CNN_CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Find protein embedding layer
    prot_encoder = model.protein_extractor

    saliency_accum = np.zeros(seq_len)
    n = 0

    for batch in loader:
        v_d, v_p, labels = batch
        if isinstance(v_d, tuple):
            v_d = tuple(x.to(DEVICE) for x in v_d)
        else:
            v_d = v_d.to(DEVICE)

        # v_p is protein token indices, shape [batch, L]
        v_p = v_p.to(DEVICE) if not isinstance(v_p, tuple) else tuple(x.to(DEVICE) for x in v_p)

        # --- Gradient x Input via forward hook ---
        # Make embedding output a leaf node so it accumulates grad
        captured = [None]

        def _hook(module, inp, out):
            leaf = out.detach().requires_grad_(True)
            captured[0] = leaf
            return leaf   # replace output in computation graph

        handle = prot_encoder.embedding.register_forward_hook(_hook)
        try:
            _, _, score, _ = model(v_d, v_p, mode="eval")  # score = prediction
            score.sum().backward()
        finally:
            handle.remove()

        emb  = captured[0]                              # [1, L, emb_dim]
        grad = emb.grad                                 # [1, L, emb_dim]
        if grad is None:
            # Fallback: plain gradient (without ×input)
            raise RuntimeError("grad is None — check model forward graph")
        # Gradient x Input saliency
        sal = (grad * emb.detach()).abs().sum(dim=-1)   # [1, L]
        sal = sal.squeeze(0).cpu().numpy()               # [L]
        sal = sal[:seq_len]

        # Normalize to [0, 1]
        mn, mx = sal.min(), sal.max()
        sal_n = (sal - mn) / (mx - mn + 1e-10)

        saliency_accum += sal_n
        n += 1

    # Average saliency across all drugs
    avg_sal = saliency_accum / n
    # Normalize again
    mn, mx = avg_sal.min(), avg_sal.max()
    avg_sal = (avg_sal - mn) / (mx - mn + 1e-10)

    # Apply same threshold: mean + std
    thr = avg_sal.mean() + avg_sal.std()
    print(f"CNN Saliency threshold: {thr:.3f}")
    red    = [i for i in range(seq_len) if avg_sal[i] > thr]
    orange_mask = (avg_sal > avg_sal.mean() + 0.5*avg_sal.std()) & (avg_sal <= thr)
    orange = [i for i in range(seq_len) if orange_mask[i]]

    print(f"CNN Saliency RED   ({len(red)}): PDB {[i+PDB_OFFSET for i in red]}")
    print(f"CNN Saliency ORANGE({len(orange)}): PDB {[i+PDB_OFFSET for i in orange]}")
    return avg_sal, red, orange


# ── Summary comparison ───────────────────────────────────────────
def compare(sal_red_idx, pocket_idx, seq_len, label="CNN Saliency"):
    contact = pocket_idx
    red_set = set(sal_red_idx)
    hit = red_set & contact
    exp = len(red_set) * len(contact) / seq_len
    prec = len(hit) / len(red_set) if red_set else 0
    rec  = len(hit) / len(contact) if contact else 0
    print(f"\n=== {label} ===")
    print(f"  RED residues: {len(red_set)} ({len(red_set)/seq_len:.1%})")
    print(f"  Contact hit : {sorted([i+PDB_OFFSET for i in hit])} ({len(hit)} residues)")
    print(f"  Precision   : {len(hit)}/{len(red_set)} = {prec:.1%}")
    print(f"  Recall      : {len(hit)}/{len(contact)} = {rec:.1%}")
    print(f"  vs random   : expected {exp:.1f}, actual {len(hit)} = {len(hit)/exp:.1f}x")


# ── Plot saliency bar chart ──────────────────────────────────────
def plot_saliency(avg_sal, pocket_idx, pdb_residue_nums, red_idx, out_dir):
    seq_len = len(avg_sal)
    thr = avg_sal.mean() + avg_sal.std()
    colors = []
    for i in range(seq_len):
        if avg_sal[i] > thr:
            colors.append("#e74c3c")
        elif avg_sal[i] > avg_sal.mean() + 0.5*avg_sal.std():
            colors.append("#e67e22")
        else:
            colors.append("#aed6f1")

    fig, ax = plt.subplots(figsize=(22, 4))
    pdb_nums = pdb_residue_nums
    xs = list(range(seq_len))
    ax.bar(xs, avg_sal, color=colors, width=0.85, zorder=2)

    # Mark contact residues with blue border
    for i in pocket_idx:
        ax.axvspan(i - 0.45, i + 0.45, alpha=0.25, color="#2980b9", zorder=1)

    ax.axhline(thr, color="gray", linestyle="--", linewidth=0.8, label=f"threshold (mean+std={thr:.2f})")
    ax.set_xticks([i for i in xs if pdb_nums[i] % 10 == 0])
    ax.set_xticklabels([str(pdb_nums[i]) for i in xs if pdb_nums[i] % 10 == 0], fontsize=7)
    ax.set_xlabel("PDB Residue Number (42-168)")
    ax.set_ylabel("Gradient x Input Saliency (normalized)")
    ax.set_title("5HLS - DrugBAN CNN: Gradient x Input Saliency (correct per-residue attribution)\n"
                 "Red bars = saliency > mean+std  |  Blue shading = direct contact <=5A")
    ax.legend(fontsize=8)

    patches = [
        mpatches.Patch(color="#e74c3c", label="Saliency RED (>mean+std)"),
        mpatches.Patch(color="#e67e22", label="Saliency ORANGE"),
        mpatches.Patch(color="#2980b9", alpha=0.4, label="Direct contact <=5A"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    plt.tight_layout()
    out = os.path.join(out_dir, "5HLS_CNN_saliency.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    pocket_idx, pdb_residue_nums = load_pocket_indices()

    dfs = []
    for split in ["train", "val"]:
        path = os.path.join(ROOT_DIR, f"datasets/bindingdb/overlap_with_mask/{split}.csv")
        df = pd.read_csv(path)
        df["split"] = split
        dfs.append(df)
    df_all  = pd.concat(dfs, ignore_index=True)
    df_5hls = df_all[df_all["PDB_ID"].str.upper().str.strip() == "5HLS"].reset_index(drop=True)
    seq_len = len(df_5hls.iloc[0]["Protein"])

    print(f"Device: {DEVICE}  |  Samples: {len(df_5hls)}  |  Seq len: {seq_len}\n")
    avg_sal, red_idx, orange_idx = compute_cnn_saliency(df_5hls, seq_len)
    compare(red_idx, pocket_idx, seq_len, "CNN Gradient Saliency")
    plot_saliency(avg_sal, pocket_idx, pdb_residue_nums, red_idx, OUTPUT_DIR)
    print("\nDone.")
