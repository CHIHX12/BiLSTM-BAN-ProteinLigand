#!/usr/bin/env python3
"""
run_all_proteins.py
===================
Run CNN vs BiLSTM attention frequency comparison for all 4 proteins
with available PDB structure (2YC0, 5HLS, 6G3Q, 6N4E).

For each protein:
  - Compute direct contact residues (<=5A, <=8A, <=10A)
  - Run DrugBAN (CNN+GCN) and DrugBAN_BiLSTM (BiLSTM+GCN) attention frequency
  - Compare RED residues vs binding site
  - Generate per-protein PNG and PyMOL .pml (ASCII only)
  - Write summary table to repro_comparison/SUMMARY.txt

Run from DrugBAN_BiLSTM root:
  python repro_comparison/run_all_proteins.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, OrderedDict
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from models import DrugBAN
from dataloader import DTIDataset
from utils import graph_collate_func

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(COMP_DIR, "without_binding_site")   # PDB/SDF files live here
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PDB_ROOT   = os.path.join(ROOT_DIR, "P-L/2011-2019")

PROTEINS = {
    "1AQ1":  {"splits": ["train", "val", "test"]},
    "1OXG":  {"splits": ["train", "val", "test"]},
    "1QFS":  {"splits": ["train", "val", "test"]},
    "1T48":  {"splits": ["train", "val", "test"]},
    "2QK8":  {"splits": ["train", "val", "test"]},
    "3NZC":  {"splits": ["train", "val", "test"]},
    "5HLS":  {"splits": ["train", "val"]},
    "6G3Q":  {"splits": ["train", "val", "test"]},
}

MODELS = [
    {
        "label": "CNN",
        "cfg":   os.path.join(ROOT_DIR, "configs/DrugBAN.yaml"),
        "ckpt":  os.path.join(ROOT_DIR, "result/DrugBAN/best_model_epoch_90.pth"),
    },
    {
        "label": "BiLSTM",
        "cfg":   os.path.join(ROOT_DIR, "configs/DrugBAN_BiLSTM.yaml"),
        "ckpt":  os.path.join(ROOT_DIR, "result/DrugBAN_BiLSTM/best_model_epoch_94.pth"),
    },
]

CUTOFFS = [5.0, 8.0, 10.0]


# ── Utilities ────────────────────────────────────────────────────
def read_sdf_coords(sdf_path):
    with open(sdf_path) as f:
        lines = f.readlines()
    # V2000 counts line is fixed-width: chars 0-2 = natoms, 3-5 = nbonds
    counts_line = lines[3]
    try:
        natoms = int(counts_line[0:3].strip())
    except ValueError:
        natoms = int(counts_line.split()[0])
    coords = []
    for i in range(natoms):
        parts = lines[4 + i].split()
        try:
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except (ValueError, IndexError):
            continue
    return np.array(coords)


def read_protein_pdb(pdb_path):
    res_atoms, ca_order, ca_seen = OrderedDict(), [], set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            rn = int(line[22:26].strip())
            an = line[12:16].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if rn not in res_atoms:
                res_atoms[rn] = []
            res_atoms[rn].append([x, y, z])
            if an == "CA" and rn not in ca_seen:
                ca_order.append(rn)
                ca_seen.add(rn)
    return res_atoms, ca_order


def compute_contacts(pdb_path, sdf_path, cutoff):
    lig = read_sdf_coords(sdf_path)
    res_atoms, ca_order = read_protein_pdb(pdb_path)
    pdb_to_idx = {r: i for i, r in enumerate(ca_order)}
    contact_pdb = []
    for rn, atoms in res_atoms.items():
        if rn not in pdb_to_idx:
            continue
        d = np.sqrt(((np.array(atoms)[:, None] - lig[None])**2).sum(2)).min()
        if d <= cutoff:
            contact_pdb.append(rn)
    contact_idx = set(pdb_to_idx[r] for r in contact_pdb)
    return contact_idx, contact_pdb, ca_order


def load_df(pdb_id, splits):
    dfs = []
    for sp in splits:
        path = os.path.join(ROOT_DIR,
               f"datasets/bindingdb/overlap_with_mask/{sp}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["split"] = sp
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    mask = df_all["PDB_ID"].str.upper().str.strip() == pdb_id.upper()
    return df_all[mask].reset_index(drop=True)


def compute_frequency(model_info, df, seq_len):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(model_info["cfg"])

    dataset = DTIDataset(
        df.index.values, df,
        max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
        use_features=cfg.PROTEIN.get("USE_BILSTM", False),
        use_selfies=False, selfies_vocab=None,
        max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
        use_drug_features=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=graph_collate_func)

    model = DrugBAN(**cfg).to(DEVICE)
    state = torch.load(model_info["ckpt"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    votes = defaultdict(int)
    n = 0
    with torch.no_grad():
        for batch in loader:
            v_d, v_p, labels = batch
            v_d = tuple(x.to(DEVICE) for x in v_d) if isinstance(v_d, tuple) else v_d.to(DEVICE)
            v_p = tuple(x.to(DEVICE) for x in v_p) if isinstance(v_p, tuple) else v_p.to(DEVICE)

            _, _, _, attn = model(v_d, v_p, mode="eval")
            attn = attn.squeeze(0)
            if attn.dim() == 3:
                attn = attn.mean(0)
            att = attn.sum(0).cpu().numpy()[:seq_len]
            mn, mx = att.min(), att.max()
            att_n = (att - mn) / (mx - mn + 1e-10)
            thr = att_n.mean() + att_n.std()
            for pos in np.where(att_n > thr)[0]:
                votes[int(pos)] += 1
            n += 1

    freq = np.zeros(seq_len)
    for pos, cnt in votes.items():
        if pos < seq_len:
            freq[pos] = cnt / n
    return freq, n


def freq_stats(freq, pdb_offset):
    red    = [i + pdb_offset for i in range(len(freq)) if freq[i] > 0.75]
    orange = [i + pdb_offset for i in range(len(freq)) if 0.5  < freq[i] <= 0.75]
    yellow = [i + pdb_offset for i in range(len(freq)) if 0.25 < freq[i] <= 0.5]
    return red, orange, yellow


def overlap_metrics(red_pdb, contact_pdb_set, seq_len, n_contact):
    red_s = set(red_pdb)
    hit   = red_s & contact_pdb_set
    prec  = len(hit) / len(red_s) if red_s else 0
    rec   = len(hit) / n_contact  if n_contact else 0
    f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
    exp   = len(red_s) * n_contact / seq_len
    vsr   = len(hit) / exp if exp > 0 else 0
    return {"hit": sorted(hit), "n_hit": len(hit), "prec": prec,
            "rec": rec, "f1": f1, "vs_rand": vsr}


# ── Plot per-protein comparison ───────────────────────────────────
def plot_protein(pdb_id, protein_seq, freq_dict, contact_5A_pdb, pdb_residue_nums, out_dir):
    seq_len = len(protein_seq)
    pdb_offset = pdb_residue_nums[0]

    def fc(f):
        if   f > 0.75: return "#e74c3c"
        elif f > 0.50: return "#e67e22"
        elif f > 0.25: return "#f1c40f"
        else:          return "#ecf0f1"

    contact_idx = set(i for i, r in enumerate(pdb_residue_nums) if r in set(contact_5A_pdb))

    COLS = 25
    n_chunks = (seq_len + COLS - 1) // COLS
    n_rows   = 2 + len(freq_dict)   # seq + binding + models

    BOX_W = 0.50; BOX_H = 0.68
    LM    = 3.0;  row_h = 1.05; gap = 0.55

    fig_w = COLS * BOX_W + LM + 0.5
    fig_h = n_chunks * (n_rows * row_h + gap) + 1.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")

    n_drugs_str = ",".join(f"{m}:{freq_dict[m][1]}" for m in freq_dict)
    ax.text(fig_w/2, fig_h-0.3,
            f"{pdb_id}  |  CNN vs BiLSTM Attention Frequency  (n drugs: {n_drugs_str})\n"
            f"PDB residues {pdb_residue_nums[0]}-{pdb_residue_nums[-1]}  |  Blue box = direct contact <=5A",
            ha="center", va="top", fontsize=10, fontweight="bold", linespacing=1.5)

    row_labels = ["Sequence", "Binding Site (5A)"] + list(freq_dict.keys())

    for ci in range(n_chunks):
        s = ci * COLS; e = min(s + COLS, seq_len)
        top = fig_h - 0.75 - ci * (n_rows * row_h + gap)

        for ri in range(n_rows):
            yc = top - ri * row_h - 0.35
            lc = "#1a5276" if ri == 1 else "#2c3e50"
            ax.text(LM - 0.2, yc, row_labels[ri],
                    ha="right", va="center", fontsize=6.5, color=lc)

            for col_i, pos in enumerate(range(s, e)):
                x = LM + col_i * BOX_W

                if ri == 0:   # Sequence row
                    pdb_num = pdb_residue_nums[pos]
                    if pdb_num % 5 == 0:
                        ax.text(x + BOX_W/2, yc + 0.33, str(pdb_num),
                                ha="center", va="bottom", fontsize=5.5, color="#555")
                    ax.text(x + BOX_W/2, yc, protein_seq[pos],
                            ha="center", va="center", fontsize=9, fontweight="bold",
                            color="#2c3e50", fontfamily="monospace")

                elif ri == 1:  # Binding site row
                    color = "#2980b9" if pos in contact_idx else "#d6eaf8"
                    edge  = "#1a5276" if pos in contact_idx else "#aed6f1"
                    lw    = 1.8       if pos in contact_idx else 0.4
                    rect = mpatches.FancyBboxPatch(
                        (x+0.03, yc-BOX_H/2), BOX_W-0.06, BOX_H,
                        boxstyle="round,pad=0.03", facecolor=color, edgecolor=edge, linewidth=lw)
                    ax.add_patch(rect)
                    if pos in contact_idx:
                        ax.text(x+BOX_W/2, yc, protein_seq[pos],
                                ha="center", va="center", fontsize=7.5,
                                color="white", fontweight="bold", fontfamily="monospace")

                else:          # Model rows
                    m_key = list(freq_dict.keys())[ri - 2]
                    freq  = freq_dict[m_key][0]
                    f     = freq[pos]
                    rect  = mpatches.FancyBboxPatch(
                        (x+0.03, yc-BOX_H/2), BOX_W-0.06, BOX_H,
                        boxstyle="round,pad=0.03", facecolor=fc(f),
                        edgecolor="#bdc3c7", linewidth=0.4)
                    ax.add_patch(rect)
                    if f > 0.25:
                        tc = "white" if f > 0.5 else "#555"
                        ax.text(x+BOX_W/2, yc, f"{int(f*100)}%",
                                ha="center", va="center", fontsize=5.0,
                                color=tc, fontweight="bold")
                    if pos in contact_idx:
                        border = mpatches.FancyBboxPatch(
                            (x+0.03, yc-BOX_H/2), BOX_W-0.06, BOX_H,
                            boxstyle="round,pad=0.03", facecolor="none",
                            edgecolor="#1a5276", linewidth=2.2)
                        ax.add_patch(border)

        if ci < n_chunks - 1:
            ax.axhline(top - n_rows*row_h - 0.1, xmin=0.01, xmax=0.99,
                       color="#bdc3c7", linewidth=0.5, linestyle="--")

    # Legend
    items = [("#e74c3c","#c0392b",">75% RED"), ("#e67e22","#d35400","50-75% ORANGE"),
             ("#f1c40f","#d4ac0d","25-50% YELLOW"), ("#ecf0f1","#bdc3c7","<25% bg"),
             ("#2980b9","#1a5276","Contact <=5A")]
    lx, ly = 0.4, 0.85
    for fc_, ec_, lbl in items:
        rect = mpatches.FancyBboxPatch((lx, ly-0.18), 0.28, 0.30,
            boxstyle="round,pad=0.02", facecolor=fc_, edgecolor=ec_, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(lx+0.38, ly, lbl, ha="left", va="center", fontsize=7.5)
        lx += 3.0

    plt.tight_layout(pad=0.3)
    out = os.path.join(out_dir, f"{pdb_id}_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Generate PyMOL script (ASCII only) ───────────────────────────
def write_pymol(pdb_id, freq_dict, contact_5A_pdb, out_dir, pdb_lower):
    for model_label, (freq, n_drugs) in freq_dict.items():
        pdb_off = None  # determined from freq indexing
        red_pdb = [i for i, f in enumerate(freq) if f > 0.75]   # 0-based idx
        # We need PDB numbers — read from protein
        prot_pdb = os.path.join(out_dir, f"{pdb_lower}_protein.pdb")
        _, ca_order = read_protein_pdb(prot_pdb)

        def idx_to_pdb(idxs):
            return [ca_order[i] for i in idxs if i < len(ca_order)]

        red_r    = idx_to_pdb([i for i, f in enumerate(freq) if f > 0.75])
        orange_r = idx_to_pdb([i for i, f in enumerate(freq) if 0.5 < f <= 0.75])
        yellow_r = idx_to_pdb([i for i, f in enumerate(freq) if 0.25 < f <= 0.5])

        def resi_str(lst):
            return "+".join(str(r) for r in sorted(lst)) if lst else "none"

        fname = os.path.join(out_dir, f"{pdb_id}_{model_label}.pml")
        with open(fname, "w") as f:
            f.write(f"# PyMOL Script - {pdb_id} {model_label}\n")
            f.write(f"# Attention frequency (n={n_drugs} ligands)\n")
            f.write(f"# RED>75%  ORANGE 50-75%  YELLOW 25-50%\n")
            f.write(f"# BLUE = direct contact <=5A  MAGENTA = RED+contact\n")
            f.write(f"# Usage: File -> Run Script (keep pdb/sdf in same folder)\n\n")
            f.write(f"reinitialize\n")
            f.write(f"load {pdb_lower}_protein.pdb, protein\n")
            f.write(f"load {pdb_lower}_ligand.sdf,  ligand\n")
            f.write(f"hide everything, all\n")
            f.write(f"show cartoon, protein\n")
            f.write(f"color gray70, protein\n")
            f.write(f"show sticks, ligand\n")
            f.write(f"color cyan, ligand\n\n")
            if red_r:
                f.write(f"select attn_red, (protein and resi {resi_str(red_r)})\n")
                f.write(f"color red, attn_red\n")
                f.write(f"show sticks, attn_red\n\n")
            if orange_r:
                f.write(f"select attn_orange, (protein and resi {resi_str(orange_r)})\n")
                f.write(f"color orange, attn_orange\n")
                f.write(f"show sticks, attn_orange\n\n")
            if yellow_r:
                f.write(f"select attn_yellow, (protein and resi {resi_str(yellow_r)})\n")
                f.write(f"color yellow, attn_yellow\n")
                f.write(f"show sticks, attn_yellow\n\n")
            f.write(f"select binding_site, (protein and resi {resi_str(contact_5A_pdb)})\n")
            f.write(f"color blue, binding_site\n")
            f.write(f"show sticks, binding_site\n\n")
            if red_r:
                f.write(f"select consensus, attn_red and binding_site\n")
                f.write(f"color magenta, consensus\n")
                f.write(f"show sticks, consensus\n\n")
            f.write(f"zoom\nbg_color white\n\n")
            f.write(f'print "=== {pdb_id} {model_label} ==="\n')
            f.write(f'print "RED    ({len(red_r)}): {red_r}"\n')
            f.write(f'print "Contact ({len(contact_5A_pdb)}): {sorted(contact_5A_pdb)}"\n')
        print(f"  PyMOL: {fname}")


# ── Main ─────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")
    summary_rows = []

    for pdb_id, info in PROTEINS.items():
        pdb_lower  = pdb_id.lower()
        out_dir    = os.path.join(OUT_DIR, pdb_id)
        prot_pdb   = os.path.join(out_dir, f"{pdb_lower}_protein.pdb")
        lig_sdf    = os.path.join(out_dir, f"{pdb_lower}_ligand.sdf")

        print(f"\n{'='*60}")
        print(f"  {pdb_id}")
        print(f"{'='*60}")

        if not os.path.exists(prot_pdb) or not os.path.exists(lig_sdf):
            print(f"  SKIP: missing PDB/SDF files")
            continue

        # Load data
        df = load_df(pdb_id, info["splits"])
        if len(df) == 0:
            print(f"  SKIP: no data rows found")
            continue

        protein_seq = df.iloc[0]["Protein"]
        seq_len     = len(protein_seq)
        print(f"  Drugs: {len(df)}  |  Seq len: {seq_len}")

        # Compute contacts at multiple cutoffs
        contacts = {}
        for co in CUTOFFS:
            cidx, cpdb, ca_order = compute_contacts(prot_pdb, lig_sdf, co)
            contacts[co] = {"idx": cidx, "pdb": set(cpdb)}
        pdb_residue_nums = ca_order  # from last compute_contacts call (same for all cutoffs)
        pdb_offset = pdb_residue_nums[0]
        print(f"  PDB range: {pdb_residue_nums[0]}-{pdb_residue_nums[-1]}")
        for co in CUTOFFS:
            print(f"  Contact {co}A: {len(contacts[co]['pdb'])} residues  {sorted(contacts[co]['pdb'])[:10]}{'...' if len(contacts[co]['pdb'])>10 else ''}")

        # Compute frequency for each model
        freq_dict = {}
        for m in MODELS:
            print(f"\n  Running {m['label']}...")
            try:
                freq, n_drugs = compute_frequency(m, df, seq_len)
                freq_dict[m["label"]] = (freq, n_drugs)
                red_pdb = [pdb_residue_nums[i] for i in range(seq_len) if freq[i] > 0.75]
                print(f"    RED={len(red_pdb)}  ORANGE={sum(1 for f in freq if 0.5<f<=0.75)}  YELLOW={sum(1 for f in freq if 0.25<f<=0.5)}")
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        if not freq_dict:
            continue

        # Metrics summary
        print(f"\n  === Overlap metrics ===")
        contact_5A_pdb_set = contacts[5.0]["pdb"]
        for mlabel, (freq, n_drugs) in freq_dict.items():
            red_pdb = [pdb_residue_nums[i] for i in range(seq_len) if freq[i] > 0.75]
            for co in CUTOFFS:
                m = overlap_metrics(red_pdb, contacts[co]["pdb"],
                                    seq_len, len(contacts[co]["pdb"]))
                print(f"    {mlabel:10s} {co}A: "
                      f"n={len(red_pdb):2d}, hit={m['n_hit']:2d}/{len(contacts[co]['pdb']):2d}, "
                      f"P={m['prec']:.0%}, R={m['rec']:.0%}, F1={m['f1']:.2f}, "
                      f"vs_rand={m['vs_rand']:.1f}x")
                summary_rows.append({
                    "PDB": pdb_id, "Model": mlabel,
                    "Cutoff": f"{co}A", "n_drugs": n_drugs,
                    "n_red": len(red_pdb), "n_contact": len(contacts[co]["pdb"]),
                    "n_hit": m["n_hit"], "Precision": m["prec"],
                    "Recall": m["rec"], "F1": m["f1"], "vs_rand": m["vs_rand"],
                })

        # Plot
        print(f"\n  Plotting...")
        plot_protein(pdb_id, protein_seq, freq_dict,
                     sorted(contact_5A_pdb_set), pdb_residue_nums, out_dir)

        # PyMOL scripts
        write_pymol(pdb_id, freq_dict, sorted(contact_5A_pdb_set), out_dir, pdb_lower)

    # Write summary table
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out_sum = os.path.join(COMP_DIR, "SUMMARY.csv")
        df_sum.to_csv(out_sum, index=False)
        print(f"\n{'='*60}")
        print(f"SUMMARY saved: {out_sum}")
        print(f"\n5A Cutoff Results:")
        sub = df_sum[df_sum["Cutoff"]=="5.0A"][["PDB","Model","n_drugs","n_red","n_hit","n_contact","Precision","Recall","F1","vs_rand"]]
        pd.set_option("display.float_format", lambda x: f"{x:.2f}")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 120)
        print(sub.to_string(index=False))

if __name__ == "__main__":
    main()
