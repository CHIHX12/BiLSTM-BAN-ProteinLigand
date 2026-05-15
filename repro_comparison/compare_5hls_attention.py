#!/usr/bin/env python3
"""
compare_5hls_attention.py
=========================
對 5HLS 蛋白質，比較兩個模型的注意力頻率圖，並標出 PDB 結合位點。

每段（每 25 個殘基）包含：
  Row 0 : Sequence        - 胺基酸字母 + 序號
  Row 1 : Binding Site    - 藍色框 = PDB 實際口袋殘基（5hls_pocket.pdb）
  Row 2 : DrugBAN         - RED/ORANGE/YELLOW 頻率
  Row 3 : DrugBAN_BiLSTM  - RED/ORANGE/YELLOW 頻率

執行（在 DrugBAN_BiLSTM 根目錄）：
  python repro_comparison/compare_5hls_attention.py

輸出：
  repro_comparison/5HLS_attention_comparison.png / .svg
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch
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

# ── 路徑設定 ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = SCRIPT_DIR

LIG_FILE   = os.path.join(ROOT_DIR, "P-L/2011-2019/5hls/5hls_ligand.sdf")
PROT_FILE  = os.path.join(ROOT_DIR, "P-L/2011-2019/5hls/5hls_protein.pdb")
CONTACT_CUTOFF = 5.0   # Å — 直接接觸殘基

MODELS = [
    {
        # ⚠️ CNN 注意力不對應個別殘基，僅供參考比較
        "label": "DrugBAN  (ProteinCNN + GCN)  [參考]",
        "cfg":   os.path.join(ROOT_DIR, "configs/DrugBAN.yaml"),
        "ckpt":  os.path.join(ROOT_DIR, "result/DrugBAN/best_model_epoch_90.pth"),
    },
    {
        "label": "DrugBAN_BiLSTM  (ProteinBiLSTM + GCN)",
        "cfg":   os.path.join(ROOT_DIR, "configs/DrugBAN_BiLSTM.yaml"),
        "ckpt":  os.path.join(ROOT_DIR, "result/DrugBAN_BiLSTM/best_model_epoch_94.pth"),
    },
]

DATA_SPLITS = ["train", "val"]
PDB_TARGET  = "5HLS"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 讀取 PDB 口袋殘基（0-based 序列索引）─────────────────────
def load_pocket_indices():
    """
    用配體原子座標計算直接接觸殘基（≤ CONTACT_CUTOFF Å）。
    返回：
      pocket_idx      : set of 0-based 序列索引
      pdb_residue_nums: list，pdb_residue_nums[i] = 序列第 i 位的 PDB 殘基號
    """
    import numpy as np
    from collections import OrderedDict

    # 1. 讀配體座標（SDF格式）
    def read_sdf_coords(sdf_path):
        with open(sdf_path) as f:
            lines = f.readlines()
        natoms = int(lines[3].split()[0])
        coords = []
        for i in range(4, 4 + natoms):
            p = lines[i].split()
            coords.append([float(p[0]), float(p[1]), float(p[2])])
        return np.array(coords)

    # 2. 讀蛋白質：每個殘基的所有重原子座標 + CA 順序
    def read_protein(pdb_path):
        res_atoms  = OrderedDict()   # (resnum) → [(x,y,z),...]
        ca_order   = []              # CA 原子的殘基號順序（序列順序）
        ca_seen    = set()
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                resnum    = int(line[22:26].strip())
                atom_name = line[12:16].strip()
                x, y, z   = float(line[30:38]), float(line[38:46]), float(line[46:54])
                if resnum not in res_atoms:
                    res_atoms[resnum] = []
                res_atoms[resnum].append([x, y, z])
                if atom_name == "CA" and resnum not in ca_seen:
                    ca_order.append(resnum)
                    ca_seen.add(resnum)
        return res_atoms, ca_order

    lig_coords          = read_sdf_coords(LIG_FILE)
    res_atoms, ca_order = read_protein(PROT_FILE)
    pdb_residue_nums    = ca_order          # index i → PDB residue num

    # PDB編號 → 0-based序列索引
    pdb_to_idx = {r: i for i, r in enumerate(ca_order)}

    # 3. 每個殘基 vs 所有配體原子 → 最小距離
    lig_arr = lig_coords
    contact_pdb = []
    for resnum, atoms in res_atoms.items():
        if resnum not in pdb_to_idx:
            continue
        prot_arr = np.array(atoms)
        diffs    = prot_arr[:, None, :] - lig_arr[None, :, :]
        min_dist = np.sqrt((diffs**2).sum(axis=2)).min()
        if min_dist <= CONTACT_CUTOFF:
            contact_pdb.append(resnum)

    pocket_idx = set(pdb_to_idx[r] for r in contact_pdb if r in pdb_to_idx)

    print(f"PDB 蛋白質殘基範圍: {ca_order[0]}-{ca_order[-1]}  共 {len(ca_order)} 個")
    print(f"直接接觸殘基 (≤{CONTACT_CUTOFF}Å): {len(pocket_idx)} 個  "
          f"PDB號: {sorted(contact_pdb)}")

    return pocket_idx, pdb_residue_nums


# ── 載入 5HLS 資料 ──────────────────────────────────────────────
def load_5hls_data():
    dfs = []
    for split in DATA_SPLITS:
        path = os.path.join(ROOT_DIR,
               f"datasets/bindingdb/overlap_with_mask/{split}.csv")
        df = pd.read_csv(path)
        df["split"] = split
        dfs.append(df)
    df_all  = pd.concat(dfs, ignore_index=True)
    mask    = df_all["PDB_ID"].str.upper().str.strip() == PDB_TARGET
    df_5hls = df_all[mask].reset_index(drop=True)
    print(f"找到 {len(df_5hls)} 筆 {PDB_TARGET} 樣本")
    return df_5hls


# ── 計算每個殘基的高注意力頻率 ─────────────────────────────────
def compute_frequency(model_info, df, seq_len):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(model_info["cfg"])

    dataset = DTIDataset(
        df.index.values, df,
        max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
        use_features=cfg.PROTEIN.get("USE_BILSTM", False),
        use_selfies=False,
        selfies_vocab=None,
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
    n     = 0

    with torch.no_grad():
        for batch in loader:
            v_d, v_p, labels = batch
            if isinstance(v_d, tuple):
                v_d = tuple(x.to(DEVICE) for x in v_d)
            else:
                v_d = v_d.to(DEVICE)
            if isinstance(v_p, tuple):
                v_p = tuple(x.to(DEVICE) for x in v_p)
            else:
                v_p = v_p.to(DEVICE)

            _, _, _, attn = model(v_d, v_p, mode="eval")
            attn = attn.squeeze(0)
            if attn.dim() == 3:
                attn = attn.mean(0)
            att = attn.sum(0).cpu().numpy()

            mn, mx = att.min(), att.max()
            att_n  = (att - mn) / (mx - mn + 1e-10)
            att_n  = att_n[:seq_len]

            thr = att_n.mean() + att_n.std()
            for pos in np.where(att_n > thr)[0]:
                votes[int(pos)] += 1
            n += 1

    freq = np.zeros(seq_len)
    for pos, cnt in votes.items():
        if pos < seq_len:
            freq[pos] = cnt / n

    print(f"  {model_info['label']}: n={n} drugs  "
          f"RED={sum(freq>0.75)}  "
          f"ORANGE={sum((freq>0.5)&(freq<=0.75))}  "
          f"YELLOW={sum((freq>0.25)&(freq<=0.5))}")
    return freq


# ── 繪圖 ──────────────────────────────────────────────────────
def plot_comparison(protein_seq, freq_list, model_labels, pocket_idx,
                    pdb_residue_nums, out_dir):
    seq_len = len(protein_seq)

    def freq_to_color(f):
        if   f > 0.75: return "#e74c3c"   # RED
        elif f > 0.50: return "#e67e22"   # ORANGE
        elif f > 0.25: return "#f1c40f"   # YELLOW
        else:          return "#ecf0f1"   # 淺灰背景

    COLS_PER_ROW = 25
    n_chunks  = (seq_len + COLS_PER_ROW - 1) // COLS_PER_ROW

    # rows per chunk: 1 sequence + 1 binding site + n_models
    n_rows    = 2 + len(freq_list)

    BOX_W      = 0.50
    BOX_H      = 0.68
    LEFT_MARGIN = 3.2
    row_height  = 1.05
    chunk_gap   = 0.65

    fig_width    = COLS_PER_ROW * BOX_W + LEFT_MARGIN + 0.5
    total_height = n_chunks * (n_rows * row_height + chunk_gap) + 1.8

    fig, ax = plt.subplots(figsize=(fig_width, total_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    # 標題
    ax.text(fig_width / 2, total_height - 0.35,
            f"5HLS  |  DrugBAN_BiLSTM Attention Frequency  (n=5 ligands)\n"
            f"序號 = PDB 殘基號 (42–168)  |  藍框 = 配體直接接觸殘基 (≤{CONTACT_CUTOFF}Å)",
            ha="center", va="top", fontsize=11, fontweight="bold", linespacing=1.5)

    ROW_LABELS = ["Sequence", "Binding Site (PDB)"] + model_labels

    for chunk_i in range(n_chunks):
        start = chunk_i * COLS_PER_ROW
        end   = min(start + COLS_PER_ROW, seq_len)

        chunk_top = total_height - 0.75 - chunk_i * (n_rows * row_height + chunk_gap)

        for row_i in range(n_rows):
            y_center = chunk_top - row_i * row_height - 0.35

            # ── 行標籤 ──
            label_color = "#1a5276" if row_i == 1 else "#2c3e50"
            label_weight = "bold" if row_i == 1 else "normal"
            ax.text(LEFT_MARGIN - 0.25, y_center,
                    ROW_LABELS[row_i],
                    ha="right", va="center",
                    fontsize=6.5, color=label_color, fontweight=label_weight)

            for col_i, pos in enumerate(range(start, end)):
                x = LEFT_MARGIN + col_i * BOX_W

                # ──────── Row 0: Sequence ────────
                if row_i == 0:
                    aa      = protein_seq[pos]
                    pdb_num = pdb_residue_nums[pos]   # 校正後的 PDB 殘基號

                    # 序號標籤：用 PDB 殘基號，每5個顯示一次
                    if pdb_num % 5 == 0:
                        ax.text(x + BOX_W / 2, y_center + 0.33,
                                str(pdb_num),
                                ha="center", va="bottom",
                                fontsize=5.5, color="#555")
                    # 胺基酸字母
                    ax.text(x + BOX_W / 2, y_center, aa,
                            ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="#2c3e50", fontfamily="monospace")

                # ──────── Row 1: Binding Site ────────
                elif row_i == 1:
                    if pos in pocket_idx:
                        rect = mpatches.FancyBboxPatch(
                            (x + 0.03, y_center - BOX_H / 2),
                            BOX_W - 0.06, BOX_H,
                            boxstyle="round,pad=0.03",
                            facecolor="#2980b9",
                            edgecolor="#1a5276",
                            linewidth=1.8,
                        )
                        ax.add_patch(rect)
                        ax.text(x + BOX_W / 2, y_center,
                                protein_seq[pos],
                                ha="center", va="center",
                                fontsize=7.5, color="white",
                                fontweight="bold", fontfamily="monospace")
                    else:
                        rect = mpatches.FancyBboxPatch(
                            (x + 0.03, y_center - BOX_H / 2),
                            BOX_W - 0.06, BOX_H,
                            boxstyle="round,pad=0.03",
                            facecolor="#d6eaf8",
                            edgecolor="#aed6f1",
                            linewidth=0.4,
                        )
                        ax.add_patch(rect)

                # ──────── Row 2+: Model attention ────────
                else:
                    m_idx = row_i - 2
                    f     = freq_list[m_idx][pos]
                    fc    = freq_to_color(f)

                    rect = mpatches.FancyBboxPatch(
                        (x + 0.03, y_center - BOX_H / 2),
                        BOX_W - 0.06, BOX_H,
                        boxstyle="round,pad=0.03",
                        facecolor=fc,
                        edgecolor="#bdc3c7",
                        linewidth=0.4,
                    )
                    ax.add_patch(rect)

                    # 數字標籤（頻率 > 25% 才顯示）
                    if f > 0.25:
                        pct        = int(f * 100)
                        txt_color  = "white" if f > 0.50 else "#555"
                        ax.text(x + BOX_W / 2, y_center,
                                f"{pct}%",
                                ha="center", va="center",
                                fontsize=5.0, color=txt_color,
                                fontweight="bold")

                    # ── 藍色外框：如果該位置同時在 pocket ──
                    if pos in pocket_idx:
                        border = mpatches.FancyBboxPatch(
                            (x + 0.03, y_center - BOX_H / 2),
                            BOX_W - 0.06, BOX_H,
                            boxstyle="round,pad=0.03",
                            facecolor="none",
                            edgecolor="#1a5276",
                            linewidth=2.2,
                        )
                        ax.add_patch(border)

        # 段落分隔線
        if chunk_i < n_chunks - 1:
            sep_y = chunk_top - n_rows * row_height - 0.1
            ax.axhline(sep_y, xmin=0.01, xmax=0.99,
                       color="#bdc3c7", linewidth=0.5, linestyle="--")

    # ── 圖例 ──────────────────────────────────────────────────
    legend_items = [
        ("#e74c3c", "#c0392b", "> 75%  RED"),
        ("#e67e22", "#d35400", "50–75% ORANGE"),
        ("#f1c40f", "#d4ac0d", "25–50% YELLOW"),
        ("#ecf0f1", "#bdc3c7", "< 25%  (background)"),
        ("#2980b9", "#1a5276", f"Direct Contact ≤{CONTACT_CUTOFF}Å (藍色框)"),
    ]
    leg_x = 0.4
    leg_y = 0.85
    for fc, ec, label in legend_items:
        rect = mpatches.FancyBboxPatch(
            (leg_x, leg_y - 0.18), 0.28, 0.30,
            boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(leg_x + 0.38, leg_y, label,
                ha="left", va="center", fontsize=7.5)
        leg_x += 3.0

    plt.tight_layout(pad=0.3)

    for ext in ["png", "svg"]:
        out_path = os.path.join(out_dir, f"5HLS_attention_comparison.{ext}")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"已儲存: {out_path}")
    plt.close()


# ── 主程式 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"執行裝置: {DEVICE}\n")

    pocket_idx, pdb_residue_nums = load_pocket_indices()
    df_5hls     = load_5hls_data()
    protein_seq = df_5hls.iloc[0]["Protein"]
    seq_len     = len(protein_seq)

    print(f"\n序列長度: {seq_len}")
    print(f"序列: {protein_seq}\n")

    freq_list    = []
    model_labels = []
    for m in MODELS:
        print(f"處理: {m['label']}")
        freq = compute_frequency(m, df_5hls, seq_len)
        freq_list.append(freq)
        model_labels.append(m["label"])
        print()

    print("繪圖中...")
    plot_comparison(protein_seq, freq_list, model_labels,
                    pocket_idx, pdb_residue_nums, OUTPUT_DIR)
    print("完成！")
