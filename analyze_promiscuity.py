#!/usr/bin/env python3
"""
analyze_promiscuity.py  —  共顎性分析 + 注意力圖輸出
=====================================================

功能：
  1. N 個藥物 × M 個受體全矩陣預測（共顎性矩陣）
  2. 輸出共顎性熱圖（binding probability matrix PNG）
  3. 對每個 BIND 對輸出注意力圖（哪些殘基被模型關注）

用法：
  python analyze_promiscuity.py \\
      --ligands  examples/drugs.txt \\
      --receptors examples/proteins.fasta \\
      --model BiLSTM \\
      --out_dir results/promiscuity

輸入格式：
  --ligands  : Name<Tab>SMILES 每行一個（# 開頭為注釋）
  --receptors: FASTA 格式

輸出（--out_dir 下）：
  promiscuity_matrix.csv     - 概率矩陣（行=藥物，列=受體）
  promiscuity_heatmap.png    - 熱圖視覺化
  attention/<drug>_<rec>.png - 每個 BIND 對的殘基注意力圖
"""

import sys, os, warnings, argparse
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import get_cfg_defaults
from models import DrugBAN
from dataloader import DTIDataset
from utils import graph_collate_func

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    "BiLSTM": {
        "cfg":       os.path.join(ROOT, "configs/DrugBAN_BiLSTM.yaml"),
        "ckpt":      os.path.join(ROOT, "result/DrugBAN_BiLSTM/best_model_epoch_94.pth"),
        "threshold": 0.1511,
    },
    "CNN": {
        "cfg":       os.path.join(ROOT, "configs/DrugBAN.yaml"),
        "ckpt":      os.path.join(ROOT, "result/DrugBAN/best_model_epoch_90.pth"),
        "threshold": 0.3100,
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 解析輸入 ──────────────────────────────────────────────────

def parse_ligands(path: str) -> list[tuple[str, str]]:
    """回傳 [(name, smiles), ...] ，跳過 # 注釋行與空行。"""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                results.append((parts[0].strip(), parts[1].strip()))
            else:
                # 只有 SMILES，沒有名稱
                results.append((f"drug_{len(results)+1}", parts[0].strip()))
    return results


def parse_fasta(path: str) -> list[tuple[str, str]]:
    """回傳 [(name, seq), ...] 。"""
    results = []
    name, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name and seq_parts:
                    results.append((name, "".join(seq_parts)))
                name = line[1:].split("|")[0].strip()
                seq_parts = []
            elif line:
                seq_parts.append(line)
    if name and seq_parts:
        results.append((name, "".join(seq_parts)))
    return results


# ── 模型載入 ─────────────────────────────────────────────────

def load_model(model_name: str):
    info = MODEL_CONFIGS[model_name]
    cfg = get_cfg_defaults()
    cfg.merge_from_file(info["cfg"])
    cfg.freeze()
    model = DrugBAN(**cfg).to(DEVICE)
    state = torch.load(info["ckpt"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[model] {model_name}  device={DEVICE}  threshold={info['threshold']}")
    return model, cfg, info["threshold"]


# ── 推論：N×M 全矩陣 ─────────────────────────────────────────

def predict_all_pairs(
    ligands: list[tuple[str, str]],
    receptors: list[tuple[str, str]],
    model_name: str = "BiLSTM",
    batch_size: int = 16,
) -> list[dict]:
    """
    對所有 (藥物, 受體) 組合進行預測。
    回傳 list of dict，每筆包含：
      lig_name, rec_name, smiles, seq, prob, label, attn_vec (np.ndarray, 長度=實際蛋白質長度)
    """
    model, cfg, threshold = load_model(model_name)
    use_features = cfg.PROTEIN.get("USE_BILSTM", False)

    # 建立全矩陣 DataFrame
    rows = []
    for lig_name, smiles in ligands:
        for rec_name, seq in receptors:
            rows.append({
                "lig_name": lig_name,
                "rec_name": rec_name,
                "SMILES":   smiles,
                "Protein":  seq,
                "Y":        0,
            })
    df_all = pd.DataFrame(rows)
    actual_seq_lens = df_all["Protein"].str.len().tolist()

    dataset = DTIDataset(
        list(range(len(df_all))), df_all,
        max_drug_nodes=290,
        max_protein_length=1200,
        use_features=use_features,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=graph_collate_func)

    all_probs, all_attn_vecs = [], []

    print(f"[predict] {len(df_all)} pairs total ({len(ligands)} ligands x {len(receptors)} receptors)...")
    with torch.no_grad():
        for batch in loader:
            v_d, v_p, _ = batch
            v_d = tuple(x.to(DEVICE) for x in v_d) if isinstance(v_d, (list, tuple)) else v_d.to(DEVICE)
            v_p = tuple(x.to(DEVICE) for x in v_p) if isinstance(v_p, (list, tuple)) else v_p.to(DEVICE)

            _, _, score, attn = model(v_d, v_p, mode="eval")
            prob = torch.sigmoid(score).squeeze(-1).cpu().numpy()
            if prob.ndim == 0:
                prob = np.array([float(prob)])
            all_probs.extend(prob.tolist())

            # 提取每樣本的殘基注意力向量
            attn_cpu = attn.cpu()
            for i in range(attn_cpu.shape[0]):
                a = attn_cpu[i].squeeze(0)
                if a.dim() == 3:
                    a = a.mean(0)       # 平均多頭 → (atoms, residues)
                if a.dim() == 2:
                    a = a.sum(0)        # 對藥物原子求和 → (residues,)
                all_attn_vecs.append(a.numpy())

    # 組裝結果
    results = []
    for idx, row in df_all.iterrows():
        prob = all_probs[idx]
        raw_attn = all_attn_vecs[idx]
        seq_len = actual_seq_lens[idx]
        attn_trimmed = raw_attn[:seq_len]

        results.append({
            "lig_name": row["lig_name"],
            "rec_name": row["rec_name"],
            "smiles":   row["SMILES"],
            "seq":      row["Protein"],
            "prob":     prob,
            "label":    "BIND" if prob >= threshold else "NO_BIND",
            "attn_vec": attn_trimmed,
        })
    return results


# ── 共顎性矩陣熱圖 ────────────────────────────────────────────

def plot_promiscuity_matrix(
    results: list[dict],
    ligands: list[tuple[str, str]],
    receptors: list[tuple[str, str]],
    out_path: str,
) -> pd.DataFrame:
    """Plot N x M polypharmacology heatmap and save CSV. Returns matrix DataFrame."""
    lig_names = [n for n, _ in ligands]
    rec_names = [n for n, _ in receptors]

    prob_matrix = pd.DataFrame(index=lig_names, columns=rec_names, dtype=float)
    label_matrix = pd.DataFrame(index=lig_names, columns=rec_names, dtype=str)
    for r in results:
        prob_matrix.loc[r["lig_name"], r["rec_name"]] = r["prob"]
        label_matrix.loc[r["lig_name"], r["rec_name"]] = r["label"]

    bind_counts = (label_matrix == "BIND").sum(axis=1)
    prob_matrix["Promiscuity_Score"] = bind_counts / len(rec_names)

    csv_path = str(out_path).replace(".png", ".csv")
    prob_matrix.to_csv(csv_path)
    print(f"[output] Matrix CSV -> {csv_path}")

    plot_mat = prob_matrix.drop(columns=["Promiscuity_Score"]).astype(float)
    n_drugs, n_rec = len(lig_names), len(rec_names)

    # Each cell = 2.0 x 1.5 inches; extra right margin for promiscuity score column
    cell_w, cell_h = 2.0, 1.5
    left_margin  = 1.8   # inches for y-tick labels
    right_margin = 2.0   # inches for promiscuity score text + colorbar
    top_margin   = 1.0
    bot_margin   = 1.2   # inches for x-tick labels
    fig_w = n_rec  * cell_w + left_margin + right_margin
    fig_h = n_drugs * cell_h + top_margin + bot_margin

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Place axes explicitly so margins never get squeezed by tight_layout
    ax = fig.add_axes([
        left_margin / fig_w,
        bot_margin  / fig_h,
        (n_rec * cell_w) / fig_w,
        (n_drugs * cell_h) / fig_h,
    ])

    im = ax.imshow(plot_mat.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(n_rec))
    ax.set_xticklabels(rec_names, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(n_drugs))
    ax.set_yticklabels(lig_names, fontsize=11)
    ax.set_xlabel("Receptor", fontsize=12, labelpad=8)
    ax.set_ylabel("Ligand",   fontsize=12, labelpad=8)
    ax.set_title("Polypharmacology Screening Matrix  (Binding Probability)",
                 fontsize=13, fontweight="bold", pad=14)

    # Cell annotation: probability (large) + BIND/NO_BIND (small italic below)
    for i, lig in enumerate(lig_names):
        for j, rec in enumerate(rec_names):
            p   = plot_mat.loc[lig, rec]
            lbl = label_matrix.loc[lig, rec]
            txt_color = "white" if (p > 0.65 or p < 0.35) else "#1a1a1a"
            ax.text(j, i - 0.16, f"{p:.3f}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_color)
            ax.text(j, i + 0.26, lbl,
                    ha="center", va="center",
                    fontsize=9, color=txt_color, style="italic")
            if lbl == "BIND":
                ax.add_patch(plt.Rectangle(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="#27ae60", linewidth=2.5))

    # Promiscuity score: plain text annotations to the right of the axes
    scores = [prob_matrix.loc[lig, "Promiscuity_Score"] for lig in lig_names]
    ax_right_edge = (left_margin + n_rec * cell_w) / fig_w   # in figure fraction
    for i, (lig, score) in enumerate(zip(lig_names, scores)):
        # Convert data coords → figure fraction
        y_frac = (bot_margin + (n_drugs - 1 - i + 0.5) * cell_h) / fig_h
        fig.text(ax_right_edge + 0.01, y_frac,
                 f"  {score:.0%}",
                 va="center", ha="left",
                 fontsize=11, color="#8e44ad", fontweight="bold",
                 transform=fig.transFigure)
    fig.text(ax_right_edge + 0.01,
             (bot_margin + n_drugs * cell_h + 0.35) / fig_h,
             "Promiscuity\nScore",
             va="bottom", ha="left",
             fontsize=10, color="#8e44ad",
             transform=fig.transFigure)

    # Colorbar: placed in the remaining right margin
    cbar_ax = fig.add_axes([
        (left_margin + n_rec * cell_w + right_margin * 0.52) / fig_w,
        bot_margin / fig_h + 0.05,
        0.025,
        (n_drugs * cell_h) / fig_h * 0.85,
    ])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Binding Probability", fontsize=11, labelpad=10)

    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[output] Polypharmacology heatmap -> {out_path}")
    return prob_matrix


# ── 單對注意力圖 ──────────────────────────────────────────────

def plot_attention_map(
    attn_vec: np.ndarray,
    seq: str,
    drug_name: str,
    rec_name: str,
    prob: float,
    out_path: str,
    cols_per_row: int = 30,
) -> None:
    """Plot BAN residue attention map for a single drug-receptor pair."""
    seq_len = len(seq)
    attn = attn_vec[:seq_len]
    mn, mx = attn.min(), attn.max()
    attn_norm = (attn - mn) / (mx - mn + 1e-10)

    def to_color(f: float) -> str:
        if f > 0.75:   return "#e74c3c"   # red   -- very high attention
        elif f > 0.50: return "#e67e22"   # orange
        elif f > 0.25: return "#f1c40f"   # yellow
        else:          return "#ecf0f1"   # light gray -- low attention

    n_chunks    = (seq_len + cols_per_row - 1) // cols_per_row
    BOX_W       = 0.56
    BOX_H       = 0.70
    LEFT_MARGIN = 0.6
    row_height  = 1.20
    chunk_gap   = 0.65
    TITLE_H     = 0.90   # reserved height for title (two lines)
    LEGEND_H    = 0.80   # reserved height for legend at bottom

    fig_width    = cols_per_row * BOX_W + LEFT_MARGIN + 0.5
    total_height = n_chunks * (row_height + chunk_gap) + TITLE_H + LEGEND_H

    fig, ax = plt.subplots(figsize=(fig_width, total_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    title = (f"{drug_name}  ->  {rec_name}\n"
             f"Binding probability = {prob:.4f}   |   Sequence length = {seq_len} aa")
    ax.text(fig_width / 2, total_height - 0.12, title,
            ha="center", va="top", fontsize=10, fontweight="bold")

    for chunk_i in range(n_chunks):
        start = chunk_i * cols_per_row
        end   = min(start + cols_per_row, seq_len)
        # First chunk starts below the title block
        y_ctr = total_height - TITLE_H - 0.20 - chunk_i * (row_height + chunk_gap)

        for col_i, pos in enumerate(range(start, end)):
            x = LEFT_MARGIN + col_i * BOX_W
            f = attn_norm[pos]
            fc = to_color(f)

            rect = mpatches.FancyBboxPatch(
                (x + 0.03, y_ctr - BOX_H / 2), BOX_W - 0.06, BOX_H,
                boxstyle="round,pad=0.03",
                facecolor=fc, edgecolor="#bdc3c7", linewidth=0.4,
            )
            ax.add_patch(rect)

            txt_color = "white" if f > 0.50 else "#2c3e50"
            ax.text(x + BOX_W / 2, y_ctr, seq[pos],
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold" if f > 0.50 else "normal",
                    color=txt_color, fontfamily="monospace")

            # Residue number label (1-based) every 10 positions
            if pos == 0 or (pos + 1) % 10 == 0:
                ax.text(x + BOX_W / 2, y_ctr + 0.42, str(pos + 1),
                        ha="center", va="bottom", fontsize=5.5, color="#666")

    # Legend — boxes sized to match text height
    legend_items = [
        ("#e74c3c", "#c0392b", "> 75%   Very high"),
        ("#e67e22", "#d35400", "50-75%  High"),
        ("#f1c40f", "#d4ac0d", "25-50%  Moderate"),
        ("#ecf0f1", "#bdc3c7", "< 25%   Low"),
    ]
    box_h  = 0.38   # box height matches text cap-height
    box_w  = 0.38   # square-ish box
    gap    = 3.2    # horizontal spacing between legend items
    lx     = 0.4
    ly     = LEGEND_H * 0.52   # vertical center of legend area
    for fc, ec, label in legend_items:
        r = mpatches.FancyBboxPatch(
            (lx, ly - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.03",
            facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(r)
        ax.text(lx + box_w + 0.15, ly, label,
                ha="left", va="center", fontsize=8.5)
        lx += gap

    ax.text(fig_width / 2, 0.12,
            "BAN Attention Map  |  Color = normalized attention weight over protein residues",
            ha="center", va="bottom", fontsize=7.5, color="#555", style="italic")

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


# ── 文字摘要 ─────────────────────────────────────────────────

def print_summary(results: list[dict], ligands: list[tuple], receptors: list[tuple]) -> None:
    print("\n" + "=" * 65)
    print(f"Polypharmacology Summary  ({len(ligands)} ligands x {len(receptors)} receptors)")
    print("=" * 65)
    for lig_name, _ in ligands:
        pair_results = [r for r in results if r["lig_name"] == lig_name]
        bind_recs = [r["rec_name"] for r in pair_results if r["label"] == "BIND"]
        score = len(bind_recs) / len(receptors)
        bar = "#" * int(score * 20)
        print(f"  {lig_name:<22} [{bar:<20}] {score:.0%}  "
              f"BIND: {', '.join(bind_recs) if bind_recs else '(none)'}")
    print()


# ── 主程式 ───────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Polypharmacology analysis + BAN attention map output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ligands",   required=True, help="Ligand file: Name<Tab>SMILES per line")
    p.add_argument("--receptors", required=True, help="Receptor FASTA file")
    p.add_argument("--model",     default="BiLSTM", choices=["BiLSTM", "CNN"])
    p.add_argument("--out_dir",   default="results/promiscuity", help="Output directory")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--attn_all",  action="store_true",
                   help="Generate attention maps for ALL pairs (default: BIND pairs only)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    attn_dir = out_dir / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    attn_dir.mkdir(exist_ok=True)

    ligands   = parse_ligands(args.ligands)
    receptors = parse_fasta(args.receptors)
    print(f"[input] {len(ligands)} ligands, {len(receptors)} receptors")

    # 推論
    results = predict_all_pairs(ligands, receptors, args.model, args.batch_size)

    # 共顎性熱圖
    matrix_png = out_dir / "promiscuity_heatmap.png"
    plot_promiscuity_matrix(results, ligands, receptors, str(matrix_png))

    # 注意力圖（BIND 對，或全部）
    attn_count = 0
    for r in results:
        if r["label"] == "BIND" or args.attn_all:
            safe_lig = r["lig_name"].replace("/", "_").replace(" ", "_")
            safe_rec = r["rec_name"].replace("/", "_").replace(" ", "_")
            out_path = attn_dir / f"{safe_lig}__{safe_rec}.png"
            plot_attention_map(
                r["attn_vec"], r["seq"],
                r["lig_name"], r["rec_name"],
                r["prob"], str(out_path),
            )
            attn_count += 1

    print(f"[output] Attention maps: {attn_count} files -> {attn_dir}/")
    print_summary(results, ligands, receptors)
    print(f"Done. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
