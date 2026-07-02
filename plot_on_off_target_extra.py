#!/usr/bin/env python3
"""Three extra figures from the ChEMBL on/off-target report:

  A. on-target predicted probability vs measured affinity  (does the model track
     real affinity?)  -> chembl_prob_vs_affinity.png
  B. most promiscuous compounds (top-N by off-target count) -> chembl_top_promiscuous.png
  C. on-target confirmation rate vs protein length (validates the long-protein
     truncation weakness at 1200 aa)                       -> chembl_confirm_vs_length.png

Usage:
  python plot_on_off_target_extra.py \
      --report chembl_on_off_target_report.csv \
      --proteome chembl-dataset/chembl-dataset/hsa.proteome \
      --topn 30
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_PROTEIN_LEN = 1200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="chembl_on_off_target_report.csv")
    p.add_argument("--proteome", default="chembl-dataset/chembl-dataset/hsa.proteome")
    p.add_argument("--topn", type=int, default=30)
    return p.parse_args()


def load_protein_lengths(path):
    lengths, name, n = {}, None, 0
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.rstrip("\n")
        if ln.startswith(">"):
            if name:
                lengths[name] = n
            name, n = ln[1:].strip(), 0
        elif ln.strip():
            n += len(ln.strip())
    if name:
        lengths[name] = n
    return lengths


def main():
    args = parse_args()
    df = pd.read_csv(args.report, usecols=[
        "compound", "on_target", "on_affinity_nM", "on_target_prob",
        "on_target_confirmed", "n_offtarget"])
    conf = df["on_target_confirmed"].astype(str)
    prob = pd.to_numeric(df["on_target_prob"], errors="coerce")
    aff = pd.to_numeric(df["on_affinity_nM"], errors="coerce")
    noff = pd.to_numeric(df["n_offtarget"], errors="coerce").fillna(0).astype(int)

    # ---- A. prob vs measured affinity (hexbin) ----
    m = aff.notna() & prob.notna() & (aff > 0)
    pAff = 9.0 - np.log10(aff[m].values)          # pKd-like: bigger = stronger
    y = prob[m].values
    fig, a = plt.subplots(figsize=(8, 6))
    hb = a.hexbin(pAff, y, gridsize=55, bins="log", cmap="viridis", mincnt=1)
    fig.colorbar(hb, label="compounds (log scale)")
    a.axhline(0.1511, color="red", ls="--", lw=1, label="BIND threshold 0.1511")
    a.set_xlabel("measured affinity  pAff = 9 - log10(nM)   (larger = stronger)")
    a.set_ylabel("TEIBAN on-target binding probability")
    a.set_title(f"A) Predicted probability vs measured affinity  (n={m.sum():,})")
    a.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("chembl_prob_vs_affinity.png", dpi=150); plt.close(fig)
    print("saved chembl_prob_vs_affinity.png")

    # ---- B. top promiscuous compounds ----
    top = df.sort_values("n_offtarget", ascending=False).head(args.topn)
    fig, a = plt.subplots(figsize=(9, max(6, args.topn * 0.3)))
    a.barh(top["compound"][::-1], top["n_offtarget"][::-1], color="#e67e22")
    a.set_xlabel("number of predicted off-targets")
    a.set_title(f"B) Top {args.topn} most promiscuous compounds")
    for i, (c, v) in enumerate(zip(top["compound"][::-1], top["n_offtarget"][::-1])):
        a.text(v, i, f" {v}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig("chembl_top_promiscuous.png", dpi=150); plt.close(fig)
    print("saved chembl_top_promiscuous.png")

    # ---- C. confirmation rate vs on-target protein length ----
    lengths = load_protein_lengths(args.proteome)
    plen = df["on_target"].map(lengths)
    edges = [0, 500, 1000, MAX_PROTEIN_LEN, 2000, np.inf]
    labels = ["<=500", "500-1000", f"1000-{MAX_PROTEIN_LEN}",
              f"{MAX_PROTEIN_LEN}-2000", ">2000"]
    rates, ns = [], []
    valid = plen.notna()
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = valid & (plen > lo) & (plen <= hi)
        tot = int(sel.sum())
        rates.append(100 * (conf[sel] == "YES").mean() if tot else 0)
        ns.append(tot)
    fig, a = plt.subplots(figsize=(9, 6))
    colors = ["#2ecc71", "#2ecc71", "#2ecc71", "#e74c3c", "#e74c3c"]
    bars = a.bar(labels, rates, color=colors)
    for b, r, cnt in zip(bars, rates, ns):
        a.text(b.get_x() + b.get_width() / 2, r, f"{r:.0f}%\n(n={cnt:,})",
               ha="center", va="bottom", fontsize=9)
    a.axvline(2.5, color="gray", ls=":", lw=1)
    a.text(3.5, 95, f"truncated at {MAX_PROTEIN_LEN} aa", color="#e74c3c", ha="center")
    a.set_ylabel("% on-target confirmed")
    a.set_xlabel("on-target protein length (aa)")
    a.set_title("C) Confirmation rate vs protein length\n(long proteins are truncated -> lower recall)")
    a.set_ylim(0, 110)
    fig.tight_layout(); fig.savefig("chembl_confirm_vs_length.png", dpi=150); plt.close(fig)
    print("saved chembl_confirm_vs_length.png")
    print("  length-bin confirmation rates:",
          {l: f"{r:.0f}% (n={c})" for l, r, c in zip(labels, rates, ns)})


if __name__ == "__main__":
    main()
