#!/usr/bin/env python3
"""Make a clear summary figure from the ChEMBL on/off-target report.

Panels:
  1. On-target outcome         (confirmed vs missed)
  2. Off-target count per compound
  3. On-target confirmation rate vs measured affinity   (model validation)
  4. On-target predicted-probability distribution

Usage:
  python plot_on_off_target.py --report chembl_on_off_target_report.csv \
                               --out chembl_on_off_target_summary.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD = 0.1511


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="chembl_on_off_target_report.csv")
    p.add_argument("--out", default="chembl_on_off_target_summary.png")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.report, usecols=[
        "on_affinity_nM", "on_target_prob", "on_target_confirmed", "n_offtarget"])
    n = len(df)
    conf = df["on_target_confirmed"].astype(str)
    prob = pd.to_numeric(df["on_target_prob"], errors="coerce")
    aff = pd.to_numeric(df["on_affinity_nM"], errors="coerce")
    noff = pd.to_numeric(df["n_offtarget"], errors="coerce").fillna(0).astype(int)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"TEIBAN on/off-target analysis — {n:,} ChEMBL compounds",
                 fontsize=15, fontweight="bold")

    # ---- Panel 1: on-target outcome ----
    yes = int((conf == "YES").sum())
    no = int((conf == "NO").sum())
    na = int((conf == "NOT_PREDICTED").sum())
    a = ax[0, 0]
    bars = a.bar(["Confirmed\n(BIND)", "Missed\n(NO_BIND)", "Not\npredictable"],
                 [yes, no, na], color=["#2ecc71", "#e74c3c", "#95a5a6"])
    for b, v in zip(bars, [yes, no, na]):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100*v/n:.1f}%)",
               ha="center", va="bottom", fontsize=10)
    a.set_title("1) On-target confirmation\n(is the strongest measured target predicted to bind?)")
    a.set_ylabel("compounds")
    a.set_ylim(0, max(yes, no, na) * 1.2)

    # ---- Panel 2: off-target count ----
    a = ax[0, 1]
    cats = ["0", "1", "2", "3", "4", "5+"]
    counts = [int((noff == k).sum()) for k in range(5)] + [int((noff >= 5).sum())]
    bars = a.bar(cats, counts, color="#3498db")
    for b, v in zip(bars, counts):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    a.set_title("2) Off-target hits per compound\n(other proteins predicted to bind)")
    a.set_xlabel("number of predicted off-targets")
    a.set_ylabel("compounds")

    # ---- Panel 3: confirmation rate vs affinity ----
    a = ax[1, 0]
    edges = [0, 1, 10, 100, 1e3, 1e4, np.inf]
    labels = ["<1", "1-10", "10-100", "100-1k", "1k-10k", ">10k"]
    rates, ns = [], []
    m = aff.notna()
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = m & (aff > lo) & (aff <= hi)
        tot = int(sel.sum())
        r = 100 * (conf[sel] == "YES").mean() if tot else 0
        rates.append(r)
        ns.append(tot)
    bars = a.bar(labels, rates, color="#9b59b6")
    for b, r, cnt in zip(bars, rates, ns):
        a.text(b.get_x() + b.get_width() / 2, r, f"{r:.0f}%\n(n={cnt:,})",
               ha="center", va="bottom", fontsize=8)
    a.set_title("3) On-target confirmation rate vs measured affinity\n(stronger binders should be recovered more)")
    a.set_xlabel("measured affinity (nM, smaller = stronger)")
    a.set_ylabel("% confirmed")
    a.set_ylim(0, 110)

    # ---- Panel 4: on-target probability distribution ----
    a = ax[1, 1]
    pv = prob.dropna().values
    a.hist(pv, bins=50, color="#f39c12", edgecolor="white")
    a.axvline(THRESHOLD, color="red", linestyle="--", label=f"threshold {THRESHOLD}")
    a.set_title("4) On-target predicted probability")
    a.set_xlabel("TEIBAN binding probability for the on-target")
    a.set_ylabel("compounds")
    a.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150)
    print(f"saved figure -> {args.out}")
    print(f"  on-target: confirmed {yes:,} ({100*yes/n:.1f}%), missed {no:,}, not-predictable {na:,}")
    print(f"  off-target: {int((noff>=1).sum()):,} compounds have >=1; max {int(noff.max())}")


if __name__ == "__main__":
    main()
