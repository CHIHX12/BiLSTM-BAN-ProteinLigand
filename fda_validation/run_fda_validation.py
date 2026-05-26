"""
run_fda_validation.py
=====================
Step 2: Run FDA drug validation using predict.py infrastructure.

Usage:
    cd GCN-BILSTM-BAN-bak
    python fda_validation/run_fda_validation.py [--model BiLSTM|CNN|both]
    python fda_validation/run_fda_validation.py --curated_only   # fast sanity check
"""

import sys, os, argparse, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).parent.parent          # GCN-BILSTM-BAN-bak/
OUT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from predict import predict_df                  # reuse existing inference pipeline

RAW_CSV  = OUT_DIR / "raw_pairs.csv"
RESULTS  = OUT_DIR / "fda_results.csv"
REPORT   = OUT_DIR / "fda_validation_report.txt"
ROCCURVE = OUT_DIR / "fda_roc_curve.png"

# Optimal thresholds from training
THRESHOLDS = {"BiLSTM": 0.1511, "CNN": 0.3100}


# ── curated gold-standard pairs ───────────────────────────────────────────────
# Well-known FDA drug–target pairs with confirmed binding/non-binding.
# Sources: DrugBank, ChEMBL, literature.

CURATED_POSITIVES = [
    # (drug_name, SMILES, UniProt, note)
    ("Imatinib",
     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
     "P00519", "BCR-ABL1 — primary target, Ki~1nM"),
    ("Erlotinib",
     "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
     "P00533", "EGFR — primary target, IC50~2nM"),
    ("Tamoxifen",
     "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
     "P03372", "ESR1 — primary target"),
    ("Methotrexate",
     "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
     "P00374", "DHFR — primary target, Ki~1pM"),
    ("Sildenafil",
     "CCCC1=NN(C)C(=O)c2[nH]c(-c3ccccc3S(=O)(=O)N3CCN(CC)CC3)nc21",
     "O76074", "PDE5A — primary target"),
    ("Atorvastatin",
     "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
     "P04035", "HMGCR — primary target"),
    ("Aspirin",
     "CC(=O)Oc1ccccc1C(=O)O",
     "P23219", "PTGS1/COX-1 — primary target"),
    ("Ibuprofen",
     "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
     "P35354", "PTGS2/COX-2 — primary target"),
    ("Gefitinib",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
     "P00533", "EGFR — approved for NSCLC"),
    ("Osimertinib",
     "C=CC(=O)Nc1cc2c(Nc3cccc(NC(=O)/C=C/CN(C)C)c3)ncnc2cc1N(C)CCN(C)C",
     "P00533", "EGFR T790M — 3rd gen TKI"),
    ("Lapatinib",
     "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
     "P00533", "EGFR/HER2 dual inhibitor"),
    ("Dasatinib",
     "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
     "P00519", "BCR-ABL / Src — 2nd gen TKI"),
    ("Nilotinib",
     "Cc1ccc(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)ccn2)cc1C(F)(F)F",
     "P00519", "BCR-ABL — 2nd gen TKI"),
    ("Anastrozole",
     "N#CC(C)(C)c1cc(Cc2cc(CC(C)(C#N)C#N)cc(C)c2)cc(C)c1",
     "P11511", "CYP19A1/Aromatase — breast cancer"),
    ("Letrozole",
     "N#Cc1ccc(C(Cn2cncn2)c2ccc(C#N)cc2)cc1",
     "P11511", "CYP19A1/Aromatase — breast cancer"),
]

CURATED_NEGATIVES = [
    # (drug_name, SMILES, UniProt, note)
    # Each drug paired with a protein that is NOT its target
    ("Imatinib",
     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
     "P03372", "Imatinib vs ESR1 — unrelated target"),
    ("Imatinib",
     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
     "P00374", "Imatinib vs DHFR — unrelated target"),
    ("Tamoxifen",
     "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
     "P00519", "Tamoxifen vs BCR-ABL — unrelated"),
    ("Tamoxifen",
     "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
     "P04035", "Tamoxifen vs HMGCR — unrelated"),
    ("Aspirin",
     "CC(=O)Oc1ccccc1C(=O)O",
     "P00533", "Aspirin vs EGFR — unrelated"),
    ("Aspirin",
     "CC(=O)Oc1ccccc1C(=O)O",
     "P03372", "Aspirin vs ESR1 — unrelated"),
    ("Sildenafil",
     "CCCC1=NN(C)C(=O)c2[nH]c(-c3ccccc3S(=O)(=O)N3CCN(CC)CC3)nc21",
     "P04035", "Sildenafil vs HMGCR — unrelated"),
    ("Sildenafil",
     "CCCC1=NN(C)C(=O)c2[nH]c(-c3ccccc3S(=O)(=O)N3CCN(CC)CC3)nc21",
     "P00374", "Sildenafil vs DHFR — unrelated"),
    ("Methotrexate",
     "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
     "P23219", "Methotrexate vs COX-1 — unrelated"),
    ("Atorvastatin",
     "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
     "O76074", "Atorvastatin vs PDE5A — unrelated"),
    ("Gefitinib",
     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
     "P04035", "Gefitinib vs HMGCR — unrelated"),
    ("Erlotinib",
     "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
     "P23219", "Erlotinib vs COX-1 — unrelated"),
    ("Ibuprofen",
     "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
     "P00519", "Ibuprofen vs BCR-ABL — unrelated"),
    ("Letrozole",
     "N#Cc1ccc(C(Cn2cncn2)c2ccc(C#N)cc2)cc1",
     "P00533", "Letrozole vs EGFR — unrelated"),
    ("Anastrozole",
     "N#CC(C)(C)c1cc(Cc2cc(CC(C)(C#N)C#N)cc(C)c2)cc(C)c1",
     "P03372", "Anastrozole vs ESR1 — different enzyme"),
]


def fetch_sequence(uniprot_id, cache={}):
    if uniprot_id in cache:
        return cache[uniprot_id]
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            seq = "".join(r.text.strip().split("\n")[1:])
            cache[uniprot_id] = seq
            return seq
    except Exception as e:
        print(f"  [warn] UniProt fetch failed for {uniprot_id}: {e}")
    cache[uniprot_id] = None
    return None


def build_curated_df():
    rows = []
    print("Fetching sequences for curated pairs from UniProt...")
    all_pairs = [(note, smiles, uid, 1) for (_, smiles, uid, note) in CURATED_POSITIVES] + \
                [(note, smiles, uid, 0) for (_, smiles, uid, note) in CURATED_NEGATIVES]

    for note, smiles, uid, label in all_pairs:
        seq = fetch_sequence(uid)
        if seq and len(seq) >= 50:
            rows.append({
                "SMILES":        smiles,
                "Protein":       seq,
                "Y":             label,
                "note":          note,
                "uniprot_id":    uid,
                "source":        "curated",
            })
        else:
            print(f"  [skip] Could not fetch sequence for {uid}")
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    print(f"  Curated: {(df['Y']==1).sum()} positives, {(df['Y']==0).sum()} negatives")
    return df


def compute_metrics(labels, probs, threshold):
    preds = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    f1    = f1_score(labels, preds, zero_division=0)
    acc   = accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(auroc=auroc, auprc=auprc, f1=f1, accuracy=acc,
                sensitivity=sens, specificity=spec,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def plot_roc_curve(roc_data, out_path):
    plt.figure(figsize=(6, 5))
    for label, (labels, probs) in roc_data.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — FDA Drug External Validation", fontsize=13)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ROC curve: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["BiLSTM", "CNN", "both"], default="both")
    ap.add_argument("--curated_only", action="store_true",
                    help="Use only the 30 curated pairs (fast sanity check)")
    ap.add_argument("--batch_size", type=int, default=32)
    return ap.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    os.chdir(ROOT)   # predict.py loads checkpoints relative to ROOT

    # --- build validation dataframe ---
    curated_df = build_curated_df()

    if args.curated_only or not RAW_CSV.exists():
        val_df = curated_df
        source_note = "curated gold-standard pairs only"
    else:
        chembl_df = pd.read_csv(RAW_CSV)
        chembl_df = chembl_df.rename(columns={
            "smiles":       "SMILES",
            "protein_seq":  "Protein",
            "label":        "Y",
        })
        chembl_df["source"] = "chembl"
        chembl_df = chembl_df[["SMILES","Protein","Y","uniprot_id","source"]]
        val_df = pd.concat([curated_df, chembl_df], ignore_index=True)
        val_df = val_df.drop_duplicates(subset=["SMILES","Protein","Y"])
        source_note = f"curated + ChEMBL FDA pairs"

    print(f"\nValidation set: {len(val_df)} pairs ({source_note})")
    print(f"  BIND    : {(val_df['Y']==1).sum()}")
    print(f"  NO BIND : {(val_df['Y']==0).sum()}")

    models = ["BiLSTM","CNN"] if args.model == "both" else [args.model]

    report_lines = [
        "FDA Drug External Validation Report",
        "====================================",
        f"Source         : {source_note}",
        f"Total pairs    : {len(val_df)}",
        f"BIND (pos)     : {(val_df['Y']==1).sum()}",
        f"NO BIND (neg)  : {(val_df['Y']==0).sum()}",
        "",
    ]

    roc_data     = {}
    all_res_rows = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        result_df = predict_df(val_df, model_name=model_name,
                               batch_size=args.batch_size)

        probs  = result_df["Y_pred_prob"].values
        labels = val_df["Y"].values
        thr    = THRESHOLDS[model_name]
        m      = compute_metrics(labels, probs, thr)
        roc_data[model_name] = (labels, probs)

        print(f"\n  AUROC       : {m['auroc']:.4f}")
        print(f"  AUPRC       : {m['auprc']:.4f}")
        print(f"  F1          : {m['f1']:.4f}")
        print(f"  Accuracy    : {m['accuracy']:.4f}")
        print(f"  Sensitivity : {m['sensitivity']:.4f}  (recall of binders)")
        print(f"  Specificity : {m['specificity']:.4f}  (recall of non-binders)")
        print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")

        # Show curated results in detail
        if "note" in val_df.columns:
            print(f"\n  --- Curated pair breakdown ({model_name}) ---")
            curated_mask = val_df["source"] == "curated"
            sub = val_df[curated_mask].copy()
            sub_probs = probs[curated_mask.values]
            for i, (_, row) in enumerate(sub.iterrows()):
                pred = "BIND" if sub_probs[i] >= thr else "NO_BIND"
                true = "BIND" if row["Y"] == 1 else "NO_BIND"
                ok   = "✓" if pred == true else "✗"
                print(f"  {ok} [{true:7s}→{pred:7s} p={sub_probs[i]:.3f}] {row['note']}")

        report_lines += [
            f"Model: {model_name}",
            "-" * 40,
            f"  AUROC       : {m['auroc']:.4f}",
            f"  AUPRC       : {m['auprc']:.4f}",
            f"  F1          : {m['f1']:.4f}",
            f"  Accuracy    : {m['accuracy']:.4f}",
            f"  Sensitivity : {m['sensitivity']:.4f}",
            f"  Specificity : {m['specificity']:.4f}",
            f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}",
            "",
        ]

        for i, (_, row) in enumerate(val_df.iterrows()):
            all_res_rows.append({
                "model":      model_name,
                "note":       row.get("note", ""),
                "uniprot_id": row.get("uniprot_id", ""),
                "true_label": int(labels[i]),
                "pred_prob":  round(float(probs[i]), 4),
                "pred_label": int(probs[i] >= thr),
                "correct":    int((probs[i] >= thr) == (labels[i] == 1)),
                "source":     row.get("source", ""),
            })

    # save
    pd.DataFrame(all_res_rows).to_csv(RESULTS, index=False)
    print(f"\nPer-pair results: {RESULTS}")

    with open(REPORT, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report         : {REPORT}")

    if len(roc_data) >= 1:
        plot_roc_curve(roc_data, ROCCURVE)

    print("\nDone!")


if __name__ == "__main__":
    main()
