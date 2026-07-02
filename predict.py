#!/usr/bin/env python3
"""
predict.py
==========
Standalone inference script for GCN-BiLSTM-BAN.

Three usage modes:

  1. Single pair (1 drug + 1 protein):
       python predict.py --drug "CCO" --protein "MKVL..." --model BiLSTM

  2. Batch prediction from CSV:
       python predict.py --input pairs.csv --model BiLSTM --output results.csv

  3. High-throughput screening:
       python predict.py --screen --drug "CCO" --protein_list proteins.txt --model BiLSTM
       python predict.py --screen --protein "MKVL..." --drug_list drugs.txt   --model BiLSTM

CSV input format (--input mode):
    SMILES,Protein,Y        <- Y can be 0 (dummy), it is ignored during prediction
    CCO,MKVL...,0

Output columns:
    SMILES, Protein, Y_true, Y_pred_prob, Y_pred_label, attention_top10
"""
import sys, os, warnings, argparse
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import get_cfg_defaults
from models import DrugBAN
from dataloader import DTIDataset
from utils import graph_collate_func

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    "BiLSTM": {
        "cfg":       os.path.join(ROOT, "configs/DrugBAN_BiLSTM.yaml"),
        "ckpt":      os.path.join(ROOT, "result/DrugBAN_BiLSTM/best_model_epoch_94.pth"),
        "threshold": 0.1511,   # optimal threshold from test set (AUROC=0.9641, Acc=0.9065)
    },
    "CNN": {
        "cfg":       os.path.join(ROOT, "configs/DrugBAN.yaml"),
        "ckpt":      os.path.join(ROOT, "result/DrugBAN/best_model_epoch_90.pth"),
        "threshold": 0.3100,   # optimal threshold from test set (AUROC=0.9603, Acc=0.9067)
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def load_model(model_name: str):
    info = MODEL_CONFIGS[model_name]
    cfg = get_cfg_defaults()
    cfg.merge_from_file(info["cfg"])
    cfg.freeze()
    model = DrugBAN(**cfg).to(DEVICE)
    state = torch.load(info["ckpt"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    threshold = info["threshold"]
    print(f"[predict] Loaded {model_name} from {info['ckpt']}")
    print(f"[predict] Device: {DEVICE}  |  Optimal threshold: {threshold}")
    return model, cfg, threshold


def predict_df(df: pd.DataFrame, model_name: str = "BiLSTM", batch_size: int = 32,
               num_workers: int = 0):
    """
    Run prediction on a DataFrame with columns [SMILES, Protein, Y].
    Y column is ignored (can be dummy 0).
    Returns DataFrame with added columns: Y_pred_prob, Y_pred_label, attention_top10
    """
    model, cfg, threshold = load_model(model_name)

    # Match training: use physicochemical features when USE_BILSTM=True (main.py line 55)
    use_features = cfg.PROTEIN.get("USE_BILSTM", False)

    # Build dataset
    df = df.copy()
    if "Y" not in df.columns:
        df["Y"] = 0
    dataset = DTIDataset(
        list(range(len(df))),
        df,
        max_drug_nodes=290,
        max_protein_length=1200,
        use_features=use_features,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graph_collate_func,
    )

    all_probs = []
    all_attn_top10 = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicting ({model_name})", unit="batch", leave=False):
            v_d, v_p, labels = batch
            if isinstance(v_d, (list, tuple)):
                v_d = tuple(x.to(DEVICE) for x in v_d)
            else:
                v_d = v_d.to(DEVICE)
            if isinstance(v_p, (list, tuple)):
                v_p = tuple(x.to(DEVICE) for x in v_p)
            else:
                v_p = v_p.to(DEVICE)

            _, _, score, attn = model(v_d, v_p, mode="eval")
            prob = torch.sigmoid(score).squeeze(-1).cpu().numpy()
            if prob.ndim == 0:
                prob = np.array([float(prob)])
            all_probs.extend(prob.tolist())

            # Attention: top-10 residue positions per sample
            # attn shape from model: (batch, heads, seq, seq) or (batch, seq)
            attn_cpu = attn.cpu()
            for i in range(attn_cpu.shape[0]):
                a = attn_cpu[i]          # (heads, seq, seq) or (seq,)
                a = a.squeeze(0)         # remove leading dim if present
                if a.dim() == 3:
                    a = a.mean(0)        # average over heads -> (seq, seq)
                if a.dim() == 2:
                    a = a.sum(0)         # sum over drug dim -> (seq,)
                a_np = a.numpy()
                seq_len = a_np.shape[0]
                top10 = np.argsort(a_np)[::-1][:10].tolist()
                all_attn_top10.append(top10)

    df = df.copy()
    df["Y_pred_prob"]  = all_probs
    df["Y_pred_label"] = (np.array(all_probs) >= threshold).astype(int).tolist()
    df["attention_top10_residues"] = [str(t) for t in all_attn_top10]
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GCN-BiLSTM-BAN prediction")
    p.add_argument("--model",   default="BiLSTM", choices=["BiLSTM", "CNN"],
                   help="Which model to use (default: BiLSTM)")
    p.add_argument("--output",  default="predictions.csv",
                   help="Output CSV path (default: predictions.csv)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0,
                   help="Parallel data-loading workers for molecule featurization "
                        "(0 = single-thread; use 8-16 to speed up large --input runs)")

    # Mode A: single pair
    p.add_argument("--drug",    type=str, help="Single drug SMILES")
    p.add_argument("--protein", type=str, help="Single protein amino acid sequence")

    # Mode B: batch CSV
    p.add_argument("--input",   type=str, help="CSV with columns SMILES,Protein[,Y]")

    # Mode C: screening
    p.add_argument("--screen",       action="store_true",
                   help="High-throughput screening mode")
    p.add_argument("--drug_list",    type=str,
                   help="Text file: one SMILES per line (used with --protein)")
    p.add_argument("--protein_list", type=str,
                   help="Text file: one sequence per line (used with --drug)")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Mode A: single pair -----------------------------------------------
    if args.drug and args.protein and not args.screen:
        print(f"\n[predict] Mode: single pair")
        df = pd.DataFrame([{"SMILES": args.drug, "Protein": args.protein, "Y": 0}])
        result = predict_df(df, model_name=args.model, batch_size=1)
        prob  = result["Y_pred_prob"].iloc[0]
        label = result["Y_pred_label"].iloc[0]
        top10 = result["attention_top10_residues"].iloc[0]
        thr   = MODEL_CONFIGS[args.model]["threshold"]
        print(f"\n  Drug:    {args.drug[:60]}...")
        print(f"  Protein: {args.protein[:40]}... (len={len(args.protein)})")
        print(f"\n  Binding probability : {prob:.4f}")
        print(f"  Predicted label     : {'BIND' if label else 'NO BIND'} (threshold={thr})")
        print(f"  Top-10 attention residues: {top10}")
        result.to_csv(args.output, index=False)
        print(f"\n  Saved: {args.output}")

    # ---- Mode B: batch CSV -------------------------------------------------
    elif args.input:
        print(f"\n[predict] Mode: batch CSV ({args.input})")
        df = pd.read_csv(args.input)
        required = {"SMILES", "Protein"}
        if not required.issubset(df.columns):
            print(f"ERROR: CSV must have columns: {required}")
            sys.exit(1)
        result = predict_df(df, model_name=args.model, batch_size=args.batch_size,
                            num_workers=args.num_workers)
        result.to_csv(args.output, index=False)
        n_bind = (result["Y_pred_label"] == 1).sum()
        print(f"\n  Total pairs : {len(result)}")
        print(f"  Predicted BIND    : {n_bind}")
        print(f"  Predicted NO BIND : {len(result) - n_bind}")
        print(f"  Saved: {args.output}")

    # ---- Mode C: screening -------------------------------------------------
    elif args.screen:
        if args.drug and args.protein_list:
            # 1 drug vs many proteins
            print(f"\n[predict] Mode: 1 drug vs many proteins")
            with open(args.protein_list) as f:
                proteins = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({
                "SMILES":  [args.drug] * len(proteins),
                "Protein": proteins,
                "Y": 0,
            })
            print(f"  Drug: {args.drug[:60]}")
            print(f"  Proteins: {len(proteins)}")
        elif args.protein and args.drug_list:
            # 1 protein vs many drugs
            print(f"\n[predict] Mode: 1 protein vs many drugs")
            with open(args.drug_list) as f:
                drugs = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({
                "SMILES":  drugs,
                "Protein": [args.protein] * len(drugs),
                "Y": 0,
            })
            print(f"  Protein: {args.protein[:40]}... (len={len(args.protein)})")
            print(f"  Drugs: {len(drugs)}")
        else:
            print("ERROR: --screen requires (--drug + --protein_list) or (--protein + --drug_list)")
            sys.exit(1)

        result = predict_df(df, model_name=args.model, batch_size=args.batch_size,
                            num_workers=args.num_workers)
        result_sorted = result.sort_values("Y_pred_prob", ascending=False)
        result_sorted.to_csv(args.output, index=False)

        print(f"\n  Top-5 binding candidates:")
        for _, row in result_sorted.head(5).iterrows():
            seq_or_smi = row["Protein"][:30] if args.drug else row["SMILES"][:40]
            print(f"    prob={row['Y_pred_prob']:.4f}  {seq_or_smi}...")
        print(f"\n  Full results saved (sorted by prob): {args.output}")

    else:
        print("ERROR: specify --drug + --protein, --input, or --screen mode")
        print("       Run with --help for usage")
        sys.exit(1)


if __name__ == "__main__":
    main()
