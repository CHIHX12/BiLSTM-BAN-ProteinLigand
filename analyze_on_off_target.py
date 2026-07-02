#!/usr/bin/env python3
"""On-target / off-target report from TEIBAN predictions on the ChEMBL dataset.

on-target  = each compound's STRONGEST measured target in ChEMBL (smallest nM
             value among exact "=" measurements; assay types are mixed, so this
             is an approximation of the primary target).
off-target = any OTHER protein TEIBAN predicts the compound BINDS.

Inputs:
  --pci   ChEMBL_hsa_PCI.txt            (for the on-target definition + affinity)
  --pred  TEIBAN prediction CSV         (columns: name=<cid>~<protein>, Y_pred_prob)
  --out   output report CSV
  --threshold  BIND threshold (default 0.1511 for BiLSTM; use 0.31 for CNN)

Output: one row per compound with its on-target, whether TEIBAN confirms it,
        and its predicted off-targets + a promiscuity score. Also prints a summary.

Usage:
  python analyze_on_off_target.py \
      --pci  chembl-dataset/chembl-dataset/ChEMBL_hsa_PCI.txt \
      --pred chembl_full_pred.csv \
      --out  chembl_on_off_target_report.csv
"""
import argparse
import csv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pci", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--out", default="chembl_on_off_target_report.csv")
    p.add_argument("--threshold", type=float, default=0.1511)
    p.add_argument("--max_offtargets_listed", type=int, default=10)
    return p.parse_args()


def load_on_target(pci_path):
    """compound -> (best_nM, protein_name, assay_type) using exact nM measurements."""
    best = {}
    for ln in open(pci_path, encoding="utf-8", errors="replace"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 7:
            continue
        cid, assay, val, rel, unit, _uni, pname = f[:7]
        if unit != "nM" or rel != "=":
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if v <= 0:
            continue
        if cid not in best or v < best[cid][0]:
            best[cid] = (v, pname, assay)
    return best


def load_predictions(pred_path):
    """compound -> {protein_name: prob}. Expects a 'name' column of <cid>~<protein>."""
    preds = {}
    with open(pred_path, encoding="utf-8", errors="replace") as fh:
        r = csv.DictReader(fh)
        prob_col = next((c for c in ("Y_pred_prob", "binding_prob") if c in r.fieldnames), None)
        if "name" not in (r.fieldnames or []) or not prob_col:
            raise SystemExit(f"ERROR: prediction CSV must have 'name' and a probability "
                             f"column; found {r.fieldnames}")
        for row in r:
            name = row["name"]
            if "~" not in name:
                continue
            cid, pname = name.split("~", 1)
            try:
                prob = float(row[prob_col])
            except (ValueError, TypeError):
                continue
            preds.setdefault(cid, {})[pname] = prob
    return preds


def main():
    args = parse_args()
    thr = args.threshold
    print(f"[1/3] loading on-target map from {args.pci} ...", flush=True)
    on_target = load_on_target(args.pci)
    print(f"      compounds with an exact-nM on-target: {len(on_target)}")

    print(f"[2/3] loading predictions from {args.pred} (threshold={thr}) ...", flush=True)
    preds = load_predictions(args.pred)
    print(f"      compounds predicted: {len(preds)}")

    print(f"[3/3] building on/off-target report -> {args.out} ...", flush=True)
    n_conf = n_on_pred = n_any_off = 0
    total_off = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["compound", "on_target", "on_affinity_nM", "on_assay",
                    "on_target_prob", "on_target_confirmed", "n_targets_predicted",
                    "n_bind", "n_offtarget", "promiscuity", "offtarget_examples"])
        for cid, tmap in preds.items():
            n_pred = len(tmap)
            binders = {p: pr for p, pr in tmap.items() if pr >= thr}
            n_bind = len(binders)
            info = on_target.get(cid)
            if info:
                aff, on_p, assay = info
            else:
                # no exact-nM measurement: fall back to the strongest predicted target
                on_p = max(tmap, key=tmap.get)
                aff, assay = "", "predicted"
            on_prob = tmap.get(on_p)
            on_conf = on_prob is not None and on_prob >= thr
            if on_prob is not None:
                n_on_pred += 1
            if on_conf:
                n_conf += 1
            offs = sorted(((p, pr) for p, pr in binders.items() if p != on_p),
                          key=lambda x: -x[1])
            if offs:
                n_any_off += 1
            total_off += len(offs)
            examples = "; ".join(f"{p}({pr:.2f})" for p, pr in offs[:args.max_offtargets_listed])
            w.writerow([cid, on_p, aff, assay,
                        f"{on_prob:.4f}" if on_prob is not None else "NA",
                        "YES" if on_conf else ("NO" if on_prob is not None else "NOT_PREDICTED"),
                        n_pred, n_bind, len(offs),
                        f"{n_bind / n_pred:.3f}" if n_pred else "0", examples])

    n = len(preds)
    print("=" * 60)
    print(f"SUMMARY ({n} compounds)")
    print(f"  on-target predictable         : {n_on_pred}")
    print(f"  on-target CONFIRMED (BIND)     : {n_conf}"
          + (f"  ({100 * n_conf / n_on_pred:.1f}% of predictable)" if n_on_pred else ""))
    print(f"  compounds with >=1 off-target : {n_any_off}  ({100 * n_any_off / n:.1f}%)")
    print(f"  total off-target hits         : {total_off}"
          + (f"  (avg {total_off / n:.2f} per compound)" if n else ""))
    print(f"  report: {args.out}")


if __name__ == "__main__":
    main()
