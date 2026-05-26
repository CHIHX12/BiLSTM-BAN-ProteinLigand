"""
fetch_fda_pairs.py
==================
Step 1: Download FDA-approved drug–target pairs from ChEMBL.

Positive pairs : pChEMBL >= 6  (IC50/Ki ≤ 1 µM)   → label 1
Negative pairs : pChEMBL <= 4  (IC50/Ki ≥ 100 µM)  → label 0

Output: fda_validation/raw_pairs.csv
"""

import os, time, requests, json
import pandas as pd
from pathlib import Path

CHEMBL_API  = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"
OUT_DIR     = Path(__file__).parent
RAW_CSV     = OUT_DIR / "raw_pairs.csv"

PCHEMBL_POS = 6.0   # ≥ 6  → binder
PCHEMBL_NEG = 4.0   # ≤ 4  → non-binder
MAX_DRUGS   = 500   # FDA drugs to query (page through ChEMBL)
MAX_PAIRS   = 3000  # cap total pairs fetched


# ── helpers ──────────────────────────────────────────────────────────────────

def chembl_get(endpoint, params, retries=3):
    url = f"{CHEMBL_API}/{endpoint}.json"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return None


def fetch_fda_drugs(max_drugs=MAX_DRUGS):
    """Return list of {chembl_id, smiles} for FDA-approved small molecules."""
    drugs, offset = [], 0
    limit = 100
    print(f"Fetching FDA-approved drugs from ChEMBL (target: {max_drugs})...")
    while len(drugs) < max_drugs:
        data = chembl_get("molecule", {
            "max_phase": 4,
            "molecule_type": "Small molecule",
            "limit": limit,
            "offset": offset,
            "format": "json",
        })
        if not data:
            break
        mols = data.get("molecules", [])
        if not mols:
            break
        for m in mols:
            smiles = (m.get("molecule_structures") or {}).get("canonical_smiles")
            cid    = m.get("molecule_chembl_id")
            name   = m.get("pref_name") or cid
            if smiles and cid:
                drugs.append({"chembl_id": cid, "name": name, "smiles": smiles})
        offset += limit
        print(f"  {len(drugs)} drugs collected...", end="\r")
        if len(mols) < limit:
            break
    print(f"\nTotal FDA drugs fetched: {len(drugs)}")
    return drugs


def fetch_activities_for_drug(chembl_id):
    """Return positive + negative activity records for one drug."""
    records = []
    for label, op, cutoff in [
        (1, "gte", PCHEMBL_POS),
        (0, "lte", PCHEMBL_NEG),
    ]:
        data = chembl_get("activity", {
            "molecule_chembl_id": chembl_id,
            f"pchembl_value__{op}": cutoff,
            "target_type":    "SINGLE PROTEIN",
            "assay_type":     "B",          # binding assay
            "limit":          50,
            "format":         "json",
        })
        if not data:
            continue
        for act in data.get("activities", []):
            pv  = act.get("pchembl_value")
            tid = act.get("target_chembl_id")
            uid = act.get("target_accession")   # UniProt ID if available
            if pv is None or not tid:
                continue
            records.append({
                "drug_chembl_id":   chembl_id,
                "target_chembl_id": tid,
                "uniprot_id":       uid,
                "pchembl_value":    float(pv),
                "label":            label,
            })
    return records


def fetch_protein_sequence(uniprot_id):
    """Return amino-acid sequence from UniProt, or None."""
    url = f"{UNIPROT_API}/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            return "".join(lines[1:])   # skip FASTA header
    except Exception:
        pass
    return None


def resolve_uniprot_from_target(target_chembl_id):
    """Look up UniProt accession from ChEMBL target."""
    data = chembl_get("target", {"target_chembl_id": target_chembl_id, "limit": 1})
    if not data:
        return None
    targets = data.get("targets", [])
    if not targets:
        return None
    comps = targets[0].get("target_components", [])
    for c in comps:
        for xref in c.get("target_component_xrefs", []):
            if xref.get("xref_src_db") == "UniProt":
                return xref.get("xref_id")
    return None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # 1. load training SMILES to exclude
    train_smiles = set()
    for csv_path in [
        "../datasets/bindingdb/overlap_with_mask/train.csv",
        "../datasets/bindingdb/overlap_with_mask/val.csv",
        "../datasets/bindingdb/overlap_with_mask/test.csv",
        "../datasets/overlap_with_mask/train.csv",
    ]:
        p = OUT_DIR.parent / csv_path.lstrip("../")
        if p.exists():
            df = pd.read_csv(p, usecols=["SMILES"])
            train_smiles.update(df["SMILES"].tolist())
    print(f"Training SMILES to exclude: {len(train_smiles)}")

    # 2. fetch FDA drugs
    drugs = fetch_fda_drugs()
    drug_map = {d["chembl_id"]: d for d in drugs}

    # 3. fetch activities
    all_records = []
    uniprot_cache = {}    # target_chembl_id → uniprot_id
    seq_cache     = {}    # uniprot_id → sequence

    for i, drug in enumerate(drugs):
        cid = drug["chembl_id"]
        if drug["smiles"] in train_smiles:
            continue                         # skip training drugs
        acts = fetch_activities_for_drug(cid)
        all_records.extend(acts)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(drugs)} drugs processed, {len(all_records)} records so far")
        if len(all_records) >= MAX_PAIRS * 2:
            break
        time.sleep(0.1)

    print(f"\nRaw activity records: {len(all_records)}")

    # 4. resolve UniProt IDs and fetch sequences
    rows = []
    for rec in all_records:
        uid = rec.get("uniprot_id")
        tid = rec["target_chembl_id"]

        if not uid:
            if tid not in uniprot_cache:
                uniprot_cache[tid] = resolve_uniprot_from_target(tid)
                time.sleep(0.05)
            uid = uniprot_cache[tid]

        if not uid:
            continue

        if uid not in seq_cache:
            seq_cache[uid] = fetch_protein_sequence(uid)
            time.sleep(0.05)
        seq = seq_cache[uid]

        if not seq or len(seq) < 50:
            continue

        drug_info = drug_map.get(rec["drug_chembl_id"], {})
        rows.append({
            "drug_name":        drug_info.get("name", rec["drug_chembl_id"]),
            "drug_chembl_id":   rec["drug_chembl_id"],
            "smiles":           drug_info.get("smiles", ""),
            "uniprot_id":       uid,
            "target_chembl_id": tid,
            "protein_seq":      seq,
            "pchembl_value":    rec["pchembl_value"],
            "label":            rec["label"],
        })
        if len(rows) >= MAX_PAIRS:
            break

    df = pd.DataFrame(rows).drop_duplicates(subset=["smiles", "protein_seq", "label"])
    df.to_csv(RAW_CSV, index=False)

    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    print(f"\nSaved {len(df)} pairs to {RAW_CSV}")
    print(f"  Positive (BIND)    : {pos}")
    print(f"  Negative (NO BIND) : {neg}")
    print(f"  Unique drugs       : {df['drug_chembl_id'].nunique()}")
    print(f"  Unique targets     : {df['uniprot_id'].nunique()}")


if __name__ == "__main__":
    main()
