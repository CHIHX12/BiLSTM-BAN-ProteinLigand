#!/usr/bin/env python3
"""Join the ChEMBL PCI + SMILES + proteome files into a clean drug-protein pairs
CSV for TEIBAN, applying the SAME preprocessing as the prediction tool:
de-salt / de-solvent, exclude abnormal molecules, and de-duplicate pairs.

Streaming and memory-conscious (each unique compound is standardised once).

Usage: python prepare_chembl_pairs.py <dataset_dir> <output.csv>
"""
import os
import sys
import csv
import hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from predict_simple import standardize_smiles, clean_protein  # reuse tool preprocessing

DS, OUT = sys.argv[1], sys.argv[2]
PCI = f"{DS}/ChEMBL_hsa_PCI.txt"
SMI = f"{DS}/SMILES.txt"
PROT = f"{DS}/hsa.proteome"


def log(msg):
    print(msg, flush=True)


# 1) proteome: name -> cleaned sequence (None if unusable)
log("[1/4] loading + cleaning proteome ...")
prot = {}
name, seq = None, []
for ln in open(PROT, encoding="utf-8", errors="replace"):
    ln = ln.rstrip("\n")
    if ln.startswith(">"):
        if name:
            prot[name] = "".join(seq)
        name, seq = ln[1:].strip(), []
    elif ln.strip():
        seq.append(ln.strip())
if name:
    prot[name] = "".join(seq)
prot_clean = {}
for nm, s in prot.items():
    cs, _ = clean_protein(s)
    if cs:
        prot_clean[nm] = cs
log(f"      proteins usable: {len(prot_clean)}/{len(prot)}")

# 2) compound ids that actually appear in PCI (only standardise those)
log("[2/4] collecting compound ids from PCI ...")
pci_ids = set()
for ln in open(PCI, encoding="utf-8", errors="replace"):
    f = ln.rstrip("\n").split("\t")
    if len(f) >= 7:
        pci_ids.add(f[0])
log(f"      unique compounds in PCI: {len(pci_ids)}")

# 3) standardise each needed compound once: id -> clean SMILES (or None)
log("[3/4] standardising SMILES (de-salt / exclude abnormal) ...")
smi_clean = {}
done = 0
for ln in open(SMI, encoding="utf-8", errors="replace"):
    f = ln.rstrip("\n").split("\t")
    if len(f) < 2:
        continue
    cid = f[0].replace("ChEMBL:", "")
    if cid in pci_ids and cid not in smi_clean:
        smi_clean[cid] = standardize_smiles(f[1])[0]
        done += 1
        if done % 100000 == 0:
            log(f"      standardised {done} ...")
log(f"      standardised {len(smi_clean)} compounds")

# 4) stream PCI, join, de-duplicate (clean_smiles, protein), write
log("[4/4] joining + de-duplicating + writing ...")
seen = set()
kept = bad_mol = bad_prot = dup = 0
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "SMILES", "Protein"])
    for ln in open(PCI, encoding="utf-8", errors="replace"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 7:
            continue
        cid, pname = f[0], f[6]
        cs = smi_clean.get(cid)
        if not cs:
            bad_mol += 1
            continue
        seq = prot_clean.get(pname)
        if not seq:
            bad_prot += 1
            continue
        key = hashlib.md5(f"{cs}|{pname}".encode()).digest()
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        w.writerow([f"{cid}~{pname}", cs, seq])
        kept += 1

log("=" * 60)
log(f"DONE. wrote {kept} clean unique pairs -> {OUT}")
log(f"  skipped bad/abnormal molecules: {bad_mol}")
log(f"  skipped missing/invalid proteins: {bad_prot}")
log(f"  removed duplicate pairs: {dup}")
