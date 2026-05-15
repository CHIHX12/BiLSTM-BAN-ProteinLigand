#!/usr/bin/env python3
"""
Input Validation Tool
======================
Validates ligand SMILES and protein sequences BEFORE running the screening.
Run this script first to catch format errors and get a clean input report.

Usage:
    python validate_inputs.py [options]

Options:
    --ligands   Path to ligands file  (default: input/ligands/ligands.txt)
    --proteins  Path to proteins file (default: input/proteins/proteins.txt)
    --multi-chain  How to handle multi-chain FASTA: first|concat|error
                   (default: first)
    --report    Path to save validation report (default: validation_report.txt)
    --strict    Exit with error code if any WARNING is found (default: only on ERROR)

Exit codes:
    0 = All inputs valid (or only warnings with --strict=False)
    1 = At least one ERROR found
    2 = File not found
    3 = Missing dependency (e.g. RDKit)
"""

import argparse
import os
import sys
from datetime import datetime

# Ensure parent project (GCN-BILSTM-BAN-bak) is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from preprocess import InputLoader, SMILESValidator, ProteinValidator
from preprocess.smiles_validator import format_validation_report as fmt_smiles
from preprocess.protein_validator import format_validation_report as fmt_protein


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate ligand SMILES and protein sequences before screening."
    )
    p.add_argument("--ligands",
                   default=os.path.join(_HERE, "input", "ligands", "ligands.txt"),
                   help="Path to ligands.txt")
    p.add_argument("--proteins",
                   default=os.path.join(_HERE, "input", "proteins", "proteins.txt"),
                   help="Path to proteins.txt")
    p.add_argument("--multi-chain",
                   choices=["first", "concat", "error"],
                   default="first",
                   help="Multi-chain FASTA handling (default: first)")
    p.add_argument("--report",
                   default=os.path.join(_HERE, "validation_report.txt"),
                   help="Path to save the validation report")
    p.add_argument("--strict",
                   action="store_true",
                   help="Exit with code 1 if any WARNING is found")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    loader = InputLoader()
    start_time = datetime.now()

    print("=" * 70)
    print("  HTS Drug-Protein Screening — Input Validation")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Load files                                                          #
    # ------------------------------------------------------------------ #
    try:
        print(f"\n[1/4] Loading ligands from: {args.ligands}")
        ligand_entries = loader.load_ligands(args.ligands)
        print(f"      Loaded {len(ligand_entries)} entries.")

        print(f"\n[2/4] Loading proteins from: {args.proteins}")
        protein_entries = loader.load_proteins(args.proteins)
        print(f"      Loaded {len(protein_entries)} entries.")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return 2

    if not ligand_entries:
        print("\n[ERROR] Ligands file is empty or contains only comments.")
        return 1

    if not protein_entries:
        print("\n[ERROR] Proteins file is empty or contains only comments.")
        return 1

    # ------------------------------------------------------------------ #
    # Validate SMILES                                                     #
    # ------------------------------------------------------------------ #
    print(f"\n[3/4] Validating {len(ligand_entries)} SMILES strings...")
    try:
        smiles_validator = SMILESValidator()
    except ImportError as e:
        print(f"\n[ERROR] {e}")
        return 3

    smiles_records = smiles_validator.validate(ligand_entries)

    smiles_errors   = [r for r in smiles_records if r.status.value == "ERROR"]
    smiles_warnings = [r for r in smiles_records if r.status.value == "WARNING"]
    smiles_ok       = [r for r in smiles_records if r.status.value == "OK"]

    print(f"      OK: {len(smiles_ok)}, Warnings: {len(smiles_warnings)}, Errors: {len(smiles_errors)}")

    # ------------------------------------------------------------------ #
    # Validate proteins                                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[4/4] Validating {len(protein_entries)} protein sequences...")
    protein_validator = ProteinValidator(multi_chain_mode=args.multi_chain)
    protein_records = protein_validator.validate(protein_entries)

    prot_errors   = [r for r in protein_records if r.status.value == "ERROR"]
    prot_warnings = [r for r in protein_records if r.status.value == "WARNING"]
    prot_ok       = [r for r in protein_records if r.status.value == "OK"]

    print(f"      OK: {len(prot_ok)}, Warnings: {len(prot_warnings)}, Errors: {len(prot_errors)}")

    # ------------------------------------------------------------------ #
    # Build and print report                                              #
    # ------------------------------------------------------------------ #
    smiles_report  = fmt_smiles(smiles_records)
    protein_report = fmt_protein(protein_records)

    usable_ligands  = [r for r in smiles_records  if r.is_usable]
    usable_proteins = [r for r in protein_records if r.is_usable]

    total_pairs = len(usable_ligands) * len(usable_proteins)

    summary = [
        "\n" + "=" * 70,
        "  VALIDATION SUMMARY",
        "=" * 70,
        f"  Ligands:  {len(smiles_records):>5} total  |  {len(usable_ligands):>5} usable  |  {len(smiles_errors):>3} error(s)  |  {len(smiles_warnings):>3} warning(s)",
        f"  Proteins: {len(protein_records):>5} total  |  {len(usable_proteins):>5} usable  |  {len(prot_errors):>3} error(s)  |  {len(prot_warnings):>3} warning(s)",
        f"  Screening pairs: {total_pairs:,}  ({len(usable_ligands)} × {len(usable_proteins)})",
        "=" * 70,
    ]

    if smiles_errors or prot_errors:
        summary.append("  ✗  Errors found — fix the above issues before running screen.py")
    elif smiles_warnings or prot_warnings:
        summary.append("  ⚠  Warnings found — entries with warnings will still be used.")
        summary.append("     Review the report and correct if needed.")
    else:
        summary.append("  ✓  All inputs are clean. Ready to run screen.py")

    summary.append("")

    full_report = "\n".join([
        f"HTS Validation Report — {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Ligands file:  {args.ligands}",
        f"Proteins file: {args.proteins}",
        "",
        smiles_report,
        "",
        protein_report,
        "\n".join(summary),
    ])

    # Print and save
    print(smiles_report)
    print(protein_report)
    print("\n".join(summary))

    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(full_report)
    print(f"\n  Report saved to: {args.report}")

    # ------------------------------------------------------------------ #
    # Exit code                                                           #
    # ------------------------------------------------------------------ #
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  Elapsed: {elapsed:.1f}s\n")

    has_errors   = bool(smiles_errors or prot_errors)
    has_warnings = bool(smiles_warnings or prot_warnings)

    if has_errors:
        return 1
    if args.strict and has_warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
