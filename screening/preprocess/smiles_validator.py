"""
SMILES Validator and Cleaner
=============================
Validates and preprocesses SMILES strings for the HTS screening pipeline.

Handles common issues:
  1. Whitespace and Unicode/encoding artifacts (copy-paste from PDFs)
  2. Multi-fragment SMILES / salt forms (e.g. "CC.Cl") → take largest fragment
  3. RDKit parse failures → report exact error
  4. Oversized molecules (> MAX_DRUG_NODES atoms, model limit)
  5. Duplicate entries (same canonical SMILES)
  6. Empty / comment-only entries
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")   # suppress RDKit warnings to stderr
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Maximum number of heavy atoms the model supports (GCN padding limit)
MAX_DRUG_NODES = 290


class ValidationStatus(str, Enum):
    OK       = "OK"
    WARNING  = "WARNING"   # Cleaned but usable
    ERROR    = "ERROR"     # Cannot be used, skip


@dataclass
class SMILESRecord:
    """One validated ligand entry."""
    original_name: str
    original_smiles: str
    clean_smiles: Optional[str]          # Canonical SMILES after cleaning
    canonical_smiles: Optional[str]      # RDKit canonical form for dedup
    status: ValidationStatus
    warnings: List[str] = field(default_factory=list)
    errors: List[str]   = field(default_factory=list)
    heavy_atom_count: int = 0
    molecular_weight: float = 0.0

    @property
    def is_usable(self) -> bool:
        return self.status != ValidationStatus.ERROR


class SMILESValidator:
    """
    Validate and clean a list of (name, SMILES) pairs.

    Usage:
        validator = SMILESValidator(max_atoms=290)
        records = validator.validate([("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"), ...])
    """

    def __init__(self, max_atoms: int = MAX_DRUG_NODES):
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "RDKit is required for SMILES validation. "
                "Install with: conda install -c conda-forge rdkit"
            )
        self.max_atoms = max_atoms

    def validate(self, entries: List[Tuple[str, str]]) -> List[SMILESRecord]:
        """
        Validate a list of (name, smiles) tuples.

        Returns a list of SMILESRecord, one per input entry (including errors).
        """
        records: List[SMILESRecord] = []
        seen_canonical: dict[str, str] = {}   # canonical_smiles → first name

        for name, raw_smiles in entries:
            record = self._validate_one(name, raw_smiles)
            records.append(record)

            # Duplicate detection (only for valid entries)
            if record.canonical_smiles:
                if record.canonical_smiles in seen_canonical:
                    first_name = seen_canonical[record.canonical_smiles]
                    record.warnings.append(
                        f"Duplicate of '{first_name}' (same canonical SMILES). "
                        "Will still be used but results will be identical."
                    )
                    if record.status == ValidationStatus.OK:
                        record.status = ValidationStatus.WARNING
                else:
                    seen_canonical[record.canonical_smiles] = name

        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_one(self, name: str, raw: str) -> SMILESRecord:
        record = SMILESRecord(
            original_name=name,
            original_smiles=raw,
            clean_smiles=None,
            canonical_smiles=None,
            status=ValidationStatus.OK,
        )

        # --- Step 1: Basic string cleaning ---
        smiles = self._clean_string(raw, record)
        if not smiles:
            record.errors.append("Empty SMILES string after cleaning.")
            record.status = ValidationStatus.ERROR
            return record

        # --- Step 2: Salt handling (multi-fragment) ---
        smiles, had_salt = self._handle_salt(smiles, record)

        # --- Step 3: RDKit parse ---
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            record.errors.append(
                f"RDKit cannot parse SMILES: '{smiles}'. "
                "Check for invalid atom types, ring closure mismatches, or valence errors."
            )
            record.status = ValidationStatus.ERROR
            return record

        # --- Step 4: Canonicalize ---
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

        # --- Step 5: Atom count check ---
        n_atoms = mol.GetNumHeavyAtoms()
        if n_atoms > self.max_atoms:
            record.errors.append(
                f"Molecule has {n_atoms} heavy atoms, exceeding model limit of "
                f"{self.max_atoms}. Cannot use this molecule."
            )
            record.status = ValidationStatus.ERROR
            return record

        if n_atoms < 3:
            record.warnings.append(
                f"Very small molecule ({n_atoms} heavy atoms). May produce unreliable predictions."
            )
            if record.status == ValidationStatus.OK:
                record.status = ValidationStatus.WARNING

        # --- Step 6: Molecular weight check ---
        mw = Descriptors.MolWt(mol)
        if mw > 1000:
            record.warnings.append(
                f"High molecular weight ({mw:.1f} Da). "
                "Typically beyond drug-like range (Lipinski Rule of 5: MW ≤ 500)."
            )
            if record.status == ValidationStatus.OK:
                record.status = ValidationStatus.WARNING

        # --- Finalize ---
        record.clean_smiles = canonical
        record.canonical_smiles = canonical
        record.heavy_atom_count = n_atoms
        record.molecular_weight = round(mw, 2)

        return record

    def _clean_string(self, raw: str, record: SMILESRecord) -> str:
        """Remove whitespace and Unicode artifacts."""
        # Normalize Unicode (handles copy-paste from PDFs)
        try:
            normalized = unicodedata.normalize("NFKC", raw)
        except Exception:
            normalized = raw

        # Remove zero-width characters and non-printable chars
        cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff\u00a0]", "", normalized)
        cleaned = cleaned.strip()

        if cleaned != raw.strip():
            record.warnings.append(
                "Whitespace or Unicode artifacts removed from SMILES. "
                f"Original: '{raw[:60]}...'" if len(raw) > 60 else f"Original: '{raw}'"
            )
            if record.status == ValidationStatus.OK:
                record.status = ValidationStatus.WARNING

        return cleaned

    def _handle_salt(self, smiles: str, record: SMILESRecord) -> Tuple[str, bool]:
        """
        If SMILES contains '.' (multi-fragment / salt form),
        keep only the largest fragment by heavy atom count.
        """
        if "." not in smiles:
            return smiles, False

        fragments = smiles.split(".")
        valid_frags: List[Tuple[int, str]] = []

        for frag in fragments:
            mol = Chem.MolFromSmiles(frag)
            if mol is not None:
                valid_frags.append((mol.GetNumHeavyAtoms(), frag))

        if not valid_frags:
            # All fragments invalid - return original for downstream error handling
            return smiles, True

        valid_frags.sort(key=lambda x: x[0], reverse=True)
        largest_frag = valid_frags[0][1]
        frag_count = len(fragments)

        record.warnings.append(
            f"Multi-fragment SMILES detected ({frag_count} fragments, "
            f"e.g. salt form). Keeping largest fragment: '{largest_frag}'. "
            "Discarded: " + ", ".join(f[1] for f in valid_frags[1:])
        )
        if record.status == ValidationStatus.OK:
            record.status = ValidationStatus.WARNING

        return largest_frag, True


def format_validation_report(records: List[SMILESRecord]) -> str:
    """Generate a human-readable validation report for SMILES."""
    ok_count      = sum(1 for r in records if r.status == ValidationStatus.OK)
    warn_count    = sum(1 for r in records if r.status == ValidationStatus.WARNING)
    error_count   = sum(1 for r in records if r.status == ValidationStatus.ERROR)
    usable_count  = sum(1 for r in records if r.is_usable)

    lines = [
        "=" * 70,
        "SMILES VALIDATION REPORT",
        "=" * 70,
        f"  Total entries:    {len(records)}",
        f"  OK (clean):       {ok_count}",
        f"  WARNING (fixed):  {warn_count}",
        f"  ERROR (skipped):  {error_count}",
        f"  Usable for screen:{usable_count}",
        "-" * 70,
    ]

    for r in records:
        if r.status != ValidationStatus.OK:
            lines.append(f"\n[{r.status.value}] {r.original_name}")
            for w in r.warnings:
                lines.append(f"  ⚠  {w}")
            for e in r.errors:
                lines.append(f"  ✗  {e}")
            if r.clean_smiles:
                lines.append(f"  →  Cleaned SMILES: {r.clean_smiles}")

    lines.append("=" * 70)
    return "\n".join(lines)
