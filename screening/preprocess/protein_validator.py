"""
Protein Sequence Validator and Cleaner
=======================================
Validates and preprocesses protein sequences for the HTS screening pipeline.

Handles common issues:
  1. FASTA format: lines starting with '>' (header) are parsed correctly
  2. Multi-chain FASTA: multiple '>' blocks → configurable behaviour
     (default: take first chain, warn about others)
  3. Non-standard amino acids: B, J, O, U, Z → warn and replace with X
  4. Lowercase letters → convert to uppercase
  5. Numbers, spaces, dashes, asterisks, newlines → strip silently
  6. Very short sequences (< MIN_LENGTH) → error
  7. Very long sequences (> MAX_PROTEIN_LENGTH) → warning, will be truncated
  8. High fraction of unknown residues (> MAX_X_FRACTION) → warning
  9. Duplicate sequences → warning
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# Standard single-letter amino acid codes accepted by the model
STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

# Non-standard codes and their meaning
NONSTANDARD_AA_MAP = {
    "B": "D/N (Asx)",
    "J": "I/L (Xle)",
    "O": "Pyl (pyrrolysine)",
    "U": "Sec (selenocysteine)",
    "Z": "E/Q (Glx)",
}

# Residues recognized by utils.CHARPROTSET (includes X as 'unknown')
MODEL_VALID_AAS = STANDARD_AAS | {"X"}

MIN_LENGTH = 10          # Shorter sequences are likely invalid
MAX_PROTEIN_LENGTH = 1200  # Model training limit; longer seqs will be truncated
MAX_X_FRACTION = 0.30    # Warn if > 30% of residues are unknown


class ValidationStatus(str, Enum):
    OK      = "OK"
    WARNING = "WARNING"
    ERROR   = "ERROR"


@dataclass
class ProteinRecord:
    """One validated protein entry."""
    original_name: str
    original_sequence: str               # Raw input (may include FASTA headers etc.)
    clean_sequence: Optional[str]        # Uppercase, standard chars only
    status: ValidationStatus
    warnings: List[str] = field(default_factory=list)
    errors: List[str]   = field(default_factory=list)
    length: int = 0
    x_fraction: float = 0.0             # Fraction of unknown (X) residues

    @property
    def is_usable(self) -> bool:
        return self.status != ValidationStatus.ERROR


class ProteinValidator:
    """
    Validate and clean protein sequences.

    Usage:
        validator = ProteinValidator()
        records = validator.validate([("EGFR", "MRPSGTAGAA..."), ...])
    """

    def __init__(
        self,
        min_length: int = MIN_LENGTH,
        max_length: int = MAX_PROTEIN_LENGTH,
        max_x_fraction: float = MAX_X_FRACTION,
        multi_chain_mode: str = "first",   # "first" | "concat" | "error"
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_x_fraction = max_x_fraction
        self.multi_chain_mode = multi_chain_mode

    def validate(self, entries: List[Tuple[str, str]]) -> List[ProteinRecord]:
        """
        Validate a list of (name, raw_sequence) tuples.

        raw_sequence may be a FASTA block (with header lines) or a plain
        amino-acid string.

        Returns a list of ProteinRecord, one per input entry.
        """
        records: List[ProteinRecord] = []
        seen_sequences: dict[str, str] = {}   # clean_sequence → first name

        for name, raw in entries:
            record = self._validate_one(name, raw)
            records.append(record)

            # Duplicate detection
            if record.clean_sequence:
                if record.clean_sequence in seen_sequences:
                    first_name = seen_sequences[record.clean_sequence]
                    record.warnings.append(
                        f"Duplicate of '{first_name}' (identical sequence). "
                        "Will still be used but results will be identical."
                    )
                    if record.status == ValidationStatus.OK:
                        record.status = ValidationStatus.WARNING
                else:
                    seen_sequences[record.clean_sequence] = name

        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_one(self, name: str, raw: str) -> ProteinRecord:
        record = ProteinRecord(
            original_name=name,
            original_sequence=raw,
            clean_sequence=None,
            status=ValidationStatus.OK,
        )

        # --- Step 1: Detect and parse FASTA format ---
        seq = self._parse_fasta(raw, record)

        # --- Step 2: Remove non-amino-acid characters ---
        seq = self._strip_non_aa_chars(seq, record)

        if not seq:
            record.errors.append(
                "Sequence is empty after stripping invalid characters. "
                "Please provide a valid amino acid sequence."
            )
            record.status = ValidationStatus.ERROR
            return record

        # --- Step 3: Uppercase ---
        seq = seq.upper()

        # --- Step 4: Replace non-standard AAs ---
        seq = self._normalize_nonstandard(seq, record)

        # --- Step 5: Remove any remaining non-model characters ---
        invalid_chars = set(seq) - MODEL_VALID_AAS
        if invalid_chars:
            record.errors.append(
                f"Unrecognized amino acid characters after cleaning: "
                f"{sorted(invalid_chars)}. Cannot process this sequence."
            )
            record.status = ValidationStatus.ERROR
            return record

        # --- Step 6: Length checks ---
        length = len(seq)

        if length < self.min_length:
            record.errors.append(
                f"Sequence too short: {length} residues "
                f"(minimum required: {self.min_length}). "
                "This may not be a valid protein sequence."
            )
            record.status = ValidationStatus.ERROR
            return record

        if length > self.max_length:
            record.warnings.append(
                f"Sequence length {length} exceeds model limit of {self.max_length}. "
                f"The sequence will be TRUNCATED to first {self.max_length} residues "
                "during prediction. Residues beyond this limit will be ignored. "
                "Consider using the full-length canonical isoform if possible."
            )
            if record.status == ValidationStatus.OK:
                record.status = ValidationStatus.WARNING

        # --- Step 7: X fraction check ---
        x_count = seq.count("X")
        x_fraction = x_count / length if length > 0 else 0.0

        if x_fraction > self.max_x_fraction:
            record.warnings.append(
                f"High fraction of unknown residues (X): {x_fraction:.1%} "
                f"({x_count}/{length}). Prediction quality may be reduced. "
                "Check if the sequence was properly retrieved from UniProt."
            )
            if record.status == ValidationStatus.OK:
                record.status = ValidationStatus.WARNING

        # --- Finalize ---
        record.clean_sequence = seq
        record.length = length
        record.x_fraction = round(x_fraction, 4)

        return record

    def _parse_fasta(self, raw: str, record: ProteinRecord) -> str:
        """
        Detect FASTA format and extract sequence(s).
        Handles single-chain and multi-chain FASTA.
        """
        lines = raw.strip().splitlines()
        has_header = any(line.startswith(">") for line in lines)

        if not has_header:
            # Plain sequence or needs whitespace stripping only
            return raw

        # Parse FASTA blocks
        chains: List[Tuple[str, str]] = []   # [(header, sequence), ...]
        current_header = ""
        current_seq_parts: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq_parts:
                    chains.append((current_header, "".join(current_seq_parts)))
                current_header = line[1:].strip()  # Remove '>' prefix
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        # Flush last chain
        if current_seq_parts:
            chains.append((current_header, "".join(current_seq_parts)))

        if not chains:
            return raw   # Fallback - downstream will handle empty case

        if len(chains) == 1:
            # Single chain - silent FASTA detection
            record.warnings.append(
                "FASTA format detected. Header line removed and sequence extracted."
            )
            if record.status == ValidationStatus.OK:
                record.status = ValidationStatus.WARNING
            return chains[0][1]

        # Multi-chain FASTA
        chain_summary = ", ".join(
            f"Chain '{h[:30]}' ({len(s)} aa)" for h, s in chains
        )
        record.warnings.append(
            f"Multi-chain FASTA detected ({len(chains)} chains): {chain_summary}."
        )
        if record.status == ValidationStatus.OK:
            record.status = ValidationStatus.WARNING

        if self.multi_chain_mode == "first":
            record.warnings.append(
                f"Using first chain only: '{chains[0][0][:40]}'. "
                "Other chains are discarded. Use --multi-chain=concat to concatenate, "
                "or split the file into separate entries."
            )
            return chains[0][1]

        elif self.multi_chain_mode == "concat":
            record.warnings.append(
                "Concatenating all chains into a single sequence. "
                "This may affect prediction accuracy for multi-chain complexes."
            )
            return "".join(seq for _, seq in chains)

        else:  # "error"
            record.errors.append(
                f"Multi-chain FASTA is not allowed in 'error' mode. "
                f"Found {len(chains)} chains. Please split into separate entries."
            )
            record.status = ValidationStatus.ERROR
            return ""

    def _strip_non_aa_chars(self, seq: str, record: ProteinRecord) -> str:
        """
        Remove characters that are definitely not amino acids:
        digits, whitespace, hyphens, asterisks (*=stop codon), slashes.
        """
        # Characters to silently remove
        stripped = re.sub(r"[\d\s\-\*\/\\|_]", "", seq)

        removed_chars = set(seq) - set(stripped) - {" ", "\n", "\r", "\t"}
        non_whitespace_removed = {c for c in removed_chars if not c.isspace()}

        if non_whitespace_removed:
            record.warnings.append(
                f"Removed non-amino-acid characters: "
                f"{sorted(non_whitespace_removed)}. This is common for sequences "
                "copied from databases that include line numbers or gap characters."
            )
            if record.status == ValidationStatus.OK:
                record.status = ValidationStatus.WARNING

        return stripped

    def _normalize_nonstandard(self, seq: str, record: ProteinRecord) -> str:
        """Replace non-standard amino acid codes with X and warn."""
        found_nonstandard: dict[str, int] = {}   # code → count

        for aa in NONSTANDARD_AA_MAP:
            count = seq.count(aa)
            if count > 0:
                found_nonstandard[aa] = count

        if found_nonstandard:
            details = ", ".join(
                f"{aa}×{cnt} [{NONSTANDARD_AA_MAP[aa]}]"
                for aa, cnt in sorted(found_nonstandard.items())
            )
            record.warnings.append(
                f"Non-standard amino acids replaced with X: {details}. "
                "These are uncommon residues not in the model's vocabulary."
            )
            if record.status == ValidationStatus.OK:
                record.status = ValidationStatus.WARNING

            for aa in NONSTANDARD_AA_MAP:
                seq = seq.replace(aa, "X")

        return seq


def format_validation_report(records: List[ProteinRecord]) -> str:
    """Generate a human-readable validation report for proteins."""
    ok_count     = sum(1 for r in records if r.status == ValidationStatus.OK)
    warn_count   = sum(1 for r in records if r.status == ValidationStatus.WARNING)
    error_count  = sum(1 for r in records if r.status == ValidationStatus.ERROR)
    usable_count = sum(1 for r in records if r.is_usable)

    lines = [
        "=" * 70,
        "PROTEIN SEQUENCE VALIDATION REPORT",
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
            if r.clean_sequence:
                seq_preview = r.clean_sequence[:60]
                suffix = "..." if len(r.clean_sequence) > 60 else ""
                lines.append(f"  →  Cleaned ({r.length} aa): {seq_preview}{suffix}")

    lines.append("=" * 70)
    return "\n".join(lines)
