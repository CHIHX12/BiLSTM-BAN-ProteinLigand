"""
Input File Loader
==================
Reads ligand and protein .txt input files for the HTS screening pipeline.

Supported formats (auto-detected):
  1. Two-column TSV:  NAME<TAB>SMILES_OR_SEQUENCE
  2. One-column:      SMILES_OR_SEQUENCE  (auto-assigned names: LIG_001, PRO_001 ...)

Rules:
  - Lines starting with '#' are comments and are skipped
  - Blank lines are skipped
  - Trailing/leading whitespace is stripped from both name and value
  - Duplicate names get a numeric suffix to stay unique (_2, _3 ...)
"""

from __future__ import annotations

import os
from typing import List, Tuple


class InputLoader:
    """Load (name, value) pairs from a plain-text file."""

    def load_ligands(self, filepath: str) -> List[Tuple[str, str]]:
        """Load (name, SMILES) pairs from ligands file."""
        return self._load_file(filepath, name_prefix="LIG")

    def load_proteins(self, filepath: str) -> List[Tuple[str, str]]:
        """Load (name, sequence) pairs from proteins file.

        Supports FASTA blocks: if a line starts with '>' it is treated as a
        FASTA header. The entire block (header + sequence lines) is collected
        and passed as the raw value to the validator.
        """
        return self._load_fasta_aware(filepath, name_prefix="PRO")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_file(self, filepath: str, name_prefix: str) -> List[Tuple[str, str]]:
        """Generic loader: two-column TSV or single-column."""
        self._check_exists(filepath)
        entries: List[Tuple[str, str]] = []
        auto_counter = 1
        seen_names: dict[str, int] = {}

        with open(filepath, "r", encoding="utf-8-sig") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.rstrip("\n")
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    continue   # Skip blank lines and comments

                parts = stripped.split("\t")

                if len(parts) >= 2:
                    # Two-column: NAME<TAB>VALUE  (extra columns ignored)
                    name = parts[0].strip()
                    value = parts[1].strip()
                elif len(parts) == 1:
                    # Single-column: no name provided
                    name = f"{name_prefix}_{auto_counter:04d}"
                    value = parts[0].strip()
                    auto_counter += 1
                else:
                    continue   # Should not happen

                if not name:
                    name = f"{name_prefix}_{auto_counter:04d}"
                    auto_counter += 1

                # Deduplicate names
                name = self._unique_name(name, seen_names)
                entries.append((name, value))

        return entries

    def _load_fasta_aware(self, filepath: str, name_prefix: str) -> List[Tuple[str, str]]:
        """
        Load proteins, aware of FASTA blocks.

        If the file is purely TSV (no '>' headers), falls back to _load_file.
        If FASTA blocks are present, each block becomes one entry.
        Mixed files (some FASTA, some plain) are handled gracefully.
        """
        self._check_exists(filepath)

        with open(filepath, "r", encoding="utf-8-sig") as fh:
            raw_content = fh.read()

        lines = raw_content.splitlines()
        has_any_fasta = any(ln.strip().startswith(">") for ln in lines)

        if not has_any_fasta:
            # No FASTA blocks - use simple TSV loader
            return self._load_file(filepath, name_prefix=name_prefix)

        # Parse FASTA-aware: mix of plain-TSV lines and FASTA blocks
        entries: List[Tuple[str, str]] = []
        auto_counter = 1
        seen_names: dict[str, int] = {}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line or line.startswith("#"):
                i += 1
                continue

            if line.startswith(">"):
                # --- FASTA block ---
                fasta_lines = [line]   # Include the header
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line or next_line.startswith("#"):
                        # Blank / comment: check if next non-blank is another FASTA header
                        # Peek ahead
                        peek = i + 1
                        while peek < len(lines) and not lines[peek].strip():
                            peek += 1
                        if peek < len(lines) and lines[peek].strip().startswith(">"):
                            break  # Next block starts - end this one
                        # Otherwise blank line within block - include and continue
                        fasta_lines.append(lines[i])
                        i += 1
                    elif next_line.startswith(">"):
                        break  # New block
                    else:
                        fasta_lines.append(lines[i])
                        i += 1

                # Extract name from FASTA header
                header = fasta_lines[0][1:].strip()   # Remove '>'
                name = header.split()[0] if header else f"{name_prefix}_{auto_counter:04d}"
                raw_block = "\n".join(fasta_lines)
                auto_counter += 1

                name = self._unique_name(name, seen_names)
                entries.append((name, raw_block))

            else:
                # --- Plain TSV line ---
                parts = line.split("\t")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    value = parts[1].strip()
                else:
                    name = f"{name_prefix}_{auto_counter:04d}"
                    value = parts[0].strip()
                    auto_counter += 1

                if not name:
                    name = f"{name_prefix}_{auto_counter:04d}"
                    auto_counter += 1

                name = self._unique_name(name, seen_names)
                entries.append((name, value))
                i += 1

        return entries

    @staticmethod
    def _check_exists(filepath: str) -> None:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Input file not found: '{filepath}'\n"
                "Please place your input file at the expected path and try again."
            )

    @staticmethod
    def _unique_name(name: str, seen: dict[str, int]) -> str:
        """Make name unique by appending _2, _3 ... if already seen."""
        if name not in seen:
            seen[name] = 1
            return name
        seen[name] += 1
        return f"{name}_{seen[name]}"
