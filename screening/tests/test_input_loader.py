"""
Tests for preprocess/input_loader.py
Covers: TSV, single-column, FASTA, comments, blank lines, duplicates.
"""
import sys, os, tempfile, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from preprocess.input_loader import InputLoader


@pytest.fixture
def loader():
    return InputLoader()


def _tmpfile(content: str) -> str:
    """Write content to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                    delete=False, encoding="utf-8")
    f.write(textwrap.dedent(content))
    f.flush()
    f.close()
    return f.name


# ─── Ligand loading ────────────────────────────────────────────────────────

class TestLoadLigands:
    def test_two_column_tsv(self, loader):
        path = _tmpfile("""\
            Aspirin\tCC(=O)Oc1ccccc1C(=O)O
            Ibuprofen\tCC(C)Cc1ccc(cc1)CC(=O)O
        """)
        entries = loader.load_ligands(path)
        assert len(entries) == 2
        assert entries[0] == ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O")
        assert entries[1][0] == "Ibuprofen"
        os.unlink(path)

    def test_single_column_auto_named(self, loader):
        path = _tmpfile("""\
            CC(=O)Oc1ccccc1C(=O)O
            CC(C)Cc1ccc(cc1)CC(=O)O
        """)
        entries = loader.load_ligands(path)
        assert len(entries) == 2
        assert entries[0][0] == "LIG_0001"
        assert entries[1][0] == "LIG_0002"
        os.unlink(path)

    def test_comments_skipped(self, loader):
        path = _tmpfile("""\
            # This is a comment
            Aspirin\tCC(=O)Oc1ccccc1C(=O)O
            # Another comment
        """)
        entries = loader.load_ligands(path)
        assert len(entries) == 1
        os.unlink(path)

    def test_blank_lines_skipped(self, loader):
        path = _tmpfile("""\
            Aspirin\tCC(=O)Oc1ccccc1C(=O)O

            Ibuprofen\tCC(C)Cc1ccc(cc1)CC(=O)O

        """)
        entries = loader.load_ligands(path)
        assert len(entries) == 2
        os.unlink(path)

    def test_whitespace_stripped(self, loader):
        path = _tmpfile("  Aspirin  \t  CC(=O)Oc1ccccc1C(=O)O  \n")
        entries = loader.load_ligands(path)
        assert entries[0][0] == "Aspirin"
        assert entries[0][1] == "CC(=O)Oc1ccccc1C(=O)O"
        os.unlink(path)

    def test_duplicate_names_made_unique(self, loader):
        path = _tmpfile("""\
            Drug\tCC(=O)O
            Drug\tCC(C)O
            Drug\tCCCO
        """)
        entries = loader.load_ligands(path)
        names = [e[0] for e in entries]
        assert len(set(names)) == 3   # all unique
        assert "Drug" in names
        assert any("Drug_2" in n or "Drug_3" in n for n in names)
        os.unlink(path)

    def test_empty_file(self, loader):
        path = _tmpfile("# only comments\n\n")
        entries = loader.load_ligands(path)
        assert entries == []
        os.unlink(path)

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_ligands("/nonexistent/path/ligands.txt")

    def test_extra_columns_ignored(self, loader):
        """Files with >2 columns: name=col0, value=col1, rest ignored."""
        path = _tmpfile("Aspirin\tCC(=O)Oc1ccccc1C(=O)O\textra_column\n")
        entries = loader.load_ligands(path)
        assert entries[0] == ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O")
        os.unlink(path)

    def test_utf8_bom_handled(self, loader):
        """Files saved with UTF-8 BOM (common in Windows Excel exports)."""
        path = _tmpfile("\ufeffAspirin\tCC(=O)O\n")
        entries = loader.load_ligands(path)
        assert entries[0][0] == "Aspirin"   # BOM stripped
        os.unlink(path)


# ─── Protein loading ───────────────────────────────────────────────────────

class TestLoadProteins:
    def test_two_column_tsv(self, loader):
        path = _tmpfile("EGFR\tMKTAYIAKQR\nCDK2\tMENFQKVEK\n")
        entries = loader.load_proteins(path)
        assert len(entries) == 2
        assert entries[0] == ("EGFR", "MKTAYIAKQR")
        os.unlink(path)

    def test_single_fasta_block(self, loader):
        content = ">sp|P00533|EGFR_HUMAN Test protein\nMKTAYIAKQR\nQISFVK\n"
        path = _tmpfile(content)
        entries = loader.load_proteins(path)
        assert len(entries) == 1
        name, raw = entries[0]
        assert name == "sp|P00533|EGFR_HUMAN"   # first word of header
        assert "MKTAYIAKQR" in raw
        os.unlink(path)

    def test_multiple_fasta_blocks(self, loader):
        content = (
            ">PROTEIN_A\nACDEFGHIKLM\n"
            ">PROTEIN_B\nNPQRSTVWY\n"
        )
        path = _tmpfile(content)
        entries = loader.load_proteins(path)
        assert len(entries) == 2
        assert entries[0][0] == "PROTEIN_A"
        assert entries[1][0] == "PROTEIN_B"
        os.unlink(path)

    def test_mixed_tsv_and_fasta(self, loader):
        content = (
            "PLAIN\tMKTAYIAKQR\n"
            ">FASTA_PROT\nACDEFGHIKLM\n"
        )
        path = _tmpfile(content)
        entries = loader.load_proteins(path)
        assert len(entries) == 2
        names = {e[0] for e in entries}
        assert "PLAIN" in names
        assert "FASTA_PROT" in names
        os.unlink(path)

    def test_fasta_multiline_seq_concatenated(self, loader):
        content = ">TEST\nACDEF\nGHIKL\nMNPQR\n"
        path = _tmpfile(content)
        entries = loader.load_proteins(path)
        name, raw = entries[0]
        assert "ACDEF" in raw and "GHIKL" in raw and "MNPQR" in raw
        os.unlink(path)
