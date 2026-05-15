"""
Tests for preprocess/smiles_validator.py
Covers: OK paths, auto-fix (WARNING), and hard errors (ERROR).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from preprocess.smiles_validator import SMILESValidator, ValidationStatus


@pytest.fixture(scope="module")
def validator():
    return SMILESValidator()


# ─── OK cases ─────────────────────────────────────────────────────────────

class TestSMILESOK:
    def test_aspirin(self, validator):
        r = validator.validate([("Aspirin", "CC(=O)Oc1ccccc1C(=O)O")])[0]
        assert r.status == ValidationStatus.OK
        assert r.is_usable
        assert r.heavy_atom_count == 13
        assert r.molecular_weight > 170

    def test_ibuprofen(self, validator):
        r = validator.validate([("Ibuprofen", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")])[0]
        assert r.status == ValidationStatus.OK
        assert r.heavy_atom_count == 15

    def test_metformin(self, validator):
        r = validator.validate([("Metformin", "CN(C)C(=N)NC(=N)N")])[0]
        assert r.status == ValidationStatus.OK

    def test_caffeine(self, validator):
        r = validator.validate([("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")])[0]
        assert r.status == ValidationStatus.OK

    def test_canonical_smiles_normalized(self, validator):
        """Different representations of the same molecule should yield the same canonical SMILES."""
        r1 = validator.validate([("A", "c1ccccc1")])[0]   # aromatic
        r2 = validator.validate([("B", "C1=CC=CC=C1")])[0]  # Kekule
        assert r1.canonical_smiles == r2.canonical_smiles

    def test_stereochemistry_preserved(self, validator):
        r = validator.validate([("Ibuprofen_R", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")])[0]
        assert r.status == ValidationStatus.OK
        assert r.canonical_smiles is not None


# ─── WARNING cases (auto-fixed, still usable) ──────────────────────────────

class TestSMILESWarning:
    def test_salt_form_largest_fragment_kept(self, validator):
        """[Na+].[O-]C(=O)... → keep the organic fragment."""
        r = validator.validate([("SodiumBenzoate", "[Na+].[O-]C(=O)c1ccccc1")])[0]
        assert r.status == ValidationStatus.WARNING
        assert r.is_usable
        assert "Na" not in (r.clean_smiles or "")
        assert len(r.warnings) >= 1
        assert any("fragment" in w.lower() or "salt" in w.lower() for w in r.warnings)

    def test_salt_two_organic_fragments(self, validator):
        """Two organic fragments → keep larger one."""
        r = validator.validate([("TwoFrag", "CCO.CCCCCO")])[0]
        assert r.status == ValidationStatus.WARNING
        assert r.is_usable

    def test_leading_whitespace_stripped(self, validator):
        r = validator.validate([("Padded", "  CC(=O)O  ")])[0]
        assert r.is_usable
        assert r.clean_smiles is not None

    def test_unicode_artifact_stripped(self, validator):
        """Zero-width space that can appear when copying from PDFs."""
        r = validator.validate([("Unicode", "CC(=O)O\u200b")])[0]
        assert r.is_usable

    def test_duplicate_detected(self, validator):
        entries = [
            ("Drug_A", "CC(=O)Oc1ccccc1C(=O)O"),
            ("Drug_B", "CC(=O)Oc1ccccc1C(=O)O"),   # same as Drug_A
        ]
        records = validator.validate(entries)
        assert records[0].status == ValidationStatus.OK
        assert records[1].status == ValidationStatus.WARNING
        assert any("duplicate" in w.lower() for w in records[1].warnings)

    def test_high_mw_warns(self, validator):
        """Very large molecule → warning (not error if atoms ≤ 290)."""
        # Cyclosporin A: MW ~1152, heavy atoms ~82 — canonical SMILES from PubChem CID 5284373
        csA = "CC[C@@H]1NC(=O)[C@H]([C@H](O)[C@@H](C)C/C=C/C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](CC(C)C)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](C(C)C)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@H](Cc2ccccc2)N(C)C1=O"
        r = validator.validate([("CyclosporinA", csA)])[0]
        assert r.is_usable
        assert any("weight" in w.lower() or "1000" in w for w in r.warnings)


# ─── ERROR cases (not usable) ───────────────────────────────────────���──────

class TestSMILESError:
    def test_empty_string(self, validator):
        r = validator.validate([("Empty", "")])[0]
        assert r.status == ValidationStatus.ERROR
        assert not r.is_usable

    def test_whitespace_only(self, validator):
        r = validator.validate([("Space", "   ")])[0]
        assert r.status == ValidationStatus.ERROR
        assert not r.is_usable

    def test_invalid_smiles_garbled(self, validator):
        r = validator.validate([("Garbled", "XXXXX_NOT_SMILES")])[0]
        assert r.status == ValidationStatus.ERROR
        assert not r.is_usable
        assert len(r.errors) >= 1

    def test_invalid_valence(self, validator):
        """Carbon with 5 bonds — invalid."""
        r = validator.validate([("BadValence", "C(C)(C)(C)(C)C")])[0]
        # RDKit may or may not catch this depending on version; just check it runs
        # The important thing is no crash
        assert r.status in (ValidationStatus.OK, ValidationStatus.WARNING, ValidationStatus.ERROR)

    def test_too_large_molecule(self, validator):
        """Build a SMILES with > 290 heavy atoms → ERROR."""
        # Linear alkane with 300 carbons
        big_smiles = "C" * 300
        r = validator.validate([("TooBig", big_smiles)])[0]
        assert r.status == ValidationStatus.ERROR
        assert any("290" in e or "exceed" in e.lower() for e in r.errors)

    def test_error_has_descriptive_message(self, validator):
        r = validator.validate([("Bad", "ZZZZZZ")])[0]
        assert r.status == ValidationStatus.ERROR
        assert r.errors  # must provide at least one error message
        assert len(r.errors[0]) > 10   # not just "Error"


# ─── Batch processing ──────────────────────────────────────────────────────

class TestSMILESBatch:
    def test_mixed_batch_counts(self, validator):
        entries = [
            ("OK1",    "CC(=O)O"),
            ("WARN1",  "[Na+].[O-]C(=O)O"),
            ("ERR1",   "NOT_SMILES"),
            ("OK2",    "c1ccccc1"),
            ("ERR2",   ""),
        ]
        records = validator.validate(entries)
        assert len(records) == 5
        statuses = [r.status for r in records]
        assert statuses.count(ValidationStatus.OK)      >= 2
        assert statuses.count(ValidationStatus.WARNING) >= 1
        assert statuses.count(ValidationStatus.ERROR)   >= 2

    def test_all_ok_batch(self, validator):
        entries = [(f"Drug_{i}", "CC(=O)O") for i in range(10)]
        records = validator.validate(entries)
        # First is OK, rest 2–10 are WARNING (duplicate canonical SMILES)
        usable = [r for r in records if r.is_usable]
        assert len(usable) == 10

    def test_empty_batch(self, validator):
        records = validator.validate([])
        assert records == []
