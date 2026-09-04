# TEIBAN multi-file screen/clean demo

Four SMILES files (id <tab> SMILES), ~239 lines each, on purpose "messy":
- many distinct molecules, plus repeats WITHIN and ACROSS files
- some salt/solvent forms (.[Na+], .[Cl-], .O, .[K+])
- a charged molecule (C[NH3+]) to show polarity is PRESERVED
- a few junk lines to show invalids are dropped

Try in the web UI:
1. Browse into this folder, click +SMILES on all four files.
2. Press "Preprocess (clean library)" -> the four files become ONE clean,
   de-duplicated library (id<tab>SMILES) + a .report.txt. Example result:
   956 in -> ~372 clean unique (32 invalid, ~552 duplicates removed; charge kept).
3. Pick the resulting clean_library.smi + paste a protein -> Submit screen.

Or skip step 2 and Submit screen directly on the four files (it cleans inline).
