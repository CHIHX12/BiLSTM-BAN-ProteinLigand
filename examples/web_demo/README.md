# TEIBAN web demo data

Sample inputs for trying the web UI (menu [5] -> teiban_web.py):

- `ligands.smi`   6 drugs as `ID <tab> SMILES` (aspirin, caffeine, ibuprofen,
                  imatinib, gefitinib, sorafenib)
- `targets.fasta` 2 protein targets (CDK2, A2AR) as a multi-chain FASTA

In the web page: browse into this folder, pick `ligands.smi` as the SMILES file
and `targets.fasta` as the protein file, then submit. That screens
6 drugs x 2 targets = 12 pairs on the cluster.
