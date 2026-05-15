# PyMOL Script - 3NZC CNN Attention (no binding site marked)
# Use case: unknown protein / AlphaFold3 structure
#
# Color scheme:
#   RED    : Attention > 75%  (most drugs consistently focus here)
#   ORANGE : Attention 50-75%
#   YELLOW : Attention 25-50%
#   GRAY   : Other residues
#
# NOTE: CNN attention is window-based (not true per-residue).
# Typically shows scattered RED with no clear cluster -> hard to interpret.
#
# Usage: File -> Run Script (keep 3nzc_protein.pdb in same folder)

reinitialize
load 3nzc_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 3nzc_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (18 residues)
select attn_red, (protein and resi 19+20+29+30+47+48+93+104+112+132+158+168+176+178+195+204+205+206)
color red, attn_red
show sticks, attn_red

zoom
bg_color white
