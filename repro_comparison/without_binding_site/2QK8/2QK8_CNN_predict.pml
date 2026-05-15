# PyMOL Script - 2QK8 CNN Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 2qk8_protein.pdb in same folder)

reinitialize
load 2qk8_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 2qk8_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (10 residues)
select attn_red, (protein and resi 29+48+66+75+84+93+94+131+140+159)
color red, attn_red
show sticks, attn_red

zoom
bg_color white
