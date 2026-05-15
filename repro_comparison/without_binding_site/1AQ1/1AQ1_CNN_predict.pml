# PyMOL Script - 1AQ1 CNN Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 1aq1_protein.pdb in same folder)

reinitialize
load 1aq1_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1aq1_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (31 residues)
select attn_red, (protein and resi 10+12+19+20+21+40+48+57+67+75+84+94+104+122+123+131+141+149+151+159+168+177+187+205+214+215+223+225+232+242+297)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (1 residues)
select attn_orange, (protein and resi 93)
color orange, attn_orange
show sticks, attn_orange

zoom
bg_color white
