# PyMOL Script - 1T48 CNN Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 1t48_protein.pdb in same folder)

reinitialize
load 1t48_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1t48_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (28 residues)
select attn_red, (protein and resi 10+12+29+30+39+56+67+75+93+94+123+131+158+167+169+178+186+188+195+205+206+214+232+241+243+252+289+297)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (1 residues)
select attn_orange, (protein and resi 132)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (3 residues)
select attn_yellow, (protein and resi 19+84+271)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
