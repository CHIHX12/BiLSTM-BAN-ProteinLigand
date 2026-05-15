# PyMOL Script - 1OXG CNN Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 1oxg_protein.pdb in same folder)

reinitialize
load 1oxg_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1oxg_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (19 residues)
select attn_red, (protein and resi 2+28+38+39+56+65+66+67+75+85+93+130+132+140+141+159+168+232+233)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (3 residues)
select attn_orange, (protein and resi 150+160+234)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (3 residues)
select attn_yellow, (protein and resi 29+104+187)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
