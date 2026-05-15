# PyMOL Script - 6G3Q CNN Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 6g3q_protein.pdb in same folder)

reinitialize
load 6g3q_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 6g3q_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (20 residues)
select attn_red, (protein and resi 10+19+30+39+56+58+66+67+75+103+104+113+151+160+178+179+198+206+215+234)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (4 residues)
select attn_orange, (protein and resi 65+141+233+244)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (2 residues)
select attn_yellow, (protein and resi 85+112)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
