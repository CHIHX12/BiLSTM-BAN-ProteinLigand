# PyMOL Script - 5HLS BiLSTM Attention (no binding site marked)
# Use case: unknown protein / AlphaFold3 structure
#
# Color scheme:
#   RED    : Attention > 75%  (most drugs consistently focus here)
#   ORANGE : Attention 50-75%
#   YELLOW : Attention 25-50%
#   GRAY   : Other residues
#
# Interpretation rule:
#   Continuous cluster of RED/ORANGE/YELLOW = high-confidence active site
#   Isolated scattered RED = noise, ignore
#
# Usage: File -> Run Script (keep 5hls_protein.pdb in same folder)

reinitialize
load 5hls_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 5hls_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (22 residues)
select attn_red, (protein and resi 42+43+44+45+46+66+67+68+69+70+71+72+73+74+75+76+77+78+79+80+81+82)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (1 residues)
select attn_orange, (protein and resi 47)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (1 residues)
select attn_yellow, (protein and resi 55)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
