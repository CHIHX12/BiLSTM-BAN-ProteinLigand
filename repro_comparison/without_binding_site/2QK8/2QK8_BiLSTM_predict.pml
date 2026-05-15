# PyMOL Script - 2QK8 BiLSTM Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 2qk8_protein.pdb in same folder)

reinitialize
load 2qk8_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 2qk8_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (16 residues)
select attn_red, (protein and resi 1+2+3+4+96+97+98+99+100+101+102+103+104+105+106+107)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (2 residues)
select attn_orange, (protein and resi 5+79)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (2 residues)
select attn_yellow, (protein and resi 16+73)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
