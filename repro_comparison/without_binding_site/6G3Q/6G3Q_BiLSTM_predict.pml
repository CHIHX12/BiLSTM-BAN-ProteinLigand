# PyMOL Script - 6G3Q BiLSTM Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 6g3q_protein.pdb in same folder)

reinitialize
load 6g3q_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 6g3q_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (26 residues)
select attn_red, (protein and resi 6+7+8+9+10+11+12+13+14+15+16+17+18+98+99+100+101+102+103+104+105+144+145+148+151+152)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (6 residues)
select attn_orange, (protein and resi 2+19+29+82+164+185)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (14 residues)
select attn_yellow, (protein and resi 1+5+20+28+81+83+84+86+146+147+149+150+156+165)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
