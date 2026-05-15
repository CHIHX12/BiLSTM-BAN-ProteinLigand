# PyMOL Script - 1OXG BiLSTM Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 1oxg_protein.pdb in same folder)

reinitialize
load 1oxg_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1oxg_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (22 residues)
select attn_red, (protein and resi 10+11+12+13+14+16+19+20+21+23+24+25+26+108+109+113+114+115+159+160+163+164)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (8 residues)
select attn_orange, (protein and resi 2+22+116+119+123+125+127+162)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (7 residues)
select attn_yellow, (protein and resi 1+17+18+106+122+124+161)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
