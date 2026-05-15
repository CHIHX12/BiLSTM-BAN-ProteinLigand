# PyMOL Script - 3NZC BiLSTM Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 3nzc_protein.pdb in same folder)

reinitialize
load 3nzc_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 3nzc_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (28 residues)
select attn_red, (protein and resi 6+7+13+16+18+19+20+21+22+23+24+25+26+27+28+29+113+116+126+127+128+129+130+131+132+133+134+135)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (8 residues)
select attn_orange, (protein and resi 1+3+4+5+30+114+136+138)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (5 residues)
select attn_yellow, (protein and resi 31+32+33+34+125)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
