# PyMOL Script - 1AQ1 BiLSTM Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 1aq1_protein.pdb in same folder)

reinitialize
load 1aq1_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1aq1_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (49 residues)
select attn_red, (protein and resi 55+58+78+79+80+81+82+83+84+85+86+87+88+89+90+91+92+93+94+95+96+97+98+99+100+101+102+103+104+105+106+108+109+111+112+120+122+123+124+125+126+127+128+202+203+204+205+206+207)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (5 residues)
select attn_orange, (protein and resi 56+57+59+67+114)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (7 residues)
select attn_yellow, (protein and resi 37+61+110+113+121+134+208)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
