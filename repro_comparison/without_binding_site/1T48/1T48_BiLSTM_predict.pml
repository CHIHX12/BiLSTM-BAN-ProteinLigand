# PyMOL Script - 1T48 BiLSTM Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 1t48_protein.pdb in same folder)

reinitialize
load 1t48_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1t48_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (59 residues)
select attn_red, (protein and resi 1+2+3+4+5+6+7+117+118+119+130+131+132+133+134+135+136+137+138+139+140+141+142+143+144+145+146+147+148+149+151+157+158+159+160+161+162+163+164+165+166+167+168+169+170+171+172+173+174+175+177+178+190+191+192+201+203+204+205)
color red, attn_red
show sticks, attn_red

# YELLOW: attention 25-50%  (6 residues)
select attn_yellow, (protein and resi 8+127+128+129+176+193)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
