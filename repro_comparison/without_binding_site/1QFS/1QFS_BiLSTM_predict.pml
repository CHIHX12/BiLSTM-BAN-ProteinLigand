# PyMOL Script - 1QFS BiLSTM Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 1qfs_protein.pdb in same folder)

reinitialize
load 1qfs_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1qfs_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (89 residues)
select attn_red, (protein and resi 1+2+3+4+5+6+7+8+104+105+106+107+108+113+114+115+116+119+120+127+129+131+132+133+134+135+136+137+138+139+140+141+142+143+144+145+146+147+148+149+150+151+152+153+154+155+156+159+160+174+192+193+194+195+196+197+198+199+200+201+202+203+215+216+220+248+249+250+251+270+272+369+371+372+386+387+388+393+394+395+397+398+399+400+401+402+403+404+405)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (3 residues)
select attn_orange, (protein and resi 157+158+175)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (5 residues)
select attn_yellow, (protein and resi 103+118+168+241+391)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
