# PyMOL Script - 1AQ1 BiLSTM + fpocket
#
# Color scheme:
#   MAGENTA : Consensus = BiLSTM RED and fpocket  (strongest signal)
#   RED     : BiLSTM attention only (>75%, not in fpocket)
#   BLUE    : fpocket only (not in BiLSTM RED)
#   ORANGE  : BiLSTM attention 50-75%
#   YELLOW  : BiLSTM attention 25-50%
#   CYAN    : Ligand
#   GRAY    : Other residues
#
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 1aq1_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1aq1_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: BiLSTM RED + fpocket consensus (26 residues)
select consensus, (protein and resi 55+58+78+80+81+82+83+84+85+86+88+89+90+91+92+93+94+95+97+98+99+100+102+103+120+134)
color magenta, consensus
show sticks, consensus

# RED: BiLSTM attention only (30 residues)
select attn_only, (protein and resi 37+61+79+87+96+101+104+105+106+108+109+110+111+112+113+121+122+123+124+125+126+127+128+202+203+204+205+206+207+208)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (40 residues)
select fpocket_only, (protein and resi 8+10+11+12+13+18+31+33+52+63+64+66+116+119+131+132+143+144+145+146+167+181+182+183+186+195+196+199+200+201+271+272+274+275+276+277+278+292+293+295)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (5 residues)
select attn_orange, (protein and resi 56+57+59+67+114)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (7 residues)
select attn_yellow, (protein and resi 37+61+110+113+121+134+208)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 1AQ1 BiLSTM+fpocket ==="
print "MAGENTA consensus: 26 residues"
print "RED attn only    : 30 residues"
print "BLUE fpocket only: 40 residues"
