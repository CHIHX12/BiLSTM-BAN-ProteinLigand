# PyMOL Script - 1AQ1 CNN + fpocket
#
# Color scheme:
#   MAGENTA : Consensus = CNN RED and fpocket  (strongest signal)
#   RED     : CNN attention only (>75%, not in fpocket)
#   BLUE    : fpocket only (not in CNN RED)
#   ORANGE  : CNN attention 50-75%
#   YELLOW  : CNN attention 25-50%
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

# MAGENTA: CNN RED + fpocket consensus (5 residues)
select consensus, (protein and resi 10+12+84+94+131)
color magenta, consensus
show sticks, consensus

# RED: CNN attention only (26 residues)
select attn_only, (protein and resi 19+20+21+40+48+57+67+75+104+122+123+141+149+151+159+168+177+187+205+214+215+223+225+232+242+297)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (61 residues)
select fpocket_only, (protein and resi 8+11+13+18+31+33+52+55+58+63+64+66+78+80+81+82+83+85+86+88+89+90+91+92+93+95+97+98+99+100+102+103+116+119+120+132+134+143+144+145+146+167+181+182+183+186+195+196+199+200+201+271+272+274+275+276+277+278+292+293+295)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (1 residues)
select attn_orange, (protein and resi 93)
color orange, attn_orange
show sticks, attn_orange

zoom
bg_color white

print "=== 1AQ1 CNN+fpocket ==="
print "MAGENTA consensus: 5 residues"
print "RED attn only    : 26 residues"
print "BLUE fpocket only: 61 residues"
