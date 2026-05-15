# PyMOL Script - 1T48 CNN + fpocket
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
load 1t48_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1t48_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: CNN RED + fpocket consensus (11 residues)
select consensus, (protein and resi 75+158+169+178+186+188+232+241+252+289+297)
color magenta, consensus
show sticks, consensus

# RED: CNN attention only (17 residues)
select attn_only, (protein and resi 10+12+29+30+39+56+67+93+94+123+131+167+195+205+206+214+243)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (46 residues)
select fpocket_only, (protein and resi 1+2+3+73+74+78+79+80+145+152+157+159+168+170+176+180+185+187+189+190+192+193+196+197+200+235+236+240+242+244+256+276+277+278+279+280+281+282+288+291+292+293+294+295+296+298)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (1 residues)
select attn_orange, (protein and resi 132)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (3 residues)
select attn_yellow, (protein and resi 19+84+271)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 1T48 CNN+fpocket ==="
print "MAGENTA consensus: 11 residues"
print "RED attn only    : 17 residues"
print "BLUE fpocket only: 46 residues"
