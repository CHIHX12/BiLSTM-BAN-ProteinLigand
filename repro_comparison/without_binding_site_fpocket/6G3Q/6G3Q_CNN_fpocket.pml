# PyMOL Script - 6G3Q CNN + fpocket
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
load 6g3q_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 6g3q_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: CNN RED + fpocket consensus (4 residues)
select consensus, (protein and resi 19+39+67+198)
color magenta, consensus
show sticks, consensus

# RED: CNN attention only (16 residues)
select attn_only, (protein and resi 10+30+56+58+66+75+103+104+113+151+160+178+179+206+215+234)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (47 residues)
select fpocket_only, (protein and resi 3+4+5+7+11+15+16+18+20+32+34+37+38+62+64+65+92+94+96+121+131+143+161+162+164+165+168+188+190+192+199+200+201+202+209+213+214+225+228+229+238+240+251+254+255+256+260)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (4 residues)
select attn_orange, (protein and resi 65+141+233+244)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (2 residues)
select attn_yellow, (protein and resi 85+112)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 6G3Q CNN+fpocket ==="
print "MAGENTA consensus: 4 residues"
print "RED attn only    : 16 residues"
print "BLUE fpocket only: 47 residues"
