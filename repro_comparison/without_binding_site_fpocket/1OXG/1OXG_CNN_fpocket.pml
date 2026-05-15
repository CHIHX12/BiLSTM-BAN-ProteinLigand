# PyMOL Script - 1OXG CNN + fpocket
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
load 1oxg_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1oxg_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: CNN RED + fpocket consensus (4 residues)
select consensus, (protein and resi 2+38+39+130)
color magenta, consensus
show sticks, consensus

# RED: CNN attention only (15 residues)
select attn_only, (protein and resi 28+56+65+66+67+75+85+93+132+140+141+159+168+232+233)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (50 residues)
select fpocket_only, (protein and resi 3+4+5+7+8+10+34+40+57+71+72+76+77+80+82+97+98+99+129+131+134+150+151+152+172+174+175+177+189+190+191+192+195+201+202+203+205+207+213+214+215+216+217+218+220+224+225+226+227+228)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (3 residues)
select attn_orange, (protein and resi 150+160+234)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (3 residues)
select attn_yellow, (protein and resi 29+104+187)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 1OXG CNN+fpocket ==="
print "MAGENTA consensus: 4 residues"
print "RED attn only    : 15 residues"
print "BLUE fpocket only: 50 residues"
