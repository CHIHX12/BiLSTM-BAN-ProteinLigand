# PyMOL Script - 1OXG BiLSTM + fpocket
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
load 1oxg_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1oxg_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: BiLSTM RED + fpocket consensus (1 residues)
select consensus, (protein and resi 10)
color magenta, consensus
show sticks, consensus

# RED: BiLSTM attention only (28 residues)
select attn_only, (protein and resi 1+11+12+13+14+16+17+18+19+20+21+23+24+25+26+106+108+109+113+114+115+122+124+159+160+161+163+164)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (53 residues)
select fpocket_only, (protein and resi 2+3+4+5+7+8+34+38+39+40+57+71+72+76+77+80+82+97+98+99+129+130+131+134+150+151+152+172+174+175+177+189+190+191+192+195+201+202+203+205+207+213+214+215+216+217+218+220+224+225+226+227+228)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (8 residues)
select attn_orange, (protein and resi 2+22+116+119+123+125+127+162)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (7 residues)
select attn_yellow, (protein and resi 1+17+18+106+122+124+161)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 1OXG BiLSTM+fpocket ==="
print "MAGENTA consensus: 1 residues"
print "RED attn only    : 28 residues"
print "BLUE fpocket only: 53 residues"
