# PyMOL Script - 6G3Q BiLSTM + fpocket
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
load 6g3q_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 6g3q_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: BiLSTM RED + fpocket consensus (8 residues)
select consensus, (protein and resi 5+7+11+15+16+18+20+165)
color magenta, consensus
show sticks, consensus

# RED: BiLSTM attention only (32 residues)
select attn_only, (protein and resi 1+6+8+9+10+12+13+14+17+28+81+83+84+86+98+99+100+101+102+103+104+105+144+145+146+147+148+149+150+151+152+156)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (43 residues)
select fpocket_only, (protein and resi 3+4+19+32+34+37+38+39+62+64+65+67+92+94+96+121+131+143+161+162+164+168+188+190+192+198+199+200+201+202+209+213+214+225+228+229+238+240+251+254+255+256+260)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (6 residues)
select attn_orange, (protein and resi 2+19+29+82+164+185)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (14 residues)
select attn_yellow, (protein and resi 1+5+20+28+81+83+84+86+146+147+149+150+156+165)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 6G3Q BiLSTM+fpocket ==="
print "MAGENTA consensus: 8 residues"
print "RED attn only    : 32 residues"
print "BLUE fpocket only: 43 residues"
