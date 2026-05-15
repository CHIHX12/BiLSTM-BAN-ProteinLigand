# PyMOL Script - 3NZC BiLSTM + fpocket
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
load 3nzc_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 3nzc_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: BiLSTM RED + fpocket consensus (21 residues)
select consensus, (protein and resi 19+20+21+23+24+25+26+27+28+31+32+33+34+125+126+127+128+129+131+134+135)
color magenta, consensus
show sticks, consensus

# RED: BiLSTM attention only (12 residues)
select attn_only, (protein and resi 6+7+13+16+18+22+29+113+116+130+132+133)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (48 residues)
select fpocket_only, (protein and resi 10+11+12+30+35+36+37+38+58+59+60+61+62+63+64+65+66+69+72+73+75+79+80+81+82+83+84+85+86+87+88+92+93+94+96+97+98+99+123+124+136+154+186+190+193+195+197+199)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (8 residues)
select attn_orange, (protein and resi 1+3+4+5+30+114+136+138)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (5 residues)
select attn_yellow, (protein and resi 31+32+33+34+125)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 3NZC BiLSTM+fpocket ==="
print "MAGENTA consensus: 21 residues"
print "RED attn only    : 12 residues"
print "BLUE fpocket only: 48 residues"
