# PyMOL Script - 3NZC CNN + fpocket
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
load 3nzc_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 3nzc_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: CNN RED + fpocket consensus (5 residues)
select consensus, (protein and resi 19+20+30+93+195)
color magenta, consensus
show sticks, consensus

# RED: CNN attention only (13 residues)
select attn_only, (protein and resi 29+47+48+104+112+132+158+168+176+178+204+205+206)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (64 residues)
select fpocket_only, (protein and resi 10+11+12+21+23+24+25+26+27+28+31+32+33+34+35+36+37+38+58+59+60+61+62+63+64+65+66+69+72+73+75+79+80+81+82+83+84+85+86+87+88+92+94+96+97+98+99+123+124+125+126+127+128+129+131+134+135+136+154+186+190+193+197+199)
color blue, fpocket_only
show sticks, fpocket_only

zoom
bg_color white

print "=== 3NZC CNN+fpocket ==="
print "MAGENTA consensus: 5 residues"
print "RED attn only    : 13 residues"
print "BLUE fpocket only: 64 residues"
