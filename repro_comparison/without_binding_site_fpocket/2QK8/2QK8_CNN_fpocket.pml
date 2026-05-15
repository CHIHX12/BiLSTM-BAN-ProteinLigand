# PyMOL Script - 2QK8 CNN + fpocket
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
load 2qk8_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 2qk8_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: CNN RED + fpocket consensus (3 residues)
select consensus, (protein and resi 29+66+131)
color magenta, consensus
show sticks, consensus

# RED: CNN attention only (7 residues)
select attn_only, (protein and resi 48+75+84+93+94+140+159)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (63 residues)
select fpocket_only, (protein and resi 6+7+8+13+15+16+17+18+19+20+21+23+24+25+27+28+30+31+32+33+34+37+44+45+46+47+50+51+53+55+56+58+63+64+65+78+79+80+96+98+99+100+101+102+103+104+106+107+109+114+116+124+125+129+130+132+134+135+136+137+155+158+160)
color blue, fpocket_only
show sticks, fpocket_only

zoom
bg_color white

print "=== 2QK8 CNN+fpocket ==="
print "MAGENTA consensus: 3 residues"
print "RED attn only    : 7 residues"
print "BLUE fpocket only: 63 residues"
