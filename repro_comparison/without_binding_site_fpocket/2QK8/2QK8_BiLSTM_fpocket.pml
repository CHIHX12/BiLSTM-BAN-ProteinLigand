# PyMOL Script - 2QK8 BiLSTM + fpocket
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
load 2qk8_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 2qk8_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: BiLSTM RED + fpocket consensus (11 residues)
select consensus, (protein and resi 16+96+98+99+100+101+102+103+104+106+107)
color magenta, consensus
show sticks, consensus

# RED: BiLSTM attention only (7 residues)
select attn_only, (protein and resi 1+2+3+4+73+97+105)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (55 residues)
select fpocket_only, (protein and resi 6+7+8+13+15+17+18+19+20+21+23+24+25+27+28+29+30+31+32+33+34+37+44+45+46+47+50+51+53+55+56+58+63+64+65+66+78+79+80+109+114+116+124+125+129+130+131+132+134+135+136+137+155+158+160)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (2 residues)
select attn_orange, (protein and resi 5+79)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (2 residues)
select attn_yellow, (protein and resi 16+73)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 2QK8 BiLSTM+fpocket ==="
print "MAGENTA consensus: 11 residues"
print "RED attn only    : 7 residues"
print "BLUE fpocket only: 55 residues"
