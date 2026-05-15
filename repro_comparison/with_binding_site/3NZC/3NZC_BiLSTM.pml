# PyMOL Script - 3NZC BiLSTM
# Attention frequency (n=7 ligands)
# RED>75%  ORANGE 50-75%  YELLOW 25-50%
# BLUE = direct contact <=5A  MAGENTA = RED+contact
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 3nzc_protein.pdb, protein
load 3nzc_ligand.sdf,  ligand
hide everything, all
show cartoon, protein
color gray70, protein
show sticks, ligand
color cyan, ligand

select attn_red, (protein and resi 6+7+13+16+18+19+20+21+22+23+24+25+26+27+28+29+113+116+126+127+128+129+130+131+132+133+134+135)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 1+3+4+5+30+114+136+138)
color orange, attn_orange
show sticks, attn_orange

select attn_yellow, (protein and resi 31+32+33+34+125)
color yellow, attn_yellow
show sticks, attn_yellow

select binding_site, (protein and resi 10+11+12+24+25+30+32+33+34+35+36+37+61+64+65+66+68+69+72+75+122+123+124+129+142+143+144+199)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 3NZC BiLSTM ==="
print "RED    (28): [6, 7, 13, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 113, 116, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135]"
print "Contact (28): [10, 11, 12, 24, 25, 30, 32, 33, 34, 35, 36, 37, 61, 64, 65, 66, 68, 69, 72, 75, 122, 123, 124, 129, 142, 143, 144, 199]"
